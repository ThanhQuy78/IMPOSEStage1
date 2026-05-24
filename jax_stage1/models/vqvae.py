"""VQ-VAE First Stage Model — Flax/Linen implementation.

Ported from the PyTorch IMPOSE codebase.  This model is loaded from a
pre-trained checkpoint and used **frozen** for encode / decode only —
it is never trained in the JAX pipeline.

Layout convention: **NHWC** (channels-last) everywhere.

Architecture config (from YAML):
    embed_dim: 3
    n_embed: 4096
    z_channels: 3
    resolution: 512
    in_channels: 1
    out_ch: 1
    ch: 64
    ch_mult: [1, 2, 4]
    num_res_blocks: 2
    attn_resolutions: [16]
    double_z: false
    dropout: 0.0
"""

from __future__ import annotations

from typing import Sequence, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

def nonlinearity(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU / Swish activation: x · σ(x)."""
    return x * jax.nn.sigmoid(x)


# ---------------------------------------------------------------------------
# ResnetBlock
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """Pre-norm residual block (GroupNorm → Swish → Conv).

    Attributes:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (defaults to *in_channels*).
        dropout:      Dropout probability (unused at inference).
        temb_channels: Timestep-embedding channels.  **0** for VQ-VAE.
    """

    in_channels: int
    out_channels: Optional[int] = None
    dropout: float = 0.0
    temb_channels: int = 0

    @nn.compact
    def __call__(self, x: jnp.ndarray, temb: Optional[jnp.ndarray] = None, deterministic: bool = True) -> jnp.ndarray:
        out_channels = self.out_channels if self.out_channels is not None else self.in_channels

        h = nn.GroupNorm(num_groups=32, name="norm1")(x)
        h = nonlinearity(h)
        h = nn.Conv(out_channels, kernel_size=(3, 3), padding="SAME", name="conv1")(h)

        if self.temb_channels > 0 and temb is not None:
            temb_proj = nn.Dense(out_channels, name="temb_proj")(nonlinearity(temb))
            # temb_proj shape: (B, out_channels) → broadcast over H, W
            h = h + temb_proj[:, None, None, :]

        h = nn.GroupNorm(num_groups=32, name="norm2")(h)
        h = nonlinearity(h)
        if self.dropout > 0.0 and not deterministic:
            h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        h = nn.Conv(out_channels, kernel_size=(3, 3), padding="SAME", name="conv2")(h)

        if self.in_channels != out_channels:
            x = nn.Conv(out_channels, kernel_size=(1, 1), name="nin_shortcut")(x)

        return x + h


# ---------------------------------------------------------------------------
# AttnBlock  (single-head self-attention, spatial)
# ---------------------------------------------------------------------------

class AttnBlock(nn.Module):
    """Channel-wise self-attention over spatial dimensions.

    Attributes:
        in_channels: Number of channels (= query/key/value dim).
    """

    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape

        h = nn.GroupNorm(num_groups=32, name="norm")(x)

        q = nn.Conv(self.in_channels, kernel_size=(1, 1), name="q")(h)  # (B,H,W,C)
        k = nn.Conv(self.in_channels, kernel_size=(1, 1), name="k")(h)
        v = nn.Conv(self.in_channels, kernel_size=(1, 1), name="v")(h)

        # Flatten spatial dims → (B, H*W, C)
        q = q.reshape(B, -1, C)
        k = k.reshape(B, -1, C)
        v = v.reshape(B, -1, C)

        # Scaled dot-product attention
        scale = C ** -0.5
        w = jnp.einsum("bic,bjc->bij", q, k) * scale       # (B, HW, HW)
        w = jax.nn.softmax(w, axis=-1)

        h = jnp.einsum("bij,bjc->bic", w, v)                # (B, HW, C)
        h = h.reshape(B, H, W, C)

        h = nn.Conv(self.in_channels, kernel_size=(1, 1), name="proj_out")(h)
        return x + h


# ---------------------------------------------------------------------------
# Downsample / Upsample
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    """Spatial 2× downsample: asymmetric pad + stride-2 conv.

    Matches PyTorch ``F.pad(x, (0,1,0,1))`` followed by stride-2 conv.

    Attributes:
        in_channels: Number of channels.
    """

    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Asymmetric padding: (0,1) on H and W (right/bottom)
        # Pad spec per axis: (batch, H, W, C)
        x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
        x = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            name="conv",
        )(x)
        return x


class Upsample(nn.Module):
    """Spatial 2× upsample: nearest-neighbor resize + conv.

    Attributes:
        in_channels: Number of channels.
    """

    in_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jax.image.resize(x, shape=(B, H * 2, W * 2, C), method="nearest")
        x = nn.Conv(
            self.in_channels,
            kernel_size=(3, 3),
            padding="SAME",
            name="conv",
        )(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """VQ-VAE Encoder.

    Downsamples the input image and produces a continuous latent map.

    Attributes:
        ch:               Base channel count.
        out_ch:           Output image channels (unused inside Encoder).
        ch_mult:          Per-level channel multipliers.
        num_res_blocks:   Number of ResnetBlocks per down-level.
        attn_resolutions: Spatial resolutions at which to insert attention.
        dropout:          Dropout rate.
        in_channels:      Input image channels.
        resolution:       Input spatial resolution.
        z_channels:       Latent channel count.
        double_z:         If True, output 2×z_channels (for VAE reparametrization).
    """

    ch: int = 64
    out_ch: int = 1
    ch_mult: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Sequence[int] = (16,)
    dropout: float = 0.0
    in_channels: int = 1
    resolution: int = 512
    z_channels: int = 3
    double_z: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        num_resolutions = len(self.ch_mult)
        in_ch_mult = (1,) + tuple(self.ch_mult)  # (1, 1, 2, 4)

        # ---- initial convolution ----
        h = nn.Conv(self.ch, kernel_size=(3, 3), padding="SAME", name="conv_in")(x)

        # ---- downsampling ----
        curr_res = self.resolution
        for i_level in range(num_resolutions):
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                h = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=self.dropout,
                    temb_channels=0,
                    name=f"down_{i_level}_block_{i_block}",
                )(h, deterministic=deterministic)
                block_in = block_out

                if curr_res in self.attn_resolutions:
                    h = AttnBlock(
                        in_channels=block_in,
                        name=f"down_{i_level}_attn_{i_block}",
                    )(h)

            if i_level != num_resolutions - 1:
                h = Downsample(
                    in_channels=block_in,
                    name=f"down_{i_level}_downsample",
                )(h)
                curr_res //= 2

        # ---- mid ----
        h = ResnetBlock(block_in, block_in, dropout=self.dropout, temb_channels=0, name="mid_block_1")(h, deterministic=deterministic)
        h = AttnBlock(block_in, name="mid_attn_1")(h)
        h = ResnetBlock(block_in, block_in, dropout=self.dropout, temb_channels=0, name="mid_block_2")(h, deterministic=deterministic)

        # ---- output ----
        h = nn.GroupNorm(num_groups=32, name="norm_out")(h)
        h = nonlinearity(h)
        out_channels = 2 * self.z_channels if self.double_z else self.z_channels
        h = nn.Conv(out_channels, kernel_size=(3, 3), padding="SAME", name="conv_out")(h)
        return h


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """VQ-VAE Decoder.

    Upsamples the quantized latent map back to image space.

    Attributes:
        ch:               Base channel count.
        out_ch:           Output image channels.
        ch_mult:          Per-level channel multipliers.
        num_res_blocks:   Number of ResnetBlocks per up-level.
        attn_resolutions: Spatial resolutions at which to insert attention.
        dropout:          Dropout rate.
        in_channels:      (unused — kept for config symmetry with Encoder).
        resolution:       Output spatial resolution.
        z_channels:       Latent channel count.
        double_z:         (unused — kept for config symmetry).
    """

    ch: int = 64
    out_ch: int = 1
    ch_mult: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Sequence[int] = (16,)
    dropout: float = 0.0
    in_channels: int = 1
    resolution: int = 512
    z_channels: int = 3
    double_z: bool = False

    @nn.compact
    def __call__(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        num_resolutions = len(self.ch_mult)
        block_in = self.ch * self.ch_mult[-1]  # 64 * 4 = 256
        curr_res = self.resolution // (2 ** (num_resolutions - 1))  # 512 // 4 = 128

        # ---- initial convolution (z_channels → block_in) ----
        h = nn.Conv(block_in, kernel_size=(3, 3), padding="SAME", name="conv_in")(z)

        # ---- mid ----
        h = ResnetBlock(block_in, block_in, dropout=self.dropout, temb_channels=0, name="mid_block_1")(h, deterministic=deterministic)
        h = AttnBlock(block_in, name="mid_attn_1")(h)
        h = ResnetBlock(block_in, block_in, dropout=self.dropout, temb_channels=0, name="mid_block_2")(h, deterministic=deterministic)

        # ---- upsampling ----
        for i_level in reversed(range(num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks + 1):  # 3 ResBlocks
                h = ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    dropout=self.dropout,
                    temb_channels=0,
                    name=f"up_{i_level}_block_{i_block}",
                )(h, deterministic=deterministic)
                block_in = block_out

                if curr_res in self.attn_resolutions:
                    h = AttnBlock(
                        in_channels=block_in,
                        name=f"up_{i_level}_attn_{i_block}",
                    )(h)

            if i_level != 0:
                h = Upsample(
                    in_channels=block_in,
                    name=f"up_{i_level}_upsample",
                )(h)
                curr_res *= 2

        # ---- output ----
        h = nn.GroupNorm(num_groups=32, name="norm_out")(h)
        h = nonlinearity(h)
        h = nn.Conv(self.out_ch, kernel_size=(3, 3), padding="SAME", name="conv_out")(h)
        return h


# ---------------------------------------------------------------------------
# VectorQuantizer
# ---------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """Codebook quantizer with straight-through gradient estimator.

    Attributes:
        n_embed:   Number of codebook entries.
        embed_dim: Dimensionality of each codebook vector.
        beta:      Commitment loss weight.
    """

    n_embed: int = 4096
    embed_dim: int = 3
    beta: float = 0.25

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple]:
        """Quantize continuous latent *z*.

        Args:
            z: Continuous latent of shape ``(B, H, W, embed_dim)``.

        Returns:
            z_q:  Quantized latent, same shape as *z*. Straight-through.
            loss: Commitment + codebook loss scalar.
            info: ``(None, None, min_encoding_indices)`` for compatibility.
        """
        embedding = self.param(
            "embedding",
            nn.initializers.lecun_uniform(),
            (self.n_embed, self.embed_dim),
        )

        # Flatten spatial dims
        z_flat = z.reshape(-1, self.embed_dim)  # (B*H*W, D)

        # Pairwise L2 distances  d_{ij} = ||z_i - e_j||^2
        #   = sum(z^2, axis=1, keepdims) + sum(e^2, axis=1) - 2 z @ e^T
        d = (
            jnp.sum(z_flat ** 2, axis=1, keepdims=True)
            + jnp.sum(embedding ** 2, axis=1)
            - 2.0 * jnp.dot(z_flat, embedding.T)
        )  # (B*H*W, n_embed)

        min_encoding_indices = jnp.argmin(d, axis=-1)           # (B*H*W,)
        z_q = embedding[min_encoding_indices].reshape(z.shape)  # (B,H,W,D)

        # Losses (not used at frozen inference, but kept for completeness)
        loss = self.beta * jnp.mean((jax.lax.stop_gradient(z_q) - z) ** 2) + jnp.mean(
            (z_q - jax.lax.stop_gradient(z)) ** 2
        )

        # Straight-through estimator
        z_q = z + jax.lax.stop_gradient(z_q - z)

        return z_q, loss, (None, None, min_encoding_indices)


# ---------------------------------------------------------------------------
# VQModelInterface  (top-level model)
# ---------------------------------------------------------------------------

class VQModelInterface(nn.Module):
    """Top-level VQ-VAE with separate encode / decode entry-points.

    The ``encode`` method returns the **continuous** (pre-quantization) latent
    — this matches the original PyTorch ``VQModelInterface.encode``.

    Attributes:
        embed_dim:        Codebook embedding dimension.
        n_embed:          Number of codebook entries.
        encoder_config:   Keyword dict forwarded to :class:`Encoder`.
        decoder_config:   Keyword dict forwarded to :class:`Decoder`.
    """

    embed_dim: int = 3
    n_embed: int = 4096
    ch: int = 64
    out_ch: int = 1
    ch_mult: Sequence[int] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Sequence[int] = (16,)
    dropout: float = 0.0
    in_channels: int = 1
    resolution: int = 512
    z_channels: int = 3
    double_z: bool = False

    def setup(self):
        shared = dict(
            ch=self.ch,
            out_ch=self.out_ch,
            ch_mult=self.ch_mult,
            num_res_blocks=self.num_res_blocks,
            attn_resolutions=self.attn_resolutions,
            dropout=self.dropout,
            in_channels=self.in_channels,
            resolution=self.resolution,
            z_channels=self.z_channels,
            double_z=self.double_z,
        )
        self.encoder = Encoder(**shared, name="encoder")
        self.decoder = Decoder(**shared, name="decoder")
        self.quantize = VectorQuantizer(
            n_embed=self.n_embed,
            embed_dim=self.embed_dim,
            name="quantize",
        )
        self.quant_conv = nn.Conv(self.embed_dim, kernel_size=(1, 1), name="quant_conv")
        self.post_quant_conv = nn.Conv(self.z_channels, kernel_size=(1, 1), name="post_quant_conv")

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Full forward: encode → quantize → decode.

        Returns:
            dec:  Reconstructed image ``(B, 512, 512, 1)``.
            loss: VQ commitment + codebook loss scalar.
        """
        h = self.encode(x)
        quant, emb_loss, _ = self.quantize(h)
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, deterministic=deterministic)
        return dec, emb_loss

    def encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode to **continuous** (pre-quantization) latent.

        Args:
            x: Input image ``(B, 512, 512, 1)``.

        Returns:
            Continuous latent ``(B, 128, 128, 3)``.
        """
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h: jnp.ndarray, force_not_quantize: bool = False, deterministic: bool = True) -> jnp.ndarray:
        """Decode from latent (optionally quantizing first).

        Args:
            h: Latent tensor ``(B, 128, 128, 3)``.
            force_not_quantize: If True, skip codebook lookup.
            deterministic: Passed to Decoder.

        Returns:
            Reconstructed image ``(B, 512, 512, 1)``.
        """
        if not force_not_quantize:
            quant, _emb_loss, _info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, deterministic=deterministic)
        return dec
