"""
UNet diffusion model for IMPOSE Stage 1 — JAX / Flax (linen).

Faithful port of ``ldm.modules.diffusionmodules.openaimodel.UNetModel``
configured for the IMPOSE rolled-fingerprint latent diffusion setting::

    image_size          = 128
    in_channels         = 3
    out_channels        = 3
    model_channels      = 64
    channel_mult        = [1, 2, 4]
    attention_resolutions = [32, 16, 8]  (no enc/dec attention with this mult)
    num_res_blocks      = 2
    num_heads           = 8
    use_scale_shift_norm = True
    resblock_updown     = True
    dropout             = 0.0

Layout convention: **NHWC** (channels-last) throughout.  All spatial
tensors have shape ``(B, H, W, C)``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------------------------------------------------------------------------
# Timestep embedding (pure function)
# ---------------------------------------------------------------------------

def timestep_embedding(
    timesteps: jnp.ndarray,
    dim: int,
    max_period: int = 10000,
) -> jnp.ndarray:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: (B,) integer or float array of timestep indices.
        dim:       Embedding dimension (``model_channels``).
        max_period: Controls the minimum frequency.

    Returns:
        (B, dim) float32 embedding.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(0, half, dtype=jnp.float32)
        / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )
    return embedding


# ---------------------------------------------------------------------------
# Upsample / Downsample helpers
# ---------------------------------------------------------------------------

class Upsample(nn.Module):
    """2× nearest-neighbour upsample, optionally followed by a 3×3 conv.

    Operates on NHWC tensors.
    """

    channels: int
    use_conv: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        x = jax.image.resize(
            x,
            shape=(B, H * 2, W * 2, C),
            method="nearest",
        )
        if self.use_conv:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                padding="SAME",
                name="conv",
            )(x)
        return x


class Downsample(nn.Module):
    """2× strided convolution downsample (PyTorch ``padding=1`` equivalent).

    With ``use_conv=True`` uses a 3×3 stride-2 conv;
    otherwise uses 2×2 average pooling.
    """

    channels: int
    use_conv: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_conv:
            # PyTorch Conv2d(k=3, s=2, p=1) → explicit pad ((1,1),(1,1))
            x = nn.Conv(
                features=self.channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=((1, 1), (1, 1)),
                name="op",
            )(x)
        else:
            # Average pooling fallback (no learnable params)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


# ---------------------------------------------------------------------------
# Lightweight up/down helpers used *inside* ResBlock (use_conv=False)
# ---------------------------------------------------------------------------

class _NearestUpsample2x(nn.Module):
    """2× nearest upsample with **no** convolution (for resblock_updown)."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, H, W, C = x.shape
        return jax.image.resize(
            x, shape=(B, H * 2, W * 2, C), method="nearest"
        )


class _AvgPoolDownsample2x(nn.Module):
    """2× average-pool downsample with **no** convolution."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))


# ---------------------------------------------------------------------------
# ResBlock with FiLM-style scale-shift-norm conditioning
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block conditioned on a timestep embedding.

    Matches ``ldm.modules.diffusionmodules.openaimodel.ResBlock`` with
    ``use_scale_shift_norm=True`` and NHWC layout.

    Attributes:
        channels:               Input channel count.
        emb_channels:           Timestep-embedding dimension (``time_embed_dim``).
        dropout:                Dropout rate (0.0 = no dropout).
        out_channels:           Output channels (defaults to ``channels``).
        use_conv:               Use 3×3 conv (vs 1×1) for skip projection.
        use_scale_shift_norm:   FiLM conditioning (True in IMPOSE config).
        up:                     If True, 2× upsample inside this block.
        down:                   If True, 2× downsample inside this block.
    """

    channels: int
    emb_channels: int
    dropout: float = 0.0
    out_channels: Optional[int] = None
    use_conv: bool = False
    use_scale_shift_norm: bool = True
    up: bool = False
    down: bool = False

    def setup(self):
        out_ch = self.out_channels if self.out_channels is not None else self.channels

        # --- in_layers ---
        self.in_norm = nn.GroupNorm(num_groups=32, name="in_norm")
        self.in_conv = nn.Conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding="SAME",
            name="in_conv",
        )

        # --- up / down ops (applied between norm+act and conv) ---
        if self.up:
            self.h_upd = _NearestUpsample2x(name="h_upd")
            self.x_upd = _NearestUpsample2x(name="x_upd")
        elif self.down:
            self.h_upd = _AvgPoolDownsample2x(name="h_upd")
            self.x_upd = _AvgPoolDownsample2x(name="x_upd")

        # --- emb_layers ---
        emb_out_dim = 2 * out_ch if self.use_scale_shift_norm else out_ch
        self.emb_dense = nn.Dense(features=emb_out_dim, name="emb_dense")

        # --- out_layers ---
        self.out_norm = nn.GroupNorm(num_groups=32, name="out_norm")
        self.dropout_layer = nn.Dropout(rate=self.dropout, name="dropout")
        self.out_conv = nn.Conv(
            features=out_ch,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="out_conv",
        )

        # --- skip_connection ---
        if out_ch != self.channels:
            if self.use_conv:
                self.skip_connection = nn.Conv(
                    features=out_ch,
                    kernel_size=(3, 3),
                    padding="SAME",
                    name="skip_connection",
                )
            else:
                self.skip_connection = nn.Conv(
                    features=out_ch,
                    kernel_size=(1, 1),
                    name="skip_connection",
                )
        # When out_ch == channels we use identity (no submodule needed).

    def __call__(
        self,
        x: jnp.ndarray,
        emb: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x:   (B, H, W, C) input feature map.
            emb: (B, emb_channels) timestep embedding.
            deterministic: If True, disable dropout.

        Returns:
            (B, H', W', out_channels) output feature map.
        """
        out_ch = self.out_channels if self.out_channels is not None else self.channels

        # --- in_layers (+ optional up/down) ---
        if self.up or self.down:
            h = self.in_norm(x)
            h = nn.silu(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
        else:
            h = self.in_norm(x)
            h = nn.silu(h)
            h = self.in_conv(h)

        # --- emb_layers ---
        emb_out = nn.silu(emb)
        emb_out = self.emb_dense(emb_out)        # (B, 2*out_ch or out_ch)
        emb_out = emb_out[:, None, None, :]       # (B, 1, 1, C) for broadcast

        # --- out_layers ---
        if self.use_scale_shift_norm:
            scale, shift = jnp.split(emb_out, 2, axis=-1)
            h = self.out_norm(h) * (1.0 + scale) + shift
            h = nn.silu(h)
            h = self.dropout_layer(h, deterministic=deterministic)
            h = self.out_conv(h)
        else:
            h = h + emb_out
            h = self.out_norm(h)
            h = nn.silu(h)
            h = self.dropout_layer(h, deterministic=deterministic)
            h = self.out_conv(h)

        # --- skip_connection ---
        if out_ch != self.channels:
            return self.skip_connection(x) + h
        else:
            return x + h


# ---------------------------------------------------------------------------
# Self-Attention (QKVAttentionLegacy style)
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    """Multi-head self-attention with ``QKVAttentionLegacy`` reshaping order.

    In the legacy order the QKV projection output ``(B, N, 3C)`` is first
    reshaped so that heads are split *before* the Q/K/V split::

        (B, N, 3C) → (B*n_heads, N, 3*dim_head) → split → q,k,v (B*n_heads, N, dim_head)

    The ``proj_out`` dense layer is zero-initialised so that at init
    the block acts as an identity (residual = 0).

    Attributes:
        channels:  Number of feature channels (last axis of NHWC tensor).
        num_heads: Number of attention heads.
    """

    channels: int
    num_heads: int = 8

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, name="norm")
        self.qkv_dense = nn.Dense(
            features=self.channels * 3,
            name="qkv",
        )
        self.proj_out = nn.Dense(
            features=self.channels,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="proj_out",
        )

    def __call__(
        self,
        x: jnp.ndarray,
        emb: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Self-attention. ``emb`` and ``deterministic`` are accepted
        (and ignored) so that the block can be called with the same
        signature as :class:`ResBlock`.
        """
        B, H, W, C = x.shape
        num_heads = self.num_heads
        dim_head = C // num_heads

        # Flatten spatial dims
        h = self.norm(x)                              # (B, H, W, C)
        h = h.reshape(B, H * W, C)                    # (B, N, C)

        # QKV projection
        qkv = self.qkv_dense(h)                       # (B, N, 3C)

        # Legacy reshape: split heads BEFORE splitting q/k/v
        # (B, N, 3C) → (B*n_heads, N, 3*dim_head) → split dim_head chunks
        qkv = qkv.reshape(B * num_heads, H * W, 3 * dim_head)
        q, k, v = jnp.split(qkv, 3, axis=-1)          # each (B*nh, N, dh)

        # Scaled dot-product attention
        # Use sqrt(sqrt(dim_head)) scaling on both q and k for fp16 stability,
        # matching the PyTorch ``QKVAttentionLegacy`` implementation.
        scale = dim_head ** -0.25
        weight = jnp.einsum(
            "bic,bjc->bij", q * scale, k * scale
        )                                              # (B*nh, N, N)
        weight = jax.nn.softmax(weight, axis=-1)

        # Aggregate values
        h = jnp.einsum("bij,bjc->bic", weight, v)     # (B*nh, N, dh)

        # Merge heads
        h = h.reshape(B, H * W, C)                    # (B, N, C)
        h = self.proj_out(h)                           # (B, N, C)
        h = h.reshape(B, H, W, C)

        return x + h


# ---------------------------------------------------------------------------
# TimestepEmbedSequential — polymorphic container
# ---------------------------------------------------------------------------

class TimestepEmbedSequential(nn.Module):
    """A sequential container that routes ``(x, emb, deterministic)`` to
    children.

    Each child can be:
    - A :class:`ResBlock` or :class:`AttentionBlock` (takes ``emb``).
    - A plain module like ``Conv`` (only takes ``x``).

    This mirrors ``ldm.modules.diffusionmodules.openaimodel.TimestepEmbedSequential``.
    """

    layers: Sequence[nn.Module]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        emb: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, (ResBlock, AttentionBlock)):
                x = layer(x, emb, deterministic=deterministic)
            else:
                x = layer(x)
        return x


# ---------------------------------------------------------------------------
# UNetModel
# ---------------------------------------------------------------------------

class UNetModel(nn.Module):
    """Full UNet denoiser for latent diffusion.

    Follows the OpenAI guided-diffusion ``UNetModel`` architecture with the
    IMPOSE config (no encoder/decoder attention, middle-block self-attention
    only, FiLM conditioning, ResBlock up/downsampling).

    All spatial tensors use **NHWC** layout.

    Attributes:
        image_size:            Spatial size of latent maps.
        in_channels:           Number of input channels.
        model_channels:        Base channel width.
        out_channels:          Number of output channels (noise prediction).
        num_res_blocks:        Residual blocks per down/up-sample stage.
        attention_resolutions: Downsample factors at which attention is used
                               (for this config, no enc/dec blocks match).
        dropout:               Dropout rate.
        channel_mult:          Per-level channel multipliers.
        num_heads:             Number of attention heads (middle block).
        num_head_channels:     If >0 overrides ``num_heads`` (set to -1 here).
        use_scale_shift_norm:  FiLM-style conditioning.
        resblock_updown:       Use ResBlock for up/downsampling.
    """

    image_size: int = 128
    in_channels: int = 3
    model_channels: int = 64
    out_channels: int = 3
    num_res_blocks: int = 2
    attention_resolutions: Sequence[int] = (32, 16, 8)
    dropout: float = 0.0
    channel_mult: Sequence[int] = (1, 2, 4)
    num_heads: int = 8
    num_head_channels: int = -1
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True

    def setup(self):
        model_channels = self.model_channels
        time_embed_dim = model_channels * 4  # 64 * 4 = 256

        # ---------------------------------------------------------------
        # Time embedding MLP: model_channels → time_embed_dim → time_embed_dim
        # ---------------------------------------------------------------
        self.time_embed_dense1 = nn.Dense(
            features=time_embed_dim, name="time_embed_dense1"
        )
        self.time_embed_dense2 = nn.Dense(
            features=time_embed_dim, name="time_embed_dense2"
        )

        # ---------------------------------------------------------------
        # Build encoder (input_blocks)
        # ---------------------------------------------------------------
        # Block 0: plain 3×3 conv  in_channels → model_channels
        conv_in = nn.Conv(
            features=model_channels,
            kernel_size=(3, 3),
            padding="SAME",
            name="input_conv",
        )

        input_blocks: list[TimestepEmbedSequential] = [
            TimestepEmbedSequential(layers=[conv_in], name="input_block_0")
        ]
        input_block_chans: list[int] = [model_channels]
        ch = model_channels
        ds = 1
        block_idx = 1  # running index for unique Flax names

        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                current_block_idx = block_idx
                layers: list[nn.Module] = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=mult * model_channels,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        name=f"input_res_{current_block_idx}",
                    )
                ]
                ch = mult * model_channels

                # Encoder attention (won't fire for this config)
                if ds in self.attention_resolutions:
                    num_heads = self._resolve_num_heads(ch)
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            name=f"input_attn_{current_block_idx}",
                        )
                    )

                input_blocks.append(
                    TimestepEmbedSequential(
                        layers=layers,
                        name=f"input_block_{block_idx}",
                    )
                )
                input_block_chans.append(ch)
                block_idx += 1

            # Downsample (except at last level)
            if level != len(self.channel_mult) - 1:
                if self.resblock_updown:
                    down_layer = ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=ch,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        down=True,
                        name=f"input_res_down_{block_idx}",
                    )
                else:
                    down_layer = Downsample(
                        channels=ch, use_conv=True, name=f"input_downsample_{block_idx}"
                    )
                input_blocks.append(
                    TimestepEmbedSequential(
                        layers=[down_layer],
                        name=f"input_block_{block_idx}",
                    )
                )
                input_block_chans.append(ch)
                block_idx += 1
                ds *= 2

        self.input_blocks = input_blocks

        # ---------------------------------------------------------------
        # Middle block: ResBlock → AttentionBlock → ResBlock
        # ---------------------------------------------------------------
        mid_num_heads = self._resolve_num_heads(ch)
        self.middle_block = TimestepEmbedSequential(
            layers=[
                ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=self.dropout,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    name="mid_res1",
                ),
                AttentionBlock(
                    channels=ch,
                    num_heads=mid_num_heads,
                    name="mid_attn",
                ),
                ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    dropout=self.dropout,
                    use_scale_shift_norm=self.use_scale_shift_norm,
                    name="mid_res2",
                ),
            ],
            name="middle_block",
        )

        # ---------------------------------------------------------------
        # Build decoder (output_blocks) — mirror of encoder
        # ---------------------------------------------------------------
        output_blocks: list[TimestepEmbedSequential] = []
        block_idx = 0

        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                current_block_idx = block_idx
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        channels=ch + ich,
                        emb_channels=time_embed_dim,
                        dropout=self.dropout,
                        out_channels=model_channels * mult,
                        use_scale_shift_norm=self.use_scale_shift_norm,
                        name=f"output_res_{current_block_idx}",
                    )
                ]
                ch = model_channels * mult

                # Decoder attention (won't fire for this config)
                if ds in self.attention_resolutions:
                    num_heads = self._resolve_num_heads(ch)
                    layers.append(
                        AttentionBlock(
                            channels=ch,
                            num_heads=num_heads,
                            name=f"output_attn_{current_block_idx}",
                        )
                    )

                # Upsample at the last sub-block of every level except 0
                if level and i == self.num_res_blocks:
                    if self.resblock_updown:
                        layers.append(
                            ResBlock(
                                channels=ch,
                                emb_channels=time_embed_dim,
                                dropout=self.dropout,
                                out_channels=ch,
                                use_scale_shift_norm=self.use_scale_shift_norm,
                                up=True,
                                name=f"output_res_up_{current_block_idx}",
                            )
                        )
                    else:
                        layers.append(
                            Upsample(
                                channels=ch, use_conv=True, name=f"output_upsample_{current_block_idx}"
                            )
                        )
                    ds //= 2

                output_blocks.append(
                    TimestepEmbedSequential(
                        layers=layers,
                        name=f"output_block_{block_idx}",
                    )
                )
                block_idx += 1

        self.output_blocks = output_blocks

        # ---------------------------------------------------------------
        # Final output projection: GroupNorm → SiLU → zero-init Conv
        # ---------------------------------------------------------------
        self.out_norm = nn.GroupNorm(num_groups=32, name="out_norm")
        self.out_conv = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="out_conv",
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _resolve_num_heads(self, ch: int) -> int:
        """Compute the number of attention heads for channel width ``ch``.

        With ``num_head_channels == -1`` (the IMPOSE config) we simply
        use ``self.num_heads`` directly.  Otherwise we compute
        ``ch // num_head_channels``.
        """
        if self.num_head_channels == -1:
            return self.num_heads
        return ch // self.num_head_channels

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def __call__(
        self,
        x: jnp.ndarray,
        timesteps: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Denoise ``x`` conditioned on ``timesteps``.

        Args:
            x:             (B, H, W, in_channels) noisy latent.
            timesteps:     (B,) integer diffusion timestep indices.
            deterministic: Disable dropout when True.

        Returns:
            (B, H, W, out_channels) predicted noise.
        """
        # --- timestep embedding ---
        emb = timestep_embedding(timesteps, self.model_channels)  # (B, 64)
        emb = self.time_embed_dense1(emb)                         # (B, 256)
        emb = nn.silu(emb)
        emb = self.time_embed_dense2(emb)                         # (B, 256)

        # --- encoder ---
        h = x
        hs: list[jnp.ndarray] = []
        for block in self.input_blocks:
            h = block(h, emb, deterministic=deterministic)
            hs.append(h)

        # --- middle ---
        h = self.middle_block(h, emb, deterministic=deterministic)

        # --- decoder ---
        for block in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)  # skip on C axis
            h = block(h, emb, deterministic=deterministic)

        # --- output head ---
        h = self.out_norm(h)
        h = nn.silu(h)
        h = self.out_conv(h)
        return h
