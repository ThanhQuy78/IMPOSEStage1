"""Convert PyTorch IMPOSE checkpoint weights to JAX/Flax format.

Extracts VQ-VAE (first_stage_model) weights from the full LatentDiffusion
checkpoint and converts them to Flax-compatible parameter dicts.

Key conversions:
- Conv2d weights: PyTorch (out, in, kH, kW) → JAX (kH, kW, in, out)
- Linear weights: PyTorch (out, in) → JAX (in, out)  
- GroupNorm weight/bias: same shape, just rename
- Embedding: same shape
"""

import os
import sys
import argparse
import numpy as np
from collections.abc import Mapping
from typing import Dict, Any, Tuple

# For type hints only
FrozenDict = Any


def load_pytorch_checkpoint(ckpt_path: str) -> dict:
    """Load PyTorch checkpoint state_dict.
    
    Args:
        ckpt_path: Path to .ckpt file.
        
    Returns:
        state_dict dictionary.
    """
    import torch
    print(f"Loading PyTorch checkpoint from {ckpt_path}...")
    try:
        pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(ckpt_path, map_location="cpu")
    
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    
    # Extract scale_factor if present
    scale_factor = None
    if "scale_factor" in sd:
        scale_factor = sd["scale_factor"].numpy().item()
        print(f"  Found scale_factor: {scale_factor}")
    
    # Convert all tensors to numpy
    np_sd = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            np_sd[k] = v.detach().cpu().numpy()
        else:
            np_sd[k] = v
    
    print(f"  Loaded {len(np_sd)} keys")
    return np_sd, scale_factor


def extract_vqvae_weights(state_dict: dict) -> dict:
    """Extract first_stage_model weights from full LatentDiffusion state_dict.
    
    Keys in checkpoint look like:
        first_stage_model.encoder.conv_in.weight
        first_stage_model.encoder.down.0.block.0.norm1.weight
        first_stage_model.decoder.up.2.block.0.conv1.weight
        first_stage_model.quantize.embedding.weight
        first_stage_model.quant_conv.weight
        first_stage_model.post_quant_conv.weight
    
    Returns:
        Dict with first_stage_model prefix stripped.
    """
    prefix = "first_stage_model."
    vqvae_sd = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            vqvae_sd[new_key] = v
    
    print(f"  Extracted {len(vqvae_sd)} VQ-VAE keys")
    return vqvae_sd


def convert_conv2d(weight: np.ndarray, bias: np.ndarray = None) -> dict:
    """Convert Conv2d weights from PyTorch to JAX format.
    
    PyTorch: (out_channels, in_channels, kH, kW)
    JAX/Flax: (kH, kW, in_channels, out_channels)
    """
    kernel = np.transpose(weight, (2, 3, 1, 0))
    result = {"kernel": kernel}
    if bias is not None:
        result["bias"] = bias
    return result


def convert_linear(weight: np.ndarray, bias: np.ndarray = None) -> dict:
    """Convert Linear weights from PyTorch to JAX format.
    
    PyTorch: (out_features, in_features)
    JAX/Flax: (in_features, out_features)
    """
    kernel = weight.T
    result = {"kernel": kernel}
    if bias is not None:
        result["bias"] = bias
    return result


def convert_group_norm(weight: np.ndarray, bias: np.ndarray) -> dict:
    """Convert GroupNorm parameters.
    
    Shape is the same in both frameworks (num_channels,).
    Flax uses 'scale' and 'bias' instead of 'weight' and 'bias'.
    """
    return {"scale": weight, "bias": bias}


def convert_embedding(weight: np.ndarray) -> dict:
    """Convert Embedding weights.
    
    Shape is the same: (num_embeddings, embedding_dim).
    Flax uses 'embedding' key.
    """
    return {"embedding": weight}


def build_resnet_block_params(sd: dict, prefix: str) -> dict:
    """Convert a ResnetBlock's parameters.
    
    Expected keys under prefix:
        norm1.weight, norm1.bias
        conv1.weight, conv1.bias
        norm2.weight, norm2.bias
        conv2.weight, conv2.bias
        [nin_shortcut.weight, nin_shortcut.bias]  (optional)
    """
    params = {}
    
    # norm1
    params["norm1"] = convert_group_norm(
        sd[f"{prefix}norm1.weight"],
        sd[f"{prefix}norm1.bias"]
    )
    
    # conv1
    params["conv1"] = convert_conv2d(
        sd[f"{prefix}conv1.weight"],
        sd[f"{prefix}conv1.bias"]
    )
    
    # norm2
    params["norm2"] = convert_group_norm(
        sd[f"{prefix}norm2.weight"],
        sd[f"{prefix}norm2.bias"]
    )
    
    # conv2
    params["conv2"] = convert_conv2d(
        sd[f"{prefix}conv2.weight"],
        sd[f"{prefix}conv2.bias"]
    )
    
    # Optional shortcut
    if f"{prefix}nin_shortcut.weight" in sd:
        params["nin_shortcut"] = convert_conv2d(
            sd[f"{prefix}nin_shortcut.weight"],
            sd[f"{prefix}nin_shortcut.bias"]
        )
    
    if f"{prefix}conv_shortcut.weight" in sd:
        params["conv_shortcut"] = convert_conv2d(
            sd[f"{prefix}conv_shortcut.weight"],
            sd[f"{prefix}conv_shortcut.bias"]
        )
    
    return params


def build_attn_block_params(sd: dict, prefix: str) -> dict:
    """Convert an AttnBlock's parameters.
    
    Keys: norm, q, k, v, proj_out (all Conv2d 1x1).
    """
    params = {}
    
    params["norm"] = convert_group_norm(
        sd[f"{prefix}norm.weight"],
        sd[f"{prefix}norm.bias"]
    )
    
    for name in ["q", "k", "v", "proj_out"]:
        params[name] = convert_conv2d(
            sd[f"{prefix}{name}.weight"],
            sd[f"{prefix}{name}.bias"]
        )
    
    return params


def build_encoder_params(sd: dict, ch_mult=(1, 2, 4), num_res_blocks=2,
                         attn_resolutions=(16,), resolution=512) -> dict:
    """Convert Encoder parameters."""
    params = {}
    num_resolutions = len(ch_mult)
    
    # conv_in
    params["conv_in"] = convert_conv2d(
        sd["encoder.conv_in.weight"],
        sd["encoder.conv_in.bias"]
    )
    
    # Down blocks. The Flax module uses compact names such as
    # down_0_block_0 instead of the PyTorch hierarchy down.0.block.0.
    curr_res = resolution
    for i_level in range(num_resolutions):
        for i_block in range(num_res_blocks):
            prefix = f"encoder.down.{i_level}.block.{i_block}."
            params[f"down_{i_level}_block_{i_block}"] = build_resnet_block_params(sd, prefix)

        if curr_res in attn_resolutions:
            for i_block in range(num_res_blocks):
                prefix = f"encoder.down.{i_level}.attn.{i_block}."
                if f"{prefix}norm.weight" in sd:
                    params[f"down_{i_level}_attn_{i_block}"] = build_attn_block_params(sd, prefix)

        if i_level != num_resolutions - 1:
            prefix = f"encoder.down.{i_level}.downsample."
            params[f"down_{i_level}_downsample"] = {
                "conv": convert_conv2d(
                    sd[f"{prefix}conv.weight"],
                    sd[f"{prefix}conv.bias"]
                )
            }
            curr_res //= 2
    
    # Mid block
    params["mid_block_1"] = build_resnet_block_params(sd, "encoder.mid.block_1.")
    params["mid_attn_1"] = build_attn_block_params(sd, "encoder.mid.attn_1.")
    params["mid_block_2"] = build_resnet_block_params(sd, "encoder.mid.block_2.")
    
    # Output
    params["norm_out"] = convert_group_norm(
        sd["encoder.norm_out.weight"],
        sd["encoder.norm_out.bias"]
    )
    params["conv_out"] = convert_conv2d(
        sd["encoder.conv_out.weight"],
        sd["encoder.conv_out.bias"]
    )
    
    return params


def build_decoder_params(sd: dict, ch_mult=(1, 2, 4), num_res_blocks=2,
                         attn_resolutions=(16,), resolution=512) -> dict:
    """Convert Decoder parameters."""
    params = {}
    num_resolutions = len(ch_mult)
    
    # conv_in
    params["conv_in"] = convert_conv2d(
        sd["decoder.conv_in.weight"],
        sd["decoder.conv_in.bias"]
    )
    
    # Mid block
    params["mid_block_1"] = build_resnet_block_params(sd, "decoder.mid.block_1.")
    params["mid_attn_1"] = build_attn_block_params(sd, "decoder.mid.attn_1.")
    params["mid_block_2"] = build_resnet_block_params(sd, "decoder.mid.block_2.")
    
    # Up blocks. Keep PyTorch level numbers because the Flax decoder loop
    # also names modules up_2, up_1, up_0 in execution order.
    curr_res = resolution // 2**(num_resolutions - 1)
    for i_level in reversed(range(num_resolutions)):
        for i_block in range(num_res_blocks + 1):
            prefix = f"decoder.up.{i_level}.block.{i_block}."
            params[f"up_{i_level}_block_{i_block}"] = build_resnet_block_params(sd, prefix)

        if curr_res in attn_resolutions:
            for i_block in range(num_res_blocks + 1):
                prefix = f"decoder.up.{i_level}.attn.{i_block}."
                if f"{prefix}norm.weight" in sd:
                    params[f"up_{i_level}_attn_{i_block}"] = build_attn_block_params(sd, prefix)

        if i_level != 0:
            prefix = f"decoder.up.{i_level}.upsample."
            params[f"up_{i_level}_upsample"] = {
                "conv": convert_conv2d(
                    sd[f"{prefix}conv.weight"],
                    sd[f"{prefix}conv.bias"]
                )
            }
            curr_res *= 2
    
    # Output
    params["norm_out"] = convert_group_norm(
        sd["decoder.norm_out.weight"],
        sd["decoder.norm_out.bias"]
    )
    params["conv_out"] = convert_conv2d(
        sd["decoder.conv_out.weight"],
        sd["decoder.conv_out.bias"]
    )
    
    return params


def build_vqvae_params(vqvae_sd: dict) -> dict:
    """Build complete VQ-VAE parameter dict for Flax.
    
    Returns nested dict matching Flax VQModelInterface structure.
    """
    params = {}
    
    # Encoder
    params["encoder"] = build_encoder_params(vqvae_sd)
    
    # Decoder
    params["decoder"] = build_decoder_params(vqvae_sd)
    
    # VectorQuantizer uses self.param("embedding", ...), so the parameter is
    # directly quantize/embedding in the Flax tree.
    params["quantize"] = convert_embedding(vqvae_sd["quantize.embedding.weight"])
    
    # quant_conv (1x1 conv)
    params["quant_conv"] = convert_conv2d(
        vqvae_sd["quant_conv.weight"],
        vqvae_sd["quant_conv.bias"]
    )
    
    # post_quant_conv (1x1 conv)
    params["post_quant_conv"] = convert_conv2d(
        vqvae_sd["post_quant_conv.weight"],
        vqvae_sd["post_quant_conv.bias"]
    )
    
    return params


def convert_checkpoint(ckpt_path: str, output_dir: str) -> Tuple[dict, float]:
    """Full conversion pipeline.
    
    Args:
        ckpt_path: Path to PyTorch .ckpt file.
        output_dir: Directory to save converted params.
        
    Returns:
        Tuple of (vqvae_params dict, scale_factor float).
    """
    # Load PyTorch checkpoint
    state_dict, scale_factor = load_pytorch_checkpoint(ckpt_path)
    
    # Extract VQ-VAE weights
    vqvae_sd = extract_vqvae_weights(state_dict)
    
    # Convert to Flax format
    vqvae_params = build_vqvae_params(vqvae_sd)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vqvae_params.npz")
    
    # Flatten nested dict for npz saving
    flat_params = _flatten_dict(vqvae_params)
    np.savez(output_path, **flat_params)
    print(f"Saved VQ-VAE params to {output_path}")
    
    # Save scale_factor
    if scale_factor is not None:
        sf_path = os.path.join(output_dir, "scale_factor.npy")
        np.save(sf_path, np.array(scale_factor, dtype=np.float32))
        print(f"Saved scale_factor={scale_factor} to {sf_path}")
    
    return vqvae_params, scale_factor


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dict with '/' separators."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, Mapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat: dict, sep: str = "/") -> dict:
    """Unflatten a dict with '/' separators back to nested dict."""
    result = {}
    for key, value in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def _has_path(d: dict, path: str, sep: str = "/") -> bool:
    """Return True if a nested dict contains a slash-separated path."""
    cur = d
    for part in path.split(sep):
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    return True


def is_valid_vqvae_param_tree(params: dict) -> bool:
    """Check that converted params match the Flax VQModelInterface tree."""
    required_paths = [
        "encoder/down_0_block_0/norm1/scale",
        "encoder/mid_attn_1/proj_out/kernel",
        "decoder/up_2_block_0/norm1/scale",
        "decoder/up_1_upsample/conv/kernel",
        "quantize/embedding",
        "quant_conv/kernel",
        "post_quant_conv/kernel",
    ]
    return all(_has_path(params, path) for path in required_paths)


def load_converted_params(params_dir: str) -> Tuple[dict, float]:
    """Load previously converted VQ-VAE params.
    
    Args:
        params_dir: Directory containing vqvae_params.npz and scale_factor.npy.
        
    Returns:
        Tuple of (vqvae_params nested dict, scale_factor).
    """
    import jax.numpy as jnp
    
    params_path = os.path.join(params_dir, "vqvae_params.npz")
    data = np.load(params_path, allow_pickle=False)
    flat_params = {k: jnp.array(v) for k, v in data.items()}
    params = _unflatten_dict(flat_params)
    
    sf_path = os.path.join(params_dir, "scale_factor.npy")
    scale_factor = float(np.load(sf_path)) if os.path.exists(sf_path) else None
    
    return params, scale_factor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IMPOSE PyTorch checkpoint to JAX")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to PyTorch .ckpt file")
    parser.add_argument("--output", type=str, default="jax_stage1/checkpoints/converted",
                        help="Output directory for converted params")
    args = parser.parse_args()
    
    params, scale_factor = convert_checkpoint(args.ckpt, args.output)
    print(f"\nConversion complete!")
    print(f"  VQ-VAE params: {sum(np.prod(v.shape) for v in _flatten_dict(params).values())} parameters")
    print(f"  Scale factor: {scale_factor}")
