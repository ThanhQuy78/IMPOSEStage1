"""Inference script for IMPOSE Stage 1 — Generate rolled fingerprint images.

Uses DDIM sampling in the latent space of the trained LDM,
then decodes through the frozen VQ-VAE to produce 512×512 grayscale images.

Usage:
    python sample.py --config configs/rolled_ldm.yaml \
                     --ckpt checkpoints/step_0100000 \
                     --n_samples 16 --ddim_steps 50 --seed 42
"""

import os
import sys
import argparse
import time

import yaml
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from PIL import Image

# Local imports
from IMPOSEStage1.jax_stage1.models.vqvae import VQModelInterface
from IMPOSEStage1.jax_stage1.models.unet import UNetModel
from IMPOSEStage1.jax_stage1.diffusion.schedule import make_beta_schedule, compute_schedule_constants
from IMPOSEStage1.jax_stage1.diffusion.ddim import make_ddim_timesteps, make_ddim_schedule, ddim_sample_loop
from IMPOSEStage1.jax_stage1.utils.checkpoint_converter import (
    convert_checkpoint,
    is_valid_vqvae_param_tree,
    load_converted_params,
)


def load_unet_checkpoint(ckpt_dir: str) -> dict:
    """Load UNet checkpoint (EMA weights).
    
    Args:
        ckpt_dir: Directory containing unet_state.npz.
        
    Returns:
        Dict of EMA parameters.
    """
    data = np.load(os.path.join(ckpt_dir, "unet_state.npz"), allow_pickle=False)
    
    step = int(data["step"])
    print(f"Loaded checkpoint from step {step}")
    
    # Extract EMA params
    ema_params = {}
    for k, v in data.items():
        if k.startswith("ema_params/"):
            key = k[len("ema_params/"):]
            ema_params[key] = jnp.array(v)
    
    # Unflatten
    from IMPOSEStage1.jax_stage1.utils.checkpoint_converter import _unflatten_dict
    params = _unflatten_dict(ema_params)
    
    return params, step


def generate(config, unet_params, vqvae_params, vqvae_model, unet_model,
             schedule, scale_factor, rng,
             n_samples=16, ddim_steps=50, ddim_eta=0.0, batch_size=None):
    """Generate fingerprint images.
    
    Args:
        config: Config dict.
        unet_params: UNet parameters (EMA).
        vqvae_params: Frozen VQ-VAE parameters.
        vqvae_model: VQ-VAE model.
        unet_model: UNet model.
        schedule: Diffusion schedule constants.
        scale_factor: Latent scaling factor.
        rng: Random key.
        n_samples: Total number of images to generate.
        ddim_steps: Number of DDIM sampling steps.
        ddim_eta: DDIM eta (0.0 = deterministic).
        batch_size: Batch size for generation. If None, generate all at once.
        
    Returns:
        Generated images as numpy array (n_samples, 512, 512), uint8.
    """
    if batch_size is None:
        batch_size = n_samples
    
    # DDIM schedule
    ddim_ts = make_ddim_timesteps(ddim_steps, schedule['betas'].shape[0])
    ddim_sched = make_ddim_schedule(schedule['alphas_cumprod'], ddim_ts, ddim_eta)
    
    # UNet apply function
    @jax.jit
    def apply_fn(params, x, timesteps, deterministic=True):
        return unet_model.apply({"params": params}, x, timesteps, 
                               deterministic=deterministic)
    
    # VQ-VAE decode function
    @jax.jit
    def decode_fn(z):
        z_unscaled = z / scale_factor
        return vqvae_model.apply({"params": vqvae_params}, z_unscaled,
                                 method=vqvae_model.decode)
    
    all_images = []
    num_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"Generating {n_samples} images in {num_batches} batches of {batch_size}...")
    
    for batch_idx in range(num_batches):
        current_batch = min(batch_size, n_samples - batch_idx * batch_size)
        rng, batch_rng = random.split(rng)
        
        t_start = time.time()
        
        # Sample latents via DDIM
        latent_shape = (current_batch, 128, 128, 3)
        z_samples = ddim_sample_loop(
            apply_fn, unet_params, latent_shape,
            ddim_sched, batch_rng, ddim_steps
        )
        
        # Decode to image space
        images = decode_fn(z_samples)
        
        # Post-process: [-1, 1] → [0, 255] uint8
        images = jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
        images = (images * 255.0).astype(jnp.uint8)
        images_np = np.array(images[:, :, :, 0])  # (B, H, W) grayscale
        
        all_images.append(images_np)
        
        t_elapsed = time.time() - t_start
        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"{current_batch} images in {t_elapsed:.1f}s "
              f"({t_elapsed / current_batch:.2f}s/image)")
    
    return np.concatenate(all_images, axis=0)[:n_samples]


def main():
    parser = argparse.ArgumentParser(description="Generate rolled fingerprints with IMPOSE LDM")
    parser.add_argument("--config", type=str, default="jax_stage1/configs/rolled_ldm.yaml",
                        help="Path to config YAML")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to UNet checkpoint directory")
    parser.add_argument("--outdir", type=str, default="jax_stage1/outputs/generated",
                        help="Output directory for generated images")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of images to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta (0.0 = deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    rng = random.PRNGKey(args.seed)
    print(f"JAX devices: {jax.device_count()} ({jax.devices()[0].platform})")
    
    # --- Load VQ-VAE ---
    print("\n=== Loading VQ-VAE ===")
    vqvae_cfg = config["model"]["first_stage"]
    vqvae_model = VQModelInterface(
        ch=vqvae_cfg["ch"],
        out_ch=vqvae_cfg["out_ch"],
        ch_mult=tuple(vqvae_cfg["ch_mult"]),
        num_res_blocks=vqvae_cfg["num_res_blocks"],
        attn_resolutions=tuple(vqvae_cfg["attn_resolutions"]),
        dropout=vqvae_cfg.get("dropout", 0.0),
        in_channels=vqvae_cfg["in_channels"],
        resolution=vqvae_cfg["resolution"],
        z_channels=vqvae_cfg["z_channels"],
        double_z=vqvae_cfg["double_z"],
        n_embed=vqvae_cfg["n_embed"],
        embed_dim=vqvae_cfg["embed_dim"],
    )
    
    converted_dir = config["paths"].get(
        "converted_vqvae_dir",
        os.path.join(config["paths"]["checkpoint_dir"], "converted"),
    )
    if os.path.exists(os.path.join(converted_dir, "vqvae_params.npz")):
        vqvae_params, scale_factor = load_converted_params(converted_dir)
        if not is_valid_vqvae_param_tree(vqvae_params):
            print("Converted VQ-VAE params are stale; reconverting checkpoint...")
            vqvae_params, scale_factor = convert_checkpoint(
                config["paths"]["vqvae_ckpt"], converted_dir
            )
    else:
        vqvae_params, scale_factor = convert_checkpoint(
            config["paths"]["vqvae_ckpt"], converted_dir
        )
    if scale_factor is None:
        raise ValueError(
            "No scale_factor found in converted VQ-VAE params. Run training once "
            "with scale_by_std enabled or provide scale_factor.npy."
        )
    print(f"VQ-VAE loaded. Scale factor: {scale_factor}")
    
    # --- Load UNet ---
    print("\n=== Loading UNet (EMA weights) ===")
    unet_cfg = config["model"]["unet"]
    unet_model = UNetModel(
        image_size=unet_cfg["image_size"],
        in_channels=unet_cfg["in_channels"],
        model_channels=unet_cfg["model_channels"],
        out_channels=unet_cfg["out_channels"],
        num_res_blocks=unet_cfg["num_res_blocks"],
        attention_resolutions=tuple(unet_cfg["attention_resolutions"]),
        dropout=unet_cfg.get("dropout", 0.0),
        channel_mult=tuple(unet_cfg["channel_mult"]),
        num_heads=unet_cfg["num_heads"],
        use_scale_shift_norm=unet_cfg["use_scale_shift_norm"],
        resblock_updown=unet_cfg["resblock_updown"],
    )
    
    unet_params, ckpt_step = load_unet_checkpoint(args.ckpt)
    print(f"UNet loaded from step {ckpt_step}")
    
    # --- Setup diffusion ---
    diff_cfg = config["model"]["diffusion"]
    betas = make_beta_schedule(
        diff_cfg["beta_schedule"],
        diff_cfg["timesteps"],
        diff_cfg["linear_start"],
        diff_cfg["linear_end"],
    )
    schedule = compute_schedule_constants(betas)
    
    # --- Generate ---
    print(f"\n=== Generating {args.n_samples} images ===")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  DDIM eta: {args.ddim_eta}")
    print(f"  Seed: {args.seed}")
    
    t_start = time.time()
    images = generate(
        config, unet_params, vqvae_params, vqvae_model, unet_model,
        schedule, scale_factor, rng,
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        batch_size=args.batch_size,
    )
    t_total = time.time() - t_start
    
    # --- Save ---
    os.makedirs(args.outdir, exist_ok=True)
    for i, img in enumerate(images):
        path = os.path.join(args.outdir, f"{i:05d}.png")
        Image.fromarray(img, mode='L').save(path)
    
    print(f"\n=== Done ===")
    print(f"  Generated {len(images)} images in {t_total:.1f}s")
    print(f"  Average: {t_total / len(images):.2f}s/image")
    print(f"  Saved to: {args.outdir}")


if __name__ == "__main__":
    main()
