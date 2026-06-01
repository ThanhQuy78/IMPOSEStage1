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
from models.vqvae import VQModelInterface
from models.unet import UNetModel
from diffusion.schedule import make_beta_schedule, compute_schedule_constants
from diffusion.ddim import make_ddim_timesteps, make_ddim_schedule, ddim_sample_loop
from utils.checkpoint_converter import (
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
    from utils.checkpoint_converter import _unflatten_dict
    params = _unflatten_dict(ema_params)
    
    return params, step


def _build_jit_fns(unet_model, vqvae_model, vqvae_params, scale_factor):
    """Build JIT-compiled UNet and VQ-VAE decode functions (created once)."""

    @jax.jit
    def apply_fn(params, x, timesteps, deterministic=True):
        return unet_model.apply({"params": params}, x, timesteps,
                               deterministic=deterministic)

    @jax.jit
    def decode_fn(z):
        z_unscaled = z / scale_factor
        return vqvae_model.apply({"params": vqvae_params}, z_unscaled,
                                 method=vqvae_model.decode)

    return apply_fn, decode_fn


def _warmup_jit(apply_fn, decode_fn, unet_params, batch_size, ddim_sched, ddim_steps):
    """Run one dummy batch so XLA compiles before the real loop."""
    print("Warming up JIT (first compilation may take a few minutes)...")
    sys.stdout.flush()
    dummy_rng = random.PRNGKey(0)
    dummy_shape = (batch_size, 128, 128, 3)
    z = ddim_sample_loop(apply_fn, unet_params, dummy_shape,
                         ddim_sched, dummy_rng, ddim_steps)
    _ = decode_fn(z)
    # Block until computation finishes so the compile is truly done.
    jax.block_until_ready(_)
    print("JIT warm-up complete.")
    sys.stdout.flush()


def generate(config, unet_params, vqvae_params, vqvae_model, unet_model,
             schedule, scale_factor, rng,
             n_samples=16, ddim_steps=50, ddim_eta=0.0, batch_size=None,
             outdir="outputs/generated", resume=True):
    """Generate fingerprint images and save each batch to disk immediately.
    
    This avoids accumulating all images in RAM (which causes OOM on Kaggle
    when generating thousands of images).  If *resume* is True and *outdir*
    already contains numbered PNGs, batches that have already been fully
    saved will be skipped.
    
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
        outdir: Directory to save images into.
        resume: If True, skip batches whose images already exist on disk.
        
    Returns:
        Total number of images saved.
    """
    if batch_size is None:
        batch_size = n_samples
    
    os.makedirs(outdir, exist_ok=True)

    # DDIM schedule
    ddim_ts = make_ddim_timesteps(ddim_steps, schedule['betas'].shape[0])
    ddim_sched = make_ddim_schedule(schedule['alphas_cumprod'], ddim_ts, ddim_eta)
    
    # Build JIT functions once (outside the loop)
    apply_fn, decode_fn = _build_jit_fns(
        unet_model, vqvae_model, vqvae_params, scale_factor
    )

    num_batches = (n_samples + batch_size - 1) // batch_size

    # Warm-up: compile for the main batch_size so the real loop is fast.
    _warmup_jit(apply_fn, decode_fn, unet_params, batch_size,
                ddim_sched, ddim_steps)

    print(f"Generating {n_samples} images in {num_batches} batches "
          f"of {batch_size} (saving to disk per batch)...")
    sys.stdout.flush()
    
    saved_count = 0
    for batch_idx in range(num_batches):
        current_batch = min(batch_size, n_samples - batch_idx * batch_size)
        batch_start_idx = batch_idx * batch_size

        # --- Resume: skip if all images in this batch exist ---
        if resume:
            all_exist = all(
                os.path.exists(os.path.join(outdir, f"{batch_start_idx + j:05d}.png"))
                for j in range(current_batch)
            )
            if all_exist:
                saved_count += current_batch
                if batch_idx % 100 == 0:  # log every 100 skipped batches
                    print(f"  Batch {batch_idx + 1}/{num_batches}: "
                          f"already exists, skipping.")
                    sys.stdout.flush()
                continue

        rng, batch_rng = random.split(rng)
        t_start = time.time()

        # --- Sample latents via DDIM ---
        # NOTE: if current_batch != batch_size (last batch), XLA will
        # recompile for the new shape.  Pad to batch_size to avoid this.
        need_pad = current_batch < batch_size
        actual_bs = batch_size if need_pad else current_batch
        latent_shape = (actual_bs, 128, 128, 3)

        z_samples = ddim_sample_loop(
            apply_fn, unet_params, latent_shape,
            ddim_sched, batch_rng, ddim_steps
        )
        
        # Decode to image space
        images = decode_fn(z_samples)
        # Block until GPU is done before measuring time / saving.
        jax.block_until_ready(images)
        
        # Post-process: [-1, 1] → [0, 255] uint8
        images = jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
        images = (images * 255.0).astype(jnp.uint8)
        images_np = np.array(images[:current_batch, :, :, 0])  # (B, H, W)

        # Free GPU memory eagerly.
        del z_samples, images

        # --- Save to disk immediately ---
        for j in range(current_batch):
            path = os.path.join(outdir, f"{batch_start_idx + j:05d}.png")
            Image.fromarray(images_np[j], mode='L').save(path)
        saved_count += current_batch

        # Free CPU memory.
        del images_np
        
        t_elapsed = time.time() - t_start
        print(f"  Batch {batch_idx + 1}/{num_batches}: "
              f"{current_batch} images in {t_elapsed:.1f}s "
              f"({t_elapsed / current_batch:.2f}s/image) "
              f"[total saved: {saved_count}/{n_samples}]")
        sys.stdout.flush()
    
    return saved_count


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
    saved_count = generate(
        config, unet_params, vqvae_params, vqvae_model, unet_model,
        schedule, scale_factor, rng,
        n_samples=args.n_samples,
        ddim_steps=args.ddim_steps,
        ddim_eta=args.ddim_eta,
        batch_size=args.batch_size,
        outdir=args.outdir,
        resume=True,
    )
    t_total = time.time() - t_start
    
    print(f"\n=== Done ===")
    print(f"  Generated {saved_count} images in {t_total:.1f}s")
    print(f"  Average: {t_total / saved_count:.2f}s/image")
    print(f"  Saved to: {args.outdir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
