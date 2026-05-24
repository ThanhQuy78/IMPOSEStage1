"""Training script for IMPOSE Stage 1 — Rolled Fingerprint LDM.

Pure JAX/Flax/Optax training loop for the unconditional Latent Diffusion Model.
Designed for TPUv5e-8 with data parallelism via jax.pmap.

Usage:
    python train.py --config configs/rolled_ldm.yaml
"""

import os
import sys
import time
import argparse
import functools
from collections.abc import Mapping
from typing import Any, Dict, Tuple, Optional

import yaml
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import optax

# Local imports
from models.vqvae import VQModelInterface
from models.unet import UNetModel
from models.ema import create_ema_state, ema_update
from diffusion.schedule import make_beta_schedule, compute_schedule_constants
from diffusion.ddpm import q_sample, p_losses, extract
from diffusion.ddim import make_ddim_timesteps, make_ddim_schedule, ddim_sample_loop
from data.dataset import (
    create_numpy_dataset, data_iterator, get_steps_per_epoch, find_all_images,
    load_and_preprocess_image,
)


# ---------------------------------------------------------------------------
# Train State
# ---------------------------------------------------------------------------
class LDMTrainState(train_state.TrainState):
    """Extended train state with EMA parameters and RNG."""
    ema_params: Any = None
    ema_step: int = 0
    rng: jnp.ndarray = None


# ---------------------------------------------------------------------------
# VQ-VAE Encoding (frozen, runs on each device)
# ---------------------------------------------------------------------------
def encode_first_stage(vqvae_params: dict, vqvae_model: VQModelInterface,
                       images: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """Encode images to latent space using frozen VQ-VAE.
    
    Args:
        vqvae_params: Frozen VQ-VAE parameters.
        vqvae_model: VQ-VAE model instance.
        images: (B, 512, 512, 1) input images.
        scale_factor: Latent scaling factor.
        
    Returns:
        Scaled latent: (B, 128, 128, 3).
    """
    z = vqvae_model.apply({"params": vqvae_params}, images, method=vqvae_model.encode)
    return z * scale_factor


def decode_first_stage(vqvae_params: dict, vqvae_model: VQModelInterface,
                       z: jnp.ndarray, scale_factor: float) -> jnp.ndarray:
    """Decode latent back to image space using frozen VQ-VAE.
    
    Args:
        vqvae_params: Frozen VQ-VAE parameters.
        vqvae_model: VQ-VAE model instance.
        z: (B, 128, 128, 3) scaled latent.
        scale_factor: Latent scaling factor.
        
    Returns:
        Decoded images: (B, 512, 512, 1).
    """
    z = z / scale_factor
    return vqvae_model.apply({"params": vqvae_params}, z, method=vqvae_model.decode)


# ---------------------------------------------------------------------------
# Training Step (pmap-compatible)
# ---------------------------------------------------------------------------
def create_train_step(unet_model, schedule, loss_type='l1', parameterization='eps',
                      ema_decay=0.9999):
    """Create a pmap-compatible training step function.
    
    Args:
        unet_model: UNet Flax model.
        schedule: Dict of precomputed schedule constants.
        loss_type: 'l1' or 'l2'.
        parameterization: 'eps' or 'x0'.
        ema_decay: EMA decay for sampling weights.
        
    Returns:
        train_step function for jax.pmap.
    """
    
    def train_step(state: LDMTrainState, z_batch: jnp.ndarray):
        """Single training step.
        
        Args:
            state: Current training state.
            z_batch: (per_device_batch, 128, 128, 3) pre-encoded latents.
            
        Returns:
            Updated state, loss metrics dict.
        """
        rng, t_rng, noise_rng = random.split(state.rng, 3)
        batch_size = z_batch.shape[0]
        
        # Sample random timesteps
        t = random.randint(t_rng, (batch_size,), 0, schedule['betas'].shape[0])
        
        def loss_fn(params):
            return p_losses(
                apply_fn=lambda p, x, ts, **kw: unet_model.apply(
                    {"params": p}, x, ts, deterministic=True
                ),
                params=params,
                x_start=z_batch,
                t=t,
                rng=noise_rng,
                schedule=schedule,
                loss_type=loss_type,
                parameterization=parameterization,
            )
        
        (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Average gradients across devices
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        
        # Apply updates
        state = state.apply_gradients(grads=grads)
        
        # Update EMA
        new_ema_params = ema_update(
            state.ema_params, state.params, 
            decay=ema_decay, step=state.ema_step
        )
        state = state.replace(
            ema_params=new_ema_params,
            ema_step=state.ema_step + 1,
            rng=rng,
        )
        
        metrics = {"loss": loss}
        return state, metrics
    
    return train_step


# ---------------------------------------------------------------------------
# Compute scale_factor from first batch
# ---------------------------------------------------------------------------
def compute_scale_factor(vqvae_params, vqvae_model, first_batch):
    """Compute scale_factor = 1/std(z) from the first batch of data.
    
    This matches the original IMPOSE `scale_by_std` behavior.
    
    Args:
        vqvae_params: VQ-VAE parameters.
        vqvae_model: VQ-VAE model.
        first_batch: (B, 512, 512, 1) images.
        
    Returns:
        scale_factor: float.
    """
    z = vqvae_model.apply({"params": vqvae_params}, first_batch, 
                          method=vqvae_model.encode)
    scale_factor = 1.0 / jnp.std(z).item()
    print(f"Computed scale_factor from data: {scale_factor:.6f}")
    return scale_factor


# ---------------------------------------------------------------------------
# Pre-encode dataset 
# ---------------------------------------------------------------------------
def pre_encode_dataset(vqvae_params, vqvae_model, images, scale_factor,
                       batch_size=32, cache_path=None):
    """Pre-encode all training images to latent space.
    
    Since VQ-VAE is frozen, we can encode once and train on latents directly.
    This saves significant compute during training.
    
    Args:
        vqvae_params: VQ-VAE parameters.
        vqvae_model: VQ-VAE model.
        images: (N, 512, 512, 1) all training images.
        scale_factor: Latent scaling factor.
        batch_size: Encoding batch size.
        
    Returns:
        Encoded latents: (N, 128, 128, 3).
    """
    n = len(images)
    # Determine latent shape from a test encode
    test_z = vqvae_model.apply({"params": vqvae_params}, images[:1], 
                               method=vqvae_model.encode)
    latent_shape = test_z.shape[1:]  # (128, 128, 3)
    print(f"Latent shape: {latent_shape}")
    
    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        latents = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(n, *latent_shape),
        )
        print(f"Writing latent cache to: {cache_path}")
    else:
        latents = np.zeros((n, *latent_shape), dtype=np.float32)
    
    @jax.jit
    def encode_batch(batch):
        z = vqvae_model.apply({"params": vqvae_params}, batch, 
                              method=vqvae_model.encode)
        return z * scale_factor
    
    print(f"Pre-encoding {n} images to latent space...")
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        batch = jnp.array(images[i:end])
        z = encode_batch(batch)
        latents[i:end] = np.array(z)
        if cache_path:
            latents.flush()
        if (i // batch_size) % 50 == 0:
            print(f"  Encoded {end}/{n}")
    
    print(f"Pre-encoding complete: {latents.shape}, {latents.nbytes / 1e9:.1f} GB")
    return latents


def load_latent_cache(cache_path: str):
    """Load precomputed scaled latents without reading source images."""
    latents = np.load(cache_path, mmap_mode="r")
    if latents.ndim != 4 or latents.shape[1:] != (128, 128, 3):
        raise ValueError(
            f"Invalid latent cache shape {latents.shape}; expected (N, 128, 128, 3)"
        )
    print(f"Loaded latent cache: {cache_path}")
    print(f"Latents: {latents.shape}, {latents.nbytes / 1e9:.1f} GB")
    return latents, len(latents)


# ---------------------------------------------------------------------------
# Sampling (for logging)
# ---------------------------------------------------------------------------
def generate_samples(unet_model, params, vqvae_params, vqvae_model,
                     schedule, scale_factor, rng, 
                     n_samples=8, ddim_steps=50, ddim_eta=0.0):
    """Generate sample images using DDIM.
    
    Args:
        unet_model: UNet model.
        params: UNet params (or EMA params).
        vqvae_params: Frozen VQ-VAE params.
        vqvae_model: VQ-VAE model.
        schedule: Diffusion schedule constants.
        scale_factor: Latent scaling factor.
        rng: Random key.
        n_samples: Number of images to generate.
        ddim_steps: DDIM sampling steps.
        ddim_eta: DDIM eta (0 = deterministic).
        
    Returns:
        Generated images: (n_samples, 512, 512, 1) in [0, 1].
    """
    # Setup DDIM schedule
    ddim_ts = make_ddim_timesteps(ddim_steps, schedule['betas'].shape[0])
    ddim_sched = make_ddim_schedule(schedule['alphas_cumprod'], ddim_ts, ddim_eta)
    
    # Sample latents
    def apply_fn(p, x, ts, **kwargs):
        return unet_model.apply({"params": p}, x, ts, deterministic=True)
    
    latent_shape = (n_samples, 128, 128, 3)
    z_samples = ddim_sample_loop(apply_fn, params, latent_shape, 
                                  ddim_sched, rng, ddim_steps)
    
    # Decode to image space
    images = decode_first_stage(vqvae_params, vqvae_model, z_samples, scale_factor)
    
    # Clamp to [0, 1]
    images = jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
    
    return images


def save_sample_images(images: np.ndarray, output_dir: str, step: int):
    """Save generated sample images as PNGs."""
    from PIL import Image
    
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        img_uint8 = (img[:, :, 0] * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode='L').save(
            os.path.join(sample_dir, f"step{step:07d}_sample{i:02d}.png")
        )


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train IMPOSE Stage 1 LDM in JAX")
    parser.add_argument("--config", type=str, default="jax_stage1/configs/rolled_ldm.yaml",
                        help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    parser.add_argument("--pre_encode", action="store_true", default=True,
                        help="Pre-encode dataset to latent space (saves compute)")
    parser.add_argument("--no_pre_encode", dest="pre_encode", action="store_false",
                        help="Disable latent pre-encoding")
    parser.add_argument("--prepare_latents_only", action="store_true",
                        help="Build latent cache from images, then exit")
    args = parser.parse_args()
    
    # --- Load config ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    seed = config["training"]["seed"]
    rng = random.PRNGKey(seed)
    
    num_devices = jax.device_count()
    print(f"JAX devices: {num_devices} ({jax.devices()[0].platform})")
    
    # --- Initialize models ---
    print("\n=== Initializing VQ-VAE (frozen) ===")
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
    
    # Load VQ-VAE weights from converted checkpoint
    from utils.checkpoint_converter import (
        convert_checkpoint,
        is_valid_vqvae_param_tree,
        load_converted_params,
    )
    
    # Separate read-only dir for converted VQ-VAE vs writable checkpoint dir
    converted_dir = config["paths"].get(
        "converted_vqvae_dir",
        os.path.join(config["paths"]["checkpoint_dir"], "converted"),
    )
    ckpt_path = config["paths"]["vqvae_ckpt"]
    
    if os.path.exists(os.path.join(converted_dir, "vqvae_params.npz")):
        print(f"Loading pre-converted VQ-VAE params from {converted_dir}...")
        vqvae_params, saved_scale_factor = load_converted_params(converted_dir)
        if not is_valid_vqvae_param_tree(vqvae_params):
            print("Converted VQ-VAE params are stale; reconverting checkpoint...")
            vqvae_params, saved_scale_factor = convert_checkpoint(ckpt_path, converted_dir)
    else:
        print("Converting PyTorch checkpoint...")
        vqvae_params, saved_scale_factor = convert_checkpoint(ckpt_path, converted_dir)
    
    print(f"VQ-VAE loaded. Saved scale_factor: {saved_scale_factor}")
    
    # --- Load dataset ---
    print("\n=== Loading Dataset ===")
    dataset_path = config["data"]["dataset_path"]
    image_size = config["data"]["image_size"]
    batch_size = config["data"]["batch_size"]
    latent_cache_path = config["data"].get("latent_cache_path")
    
    # Adjust batch size for number of devices
    per_device_batch = batch_size // num_devices
    batch_size = per_device_batch * num_devices
    print(f"Batch size: {batch_size} ({per_device_batch} per device × {num_devices} devices)")
    
    latents = None
    images = None
    if args.pre_encode and latent_cache_path and os.path.exists(latent_cache_path):
        latents, num_images = load_latent_cache(latent_cache_path)
    else:
        images, num_images = create_numpy_dataset(dataset_path, image_size, batch_size)
    steps_per_epoch = get_steps_per_epoch(num_images, batch_size)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # --- Compute or load scale_factor ---
    if config["model"]["diffusion"]["scale_by_std"]:
        if saved_scale_factor is not None:
            scale_factor = float(saved_scale_factor)
            print(f"Using saved scale_factor: {scale_factor}")
        elif images is not None:
            # Compute from first batch
            first_batch = jnp.array(images[:min(batch_size, 64)])
            scale_factor = compute_scale_factor(vqvae_params, vqvae_model, first_batch)
        else:
            raise ValueError(
                "scale_by_std is enabled but no scale_factor.npy was loaded. "
                "Provide scale_factor.npy with the converted VQ-VAE when using latent_cache_path."
            )
    else:
        scale_factor = 1.0
    
    # --- Pre-encode dataset ---
    if args.pre_encode and latents is None:
        print("\n=== Pre-encoding dataset to latent space ===")
        enc_bs = config["data"].get("pre_encode_batch_size", 2)
        print(f"  VQ-VAE encode batch size: {enc_bs}")
        latents = pre_encode_dataset(
            vqvae_params, vqvae_model, images, scale_factor,
            batch_size=enc_bs,
            cache_path=latent_cache_path,
        )
        # Free original images
        del images
    else:
        if not args.pre_encode:
            latents = None

    if args.prepare_latents_only:
        if not latent_cache_path:
            raise ValueError("Set data.latent_cache_path before using --prepare_latents_only")
        print(f"Latent cache ready at: {latent_cache_path}")
        return
    
    # --- Initialize UNet ---
    print("\n=== Initializing UNet ===")
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
    
    # Initialize parameters
    rng, init_rng = random.split(rng)
    dummy_z = jnp.ones((1, 128, 128, 3))
    dummy_t = jnp.array([0])
    unet_variables = unet_model.init(init_rng, dummy_z, dummy_t)
    unet_params = unet_variables["params"]
    
    num_params = sum(p.size for p in jax.tree.leaves(unet_params))
    print(f"UNet parameters: {num_params:,}")
    
    # --- Setup optimizer ---
    lr = config["model"]["learning_rate"]
    warmup_steps = config["model"]["warmup_steps"]
    
    # Linear warmup from 1% to 100% of base LR, then constant
    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(
                init_value=lr * 0.01,
                end_value=lr,
                transition_steps=warmup_steps,
            ),
            optax.constant_schedule(lr),
        ],
        boundaries=[warmup_steps],
    )
    
    clip_norm = config["training"]["gradient_clip_norm"]
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule_fn),
    )
    
    # --- Create train state ---
    rng, state_rng = random.split(rng)
    state = LDMTrainState.create(
        apply_fn=unet_model.apply,
        params=unet_params,
        tx=optimizer,
    )
    state = state.replace(
        ema_params=create_ema_state(unet_params),
        ema_step=0,
        rng=state_rng,
    )
    
    # --- Resume from checkpoint ---
    resume_step = 0
    if args.resume:
        resume_step = _resume_from_checkpoint(args.resume, state, optimizer)
        if resume_step > 0:
            # Reload state with resumed weights
            state = _load_train_state(
                args.resume, unet_params, optimizer, state_rng
            )
            print(f"Resumed from step {resume_step}")
    
    # --- Setup diffusion schedule ---
    diff_cfg = config["model"]["diffusion"]
    betas = make_beta_schedule(
        diff_cfg["beta_schedule"],
        diff_cfg["timesteps"],
        diff_cfg["linear_start"],
        diff_cfg["linear_end"],
    )
    schedule = compute_schedule_constants(betas)
    
    # --- Replicate state across devices ---
    # Split RNG so each device gets a unique stream *before* replicating,
    # otherwise every core trains with identical noise.
    per_device_rngs = random.split(state_rng, num_devices)
    state = jax.device_put_replicated(state, jax.devices())
    state = state.replace(rng=per_device_rngs)
    
    # --- Create pmap'd train step ---
    p_train_step = jax.pmap(
        create_train_step(
            unet_model, schedule,
            loss_type=diff_cfg["loss_type"],
            parameterization=diff_cfg["parameterization"],
            ema_decay=config["model"]["ema"]["decay"],
        ),
        axis_name="devices",
    )
    
    # --- Setup checkpointing ---
    ckpt_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # --- Training loop ---
    max_steps = args.max_steps or config["training"]["max_steps"]
    log_every = config["training"]["log_every_n_steps"]
    sample_every = config["training"]["sample_every_n_steps"]
    save_every = config["training"]["save_every_n_steps"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Starting Training ===")
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size} ({per_device_batch}/device)")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Log every: {log_every}")
    print(f"  Sample every: {sample_every}")
    print(f"  Save every: {save_every}")
    
    # Data iterator
    data_rng = np.random.RandomState(seed)
    if latents is not None:
        train_iter = data_iterator(latents, batch_size, num_devices, 
                                    shuffle=True, rng=data_rng)
    else:
        train_iter = data_iterator(images, batch_size, num_devices,
                                    shuffle=True, rng=data_rng)
    
    step = resume_step
    epoch = 0
    train_start = time.time()
    loss_accum = 0.0
    
    while step < max_steps:
        epoch += 1
        steps_in_epoch = 0
        
        for batch in train_iter:
            if step >= max_steps:
                break
            
            # If not pre-encoded, encode on the fly
            if latents is None:
                raise NotImplementedError(
                    "On-the-fly encoding not yet implemented. Use --pre_encode"
                )
            
            # batch shape: (num_devices, per_device_batch, 128, 128, 3)
            state, metrics = p_train_step(state, batch)
            
            loss = metrics["loss"][0].item()  # Take from first device
            loss_accum += loss
            step += 1
            steps_in_epoch += 1
            
            # Logging
            if step % log_every == 0:
                avg_loss = loss_accum / log_every
                elapsed = time.time() - train_start
                steps_per_sec = step / elapsed
                print(
                    f"Step {step:7d} | "
                    f"Loss: {avg_loss:.5f} | "
                    f"LR: {float(schedule_fn(step)):.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"Epoch: {epoch}"
                )
                loss_accum = 0.0
            
            # Generate samples
            if step % sample_every == 0:
                print(f"  Generating samples at step {step}...")
                # Use EMA params from first device for sampling
                ema_params_single = jax.tree.map(lambda x: x[0], state.ema_params)
                rng, sample_rng = random.split(rng)
                
                try:
                    sample_imgs = generate_samples(
                        unet_model, ema_params_single,
                        vqvae_params, vqvae_model,
                        schedule, scale_factor, sample_rng,
                        n_samples=8, ddim_steps=50,
                    )
                    save_sample_images(np.array(sample_imgs), output_dir, step)
                    print(f"  Samples saved to {output_dir}/samples/")
                except Exception as e:
                    print(f"  Sample generation failed: {e}")
            
            # Save checkpoint
            if step % save_every == 0:
                print(f"  Saving checkpoint at step {step}...")
                _save_checkpoint(state, step, ckpt_dir)
                print(f"  Checkpoint saved.")
            
            if step >= max_steps:
                break
    
    print(f"\nTraining complete! Total steps: {step}")
    total_time = time.time() - train_start
    print(f"Total time: {total_time / 3600:.1f} hours")


def _flatten_params(params, prefix=""):
    """Flatten nested param dict for saving."""
    flat = {}
    for k, v in params.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, Mapping):
            flat.update(_flatten_params(v, key))
        else:
            flat[key] = v
    return flat


def _unflatten_params(flat, sep="/"):
    """Unflatten '/' separated keys back to nested dict."""
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


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------
def _save_checkpoint(state, step: int, ckpt_dir: str):
    """Save full training state (params, ema, optimizer, step) for resume.
    
    Saves:
      - params and ema_params as flat numpy arrays
      - opt_state as flat numpy arrays
      - step and ema_step as scalars
    """
    # Unreplicate (take from device 0)
    state_single = jax.tree.map(lambda x: x[0], state)
    
    save_path = os.path.join(ckpt_dir, f"step_{step:07d}")
    os.makedirs(save_path, exist_ok=True)
    
    flat = {}
    # Params
    for k, v in _flatten_params(state_single.params).items():
        flat[f"params/{k}"] = np.array(v)
    # EMA params
    for k, v in _flatten_params(state_single.ema_params).items():
        flat[f"ema_params/{k}"] = np.array(v)
    # Optimizer state — flatten the full pytree
    opt_leaves, opt_treedef = jax.tree.flatten(state_single.opt_state)
    for i, leaf in enumerate(opt_leaves):
        flat[f"opt_state/{i}"] = np.array(leaf)
    
    np.savez(
        os.path.join(save_path, "unet_state.npz"),
        step=np.array(step),
        ema_step=np.array(int(state_single.ema_step)),
        opt_treedef_n_leaves=np.array(len(opt_leaves)),
        **flat,
    )


def _resume_from_checkpoint(ckpt_path: str, state, optimizer) -> int:
    """Check if checkpoint is valid and return the step number.
    
    Returns 0 if invalid / not found.
    """
    npz_path = os.path.join(ckpt_path, "unet_state.npz")
    if not os.path.exists(npz_path):
        print(f"WARNING: checkpoint not found at {npz_path}, training from scratch.")
        return 0
    data = np.load(npz_path, allow_pickle=False)
    step = int(data["step"])
    print(f"Found checkpoint at step {step}")
    return step


def _load_train_state(ckpt_path: str, init_params, optimizer, state_rng):
    """Load full training state from checkpoint for resume.
    
    Restores params, ema_params, optimizer state, step counters and RNG.
    Returns an LDMTrainState ready for jax.device_put_replicated.
    """
    npz_path = os.path.join(ckpt_path, "unet_state.npz")
    data = np.load(npz_path, allow_pickle=False)
    
    step = int(data["step"])
    ema_step = int(data.get("ema_step", step))
    
    # Restore params
    param_flat = {k[len("params/"):]: jnp.array(v)
                  for k, v in data.items() if k.startswith("params/")}
    params = _unflatten_params(param_flat)
    
    # Restore EMA
    ema_flat = {k[len("ema_params/"):]: jnp.array(v)
                for k, v in data.items() if k.startswith("ema_params/")}
    ema_params = _unflatten_params(ema_flat)
    
    # Restore optimizer state
    # We create a fresh state and then overwrite its opt_state leaves from
    # the checkpoint.  This avoids needing to pickle the treedef.
    state = LDMTrainState.create(
        apply_fn=None,  # not used directly
        params=params,
        tx=optimizer,
    )
    
    n_leaves = int(data.get("opt_treedef_n_leaves", 0))
    if n_leaves > 0:
        # Rebuild opt_state from flat leaves
        _, opt_treedef = jax.tree.flatten(state.opt_state)
        opt_leaves_restored = [jnp.array(data[f"opt_state/{i}"])
                               for i in range(n_leaves)]
        try:
            opt_state = jax.tree.unflatten(opt_treedef, opt_leaves_restored)
            state = state.replace(opt_state=opt_state)
            print(f"  Restored optimizer state ({n_leaves} leaves)")
        except Exception as e:
            print(f"  WARNING: Could not restore opt_state: {e}")
            print(f"  Optimizer will be re-initialized (LR warmup restarts).")
    else:
        print("  No opt_state in checkpoint; optimizer re-initialized.")
    
    # Advance the step counter so that TrainState.step matches
    state = state.replace(
        step=step,
        ema_params=ema_params,
        ema_step=ema_step,
        rng=state_rng,
    )
    
    print(f"  Loaded params, EMA (ema_step={ema_step}), step={step}")
    return state


if __name__ == "__main__":
    main()
