"""DDPM training functions — pure JAX, no PyTorch.

Implements the forward diffusion process *q(x_t | x_0)*, the training loss
computation, and helper functions for predicting x_0 from noise and computing
the reverse posterior.

All functions operate on **NHWC** layout (batch, height, width, channels) and
are compatible with ``jax.jit``.

Reference config (IMPOSE Stage 1):
    timesteps=1000, linear_start=0.0015, linear_end=0.0155,
    loss_type='l1', parameterization='eps'
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Tuple

# Type aliases for readability.
Array = jnp.ndarray
Schedule = Dict[str, Array]
Params = Any  # Flax / Haiku parameter pytree


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def extract(a: Array, t: Array, x_shape: tuple) -> Array:
    """Index into 1-D schedule array *a* at positions *t*, then broadcast.

    Args:
        a: 1-D array of shape ``(T,)`` — a schedule constant.
        t: 1-D array of shape ``(B,)`` — per-sample timestep indices.
        x_shape: Target tensor shape, e.g. ``(B, H, W, C)`` (NHWC).

    Returns:
        Array of shape ``(B, 1, 1, 1)`` (or the appropriate number of
        trailing singleton dims) for broadcasting with NHWC tensors.
    """
    b = t.shape[0]
    out = a[t]  # (B,)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # (B, 1, …, 1)


# ---------------------------------------------------------------------------
# Forward diffusion
# ---------------------------------------------------------------------------

def q_sample(
    x_start: Array,
    t: Array,
    noise: Array,
    schedule: Schedule,
) -> Array:
    """Forward diffusion: sample x_t given x_0.

    .. math::
        x_t = \\sqrt{\\bar\\alpha_t}\\, x_0
              + \\sqrt{1 - \\bar\\alpha_t}\\, \\epsilon

    Args:
        x_start: Clean latent, shape ``(B, H, W, C)``.
        t: Timestep indices, shape ``(B,)``.
        noise: i.i.d. Gaussian noise, same shape as *x_start*.
        schedule: Dict of precomputed schedule constants.

    Returns:
        Noisy latent *x_t*, same shape as *x_start*.
    """
    return (
        extract(schedule["sqrt_alphas_cumprod"], t, x_start.shape) * x_start
        + extract(schedule["sqrt_one_minus_alphas_cumprod"], t, x_start.shape) * noise
    )


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def p_losses(
    apply_fn: Callable,
    params: Params,
    x_start: Array,
    t: Array,
    rng: jax.Array,
    schedule: Schedule,
    loss_type: str = "l1",
    parameterization: str = "eps",
) -> Tuple[Array, Dict[str, Array]]:
    """Compute the simplified diffusion training loss.

    1. Sample noise ε ~ N(0, I).
    2. Corrupt *x_start* → *x_noisy* via :func:`q_sample`.
    3. Predict ε (or x_0) with the UNet.
    4. Compute per-sample loss and average over the batch.

    Args:
        apply_fn: UNet forward function with signature
            ``(params, x_noisy, t, deterministic=True) -> prediction``.
        params: UNet parameter pytree.
        x_start: Clean latent, shape ``(B, H, W, C)``.
        t: Timestep indices, shape ``(B,)``.
        rng: JAX PRNG key for noise sampling.
        schedule: Dict of precomputed schedule constants.
        loss_type: ``"l1"`` (MAE) or ``"l2"`` (MSE).
        parameterization: ``"eps"`` (predict noise) or ``"x0"`` (predict
            clean image).

    Returns:
        ``(loss, info)`` where *loss* is a scalar and *info* is a dict
        containing ``"loss_simple"``.

    Raises:
        ValueError: If *loss_type* or *parameterization* is unknown.
    """
    noise = jax.random.normal(rng, x_start.shape)
    x_noisy = q_sample(x_start, t, noise, schedule)

    model_out = apply_fn(params, x_noisy, t, deterministic=True)

    # Select target depending on parameterization.
    if parameterization == "eps":
        target = noise
    elif parameterization == "x0":
        target = x_start
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # Per-element loss.
    if loss_type == "l1":
        loss = jnp.abs(model_out - target)
    elif loss_type == "l2":
        loss = (model_out - target) ** 2
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Mean over spatial dims per sample, then mean over batch.
    loss = loss.mean(axis=(1, 2, 3))  # (B,)
    loss_simple = loss.mean()

    return loss_simple, {"loss_simple": loss_simple}


# ---------------------------------------------------------------------------
# Reverse helpers
# ---------------------------------------------------------------------------

def predict_start_from_noise(
    x_t: Array,
    t: Array,
    noise: Array,
    schedule: Schedule,
) -> Array:
    """Recover the predicted x_0 from x_t and the predicted noise.

    .. math::
        \\hat x_0 = \\frac{1}{\\sqrt{\\bar\\alpha_t}} x_t
                   - \\sqrt{\\frac{1}{\\bar\\alpha_t} - 1}\\, \\epsilon

    Args:
        x_t: Noisy latent, shape ``(B, H, W, C)``.
        t: Timestep indices, shape ``(B,)``.
        noise: Predicted noise, same shape as *x_t*.
        schedule: Dict of precomputed schedule constants.

    Returns:
        Predicted clean latent x̂_0, same shape as *x_t*.
    """
    return (
        extract(schedule["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t
        - extract(schedule["sqrt_recipm1_alphas_cumprod"], t, x_t.shape) * noise
    )


def q_posterior(
    x_start: Array,
    x_t: Array,
    t: Array,
    schedule: Schedule,
) -> Tuple[Array, Array, Array]:
    """Compute the posterior q(x_{t-1} | x_t, x_0).

    Args:
        x_start: Predicted clean latent x̂_0, shape ``(B, H, W, C)``.
        x_t: Noisy latent at step *t*, same shape.
        t: Timestep indices, shape ``(B,)``.
        schedule: Dict of precomputed schedule constants.

    Returns:
        ``(posterior_mean, posterior_variance, posterior_log_variance_clipped)``
        each with shape ``(B, 1, 1, 1)`` broadcast-ready or full spatial shape.
    """
    posterior_mean = (
        extract(schedule["posterior_mean_coef1"], t, x_t.shape) * x_start
        + extract(schedule["posterior_mean_coef2"], t, x_t.shape) * x_t
    )
    posterior_variance = extract(
        schedule["posterior_variance"], t, x_t.shape
    )
    posterior_log_variance_clipped = extract(
        schedule["posterior_log_variance_clipped"], t, x_t.shape
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped
