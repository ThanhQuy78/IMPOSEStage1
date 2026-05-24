"""Beta schedule computation for diffusion models.

Pure functions for computing diffusion noise schedules and all derived
constants needed by DDPM training and DDIM sampling.

All outputs are JAX arrays (jnp.float32) ready for use on GPU/TPU.

Reference config (IMPOSE Stage 1):
    schedule="linear", n_timestep=1000,
    linear_start=0.0015, linear_end=0.0155
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict


def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
) -> np.ndarray:
    """Create beta schedule matching IMPOSE.

    For 'linear' schedule:
        betas = linspace(sqrt(linear_start), sqrt(linear_end), n_timestep) ** 2

    This is the sqrt-linear spacing used by the original IMPOSE code, which
    produces a smoother noise ramp compared to naive linear spacing.

    Args:
        schedule: Schedule type. Currently only ``"linear"`` is supported.
        n_timestep: Number of diffusion timesteps *T*.
        linear_start: Minimum beta value (at t=0).
        linear_end: Maximum beta value (at t=T-1).

    Returns:
        1-D NumPy array of shape ``(n_timestep,)`` with beta values.

    Raises:
        ValueError: If *schedule* is not ``"linear"``.
    """
    if schedule == "linear":
        betas = (
            np.linspace(
                linear_start ** 0.5,
                linear_end ** 0.5,
                n_timestep,
                dtype=np.float64,
            )
            ** 2
        )
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


def compute_schedule_constants(betas: np.ndarray) -> Dict[str, jnp.ndarray]:
    """Precompute **all** diffusion schedule constants from a beta array.

    The returned dictionary contains everything needed for both the forward
    diffusion process *q(x_t | x_0)* and the reverse posterior
    *q(x_{t-1} | x_t, x_0)*.

    Args:
        betas: 1-D array of shape ``(T,)`` — output of
            :func:`make_beta_schedule`.

    Returns:
        Dictionary with the following ``jnp.float32`` arrays, each of shape
        ``(T,)`` (or ``(T+1,)`` for ``alphas_cumprod_prev``):

        Forward process — *q(x_t | x_0)*:
            - ``betas``
            - ``alphas_cumprod``  — ᾱ_t = ∏_{s=1}^{t} (1 - β_s)
            - ``alphas_cumprod_prev`` — [1, ᾱ_0, ᾱ_1, …, ᾱ_{T-2}]
            - ``sqrt_alphas_cumprod`` — √ᾱ_t
            - ``sqrt_one_minus_alphas_cumprod`` — √(1 - ᾱ_t)
            - ``log_one_minus_alphas_cumprod`` — log(1 - ᾱ_t)
            - ``sqrt_recip_alphas_cumprod`` — 1 / √ᾱ_t
            - ``sqrt_recipm1_alphas_cumprod`` — √(1/ᾱ_t − 1)

        Posterior — *q(x_{t-1} | x_t, x_0)*:
            - ``posterior_variance``
            - ``posterior_log_variance_clipped`` — log-var clipped at 1e-20
            - ``posterior_mean_coef1``
            - ``posterior_mean_coef2``
    """
    # Work in float64 for numerical precision, convert to float32 at the end.
    betas = np.asarray(betas, dtype=np.float64)

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

    # ---------- q(x_t | x_0) ----------
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = np.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1.0)

    # ---------- q(x_{t-1} | x_t, x_0) ----------
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_log_variance_clipped = np.log(
        np.maximum(posterior_variance, 1e-20)
    )
    posterior_mean_coef1 = (
        betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coef2 = (
        (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    # ---------- Pack & convert to jnp.float32 ----------
    constants = {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
    return {k: jnp.array(v, dtype=jnp.float32) for k, v in constants.items()}
