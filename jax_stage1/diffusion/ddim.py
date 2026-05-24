"""DDIM deterministic sampler — pure JAX, no PyTorch.

Implements the Denoising Diffusion Implicit Model (DDIM) sampling procedure
for fast, deterministic inference.  The main loop uses ``jax.lax.fori_loop``
for efficient compilation on TPU / GPU.

All tensors use **NHWC** layout ``(B, H, W, C)``.

Reference config (IMPOSE Stage 1):
    ddim_steps=50, ddim_eta=0.0 (deterministic),
    latent shape (B, 128, 128, 3)
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Callable, Dict, Tuple

# Type aliases.
Array = jnp.ndarray
Params = Any  # Flax / Haiku parameter pytree
DDIMSchedule = Dict[str, Array]


# ---------------------------------------------------------------------------
# Schedule construction
# ---------------------------------------------------------------------------

def make_ddim_timesteps(
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int = 1000,
) -> np.ndarray:
    """Create a DDIM timestep sub-sequence via uniform sub-sampling.

    With the default values (50 DDIM steps out of 1000 DDPM steps) this
    selects every 20-th timestep and adds 1 so that the alpha look-ups
    correspond to the correct cumulative products.

    Args:
        num_ddim_timesteps: Number of DDIM sampling steps (*S*).
        num_ddpm_timesteps: Total number of DDPM training timesteps (*T*).

    Returns:
        1-D NumPy int array of shape ``(S,)`` with the selected timesteps.
    """
    c = num_ddpm_timesteps // num_ddim_timesteps
    ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    # Shift by 1 so that alpha_bar indexing is correct.
    steps_out = ddim_timesteps + 1
    return steps_out


def make_ddim_schedule(
    alphas_cumprod: Array,
    ddim_timesteps: np.ndarray,
    eta: float = 0.0,
) -> DDIMSchedule:
    """Precompute the DDIM sampling schedule from DDPM alpha-bars.

    Args:
        alphas_cumprod: Full DDPM cumulative-product alphas, shape ``(T,)``.
        ddim_timesteps: Sub-sampled timestep indices from
            :func:`make_ddim_timesteps`, shape ``(S,)``.
        eta: Stochasticity parameter.  ``eta=0`` gives fully deterministic
            (DDIM) sampling; ``eta=1`` recovers DDPM.

    Returns:
        Dictionary with keys (all ``jnp.float32`` arrays of shape ``(S,)``):

        - ``ddim_timesteps`` — the integer timestep indices.
        - ``ddim_alphas`` — ᾱ at each DDIM step.
        - ``ddim_alphas_prev`` — ᾱ at the *previous* DDIM step.
        - ``ddim_sigmas`` — σ_t (zero when ``eta=0``).
        - ``ddim_sqrt_one_minus_alphas`` — √(1 − ᾱ).
    """
    ddim_alphas = alphas_cumprod[ddim_timesteps]  # (S,)
    ddim_alphas_prev = jnp.concatenate(
        [alphas_cumprod[:1], alphas_cumprod[ddim_timesteps[:-1]]]
    )  # (S,)

    ddim_sigmas = eta * jnp.sqrt(
        (1.0 - ddim_alphas_prev)
        / (1.0 - ddim_alphas)
        * (1.0 - ddim_alphas / ddim_alphas_prev)
    )
    ddim_sqrt_one_minus_alphas = jnp.sqrt(1.0 - ddim_alphas)

    return {
        "ddim_timesteps": jnp.array(ddim_timesteps, dtype=jnp.int32),
        "ddim_alphas": ddim_alphas,
        "ddim_alphas_prev": ddim_alphas_prev,
        "ddim_sigmas": ddim_sigmas,
        "ddim_sqrt_one_minus_alphas": ddim_sqrt_one_minus_alphas,
    }


# ---------------------------------------------------------------------------
# Single DDIM step
# ---------------------------------------------------------------------------

def ddim_sample_step(
    apply_fn: Callable,
    params: Params,
    x: Array,
    t_index: int,
    ddim_schedule: DDIMSchedule,
    rng: jax.Array,
) -> Array:
    """Execute a single DDIM denoising step.

    Given the current noisy sample *x* at schedule index *t_index*, predict
    the noise with the UNet and compute the DDIM update:

    .. math::
        x_{t-1} = \\sqrt{\\bar\\alpha_{t-1}}\\, \\hat x_0
                  + \\sqrt{1 - \\bar\\alpha_{t-1} - \\sigma_t^2}\\, \\epsilon_\\theta
                  + \\sigma_t\\, z

    where *z* ~ N(0, I) (only contributes when ``eta > 0``).

    Args:
        apply_fn: UNet forward function with signature
            ``(params, x, timesteps, deterministic=True) -> eps_pred``.
        params: UNet parameter pytree (or EMA parameters).
        x: Current noisy sample, shape ``(B, H, W, C)``.
        t_index: **Scalar** index into the DDIM schedule (0 … S-1).
        ddim_schedule: Dict from :func:`make_ddim_schedule`.
        rng: JAX PRNG key (used only when ``eta > 0``).

    Returns:
        Denoised sample x_{t-1}, same shape as *x*.
    """
    # Look up schedule values for this index.
    timestep = ddim_schedule["ddim_timesteps"][t_index]  # actual DDPM t
    timesteps_batch = jnp.full((x.shape[0],), timestep, dtype=jnp.int32)

    alpha = ddim_schedule["ddim_alphas"][t_index]
    alpha_prev = ddim_schedule["ddim_alphas_prev"][t_index]
    sigma = ddim_schedule["ddim_sigmas"][t_index]
    sqrt_one_minus_alpha = ddim_schedule["ddim_sqrt_one_minus_alphas"][t_index]

    # ---- Predict noise ε_θ(x_t, t) ----
    eps_pred = apply_fn(params, x, timesteps_batch, deterministic=True)

    # ---- Predict x_0 ----
    pred_x0 = (x - sqrt_one_minus_alpha * eps_pred) / jnp.sqrt(alpha)

    # ---- "Direction pointing to x_t" ----
    dir_xt = jnp.sqrt(1.0 - alpha_prev - sigma ** 2) * eps_pred

    # ---- Stochastic component (zero when eta=0) ----
    noise = jax.random.normal(rng, x.shape) * sigma

    # ---- DDIM update ----
    x_prev = jnp.sqrt(alpha_prev) * pred_x0 + dir_xt + noise

    return x_prev


# ---------------------------------------------------------------------------
# Full sampling loop
# ---------------------------------------------------------------------------

def ddim_sample_loop(
    apply_fn: Callable,
    params: Params,
    shape: Tuple[int, ...],
    ddim_schedule: DDIMSchedule,
    rng: jax.Array,
    num_steps: int = 50,
) -> Array:
    """Run the complete DDIM sampling loop.

    Starting from pure Gaussian noise, iteratively denoise using
    :func:`ddim_sample_step` for *num_steps* steps (from schedule index
    S-1 down to 0).

    The loop is implemented with ``jax.lax.fori_loop`` so that the entire
    sampling procedure compiles into a single XLA program — critical for
    efficient TPU execution.

    Args:
        apply_fn: UNet forward function (see :func:`ddim_sample_step`).
        params: UNet parameter pytree (or EMA parameters).
        shape: Desired output shape ``(B, H, W, C)``, e.g.
            ``(16, 128, 128, 3)``.
        ddim_schedule: Dict from :func:`make_ddim_schedule`.
        rng: JAX PRNG key.
        num_steps: Number of DDIM denoising steps.

    Returns:
        Generated samples of shape *shape*.
    """
    rng, init_rng = jax.random.split(rng)
    x = jax.random.normal(init_rng, shape)  # Start from pure noise.

    def body_fn(i: int, carry: Tuple[Array, jax.Array]) -> Tuple[Array, jax.Array]:
        """Single iteration: denoise at schedule index (S-1-i)."""
        x, rng = carry
        # Count down: S-1, S-2, …, 0.
        t_index = num_steps - 1 - i
        rng, step_rng = jax.random.split(rng)
        x = ddim_sample_step(apply_fn, params, x, t_index, ddim_schedule, step_rng)
        return (x, rng)

    x, _ = jax.lax.fori_loop(0, num_steps, body_fn, (x, rng))

    return x
