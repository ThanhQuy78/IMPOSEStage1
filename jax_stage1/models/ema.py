"""
EMA (Exponential Moving Average) utilities for JAX pytrees.

Port of ldm.modules.ema.LitEma — adapted to JAX's functional style.
Instead of maintaining shadow buffers inside an nn.Module, we operate
directly on pytree leaves (the model params dict).

Usage:
    ema_params = create_ema_state(params)
    # In the training loop, after each optimizer step:
    ema_params = ema_update(ema_params, params, decay=0.9999, step=step)
"""

import jax
import jax.numpy as jnp


def create_ema_state(params):
    """Initialize EMA state as an independent copy of model params.

    Args:
        params: A JAX pytree of model parameters (e.g. from
                ``model.init(rng, ...)['params']``).

    Returns:
        A pytree with the same structure, where every leaf is a copy
        of the corresponding parameter array.
    """
    return jax.tree.map(lambda p: jnp.array(p), params)


def ema_update(ema_params, model_params, decay=0.9999, step=0):
    """Update EMA parameters with warm-up schedule.

    Mirrors the PyTorch ``LitEma.forward`` logic::

        effective_decay = min(decay, (1 + num_updates) / (10 + num_updates))
        shadow = effective_decay * shadow + (1 - effective_decay) * param

    The warm-up ramp ensures that early in training (when ``step`` is
    small) the EMA tracks the fast-moving params more closely instead
    of being anchored to random initialisation.

    Args:
        ema_params:   Current EMA pytree (same structure as model_params).
        model_params: Latest model pytree after the optimizer step.
        decay:        Target EMA decay rate (default 0.9999).
        step:         Current training step (int or scalar array).

    Returns:
        Updated EMA pytree.
    """
    step = jnp.asarray(step, dtype=jnp.float32)
    effective_decay = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
    return jax.tree.map(
        lambda ema, p: effective_decay * ema + (1.0 - effective_decay) * p,
        ema_params,
        model_params,
    )


def copy_ema_to_params(ema_params):
    """Return a detached copy of the EMA pytree for evaluation.

    This is the JAX equivalent of ``LitEma.copy_to`` — since JAX params
    are immutable arrays we simply return the tree as-is (no in-place
    mutation needed).

    Args:
        ema_params: The EMA pytree.

    Returns:
        The same pytree (leaves are already plain ``jnp.ndarray``).
    """
    return ema_params
