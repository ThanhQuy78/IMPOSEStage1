"""Diffusion module for JAX-based DDPM/DDIM implementation.

This package provides pure-JAX implementations of:
- Beta schedule computation (schedule.py)
- DDPM training functions (ddpm.py)
- DDIM deterministic sampler (ddim.py)
"""

from .schedule import make_beta_schedule, compute_schedule_constants
from .ddpm import (
    extract,
    q_sample,
    p_losses,
    predict_start_from_noise,
    q_posterior,
)
from .ddim import (
    make_ddim_timesteps,
    make_ddim_schedule,
    ddim_sample_step,
    ddim_sample_loop,
)

__all__ = [
    # Schedule
    "make_beta_schedule",
    "compute_schedule_constants",
    # DDPM
    "extract",
    "q_sample",
    "p_losses",
    "predict_start_from_noise",
    "q_posterior",
    # DDIM
    "make_ddim_timesteps",
    "make_ddim_schedule",
    "ddim_sample_step",
    "ddim_sample_loop",
]
