"""Teacher-only score functions for UED experiments.

These functions never enter the PPO loss. They only map a rollout to a level
priority used by :class:`jaxued.level_sampler.LevelSampler`.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp


def compute_mna_advantages(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Compute the DEGen Maximised Negative Advantage (MNA) signal.

    This is a direct JAX adaptation of the reference implementation released
    with Mead et al. (2026), "Dynamic Environment Generation for UED".  It
    accumulates all n-step TD errors for every rollout start, clips cumulative
    errors at zero after every extension (the running max-value baseline), and
    lambda-mixes the resulting non-positive advantages.

    Args:
        gamma: Discount factor.
        lambd: Lambda weighting. Must be strictly below one.
        last_value: Bootstrap values with shape ``(num_envs,)``.
        values: Critic values with shape ``(num_steps, num_envs)``.
        rewards: Rewards with shape ``(num_steps, num_envs)``.
        dones: Episode termination flags with the same leading shape.

    Returns:
        The MNA advantages and corresponding value targets. These are teacher
        diagnostics only and must not replace PPO's ordinary GAE targets.
    """
    extended_values = jnp.concatenate((values, last_value[None, ...]), axis=0)
    deltas = rewards + gamma * extended_values[1:] * (1 - dones) - extended_values[:-1]

    num_steps = values.shape[0]
    start_index = jnp.arange(num_steps)

    def accumulate(carry, step):
        lambda_sum, cumulative_td, terminated, current_index = carry
        delta, done = step

        active_start = current_index >= start_index
        delta = delta[None, ...] * active_start[..., None]
        done = done[None, ...] * active_start[..., None]
        horizon = (current_index - start_index)[..., None]

        cumulative_td = cumulative_td + gamma**horizon * delta
        clipped_td = jnp.minimum(cumulative_td, 0.0)
        # Clipping the running cumulative error implements the max-value
        # reference used by MNA, rather than clipping only the final horizon.
        cumulative_td = clipped_td

        alive = 1 - terminated
        lambda_sum = lambda_sum + lambd**horizon * clipped_td * alive
        lambda_sum = lambda_sum + (
            lambd ** (horizon + 1) / (1 - lambd)
        ) * clipped_td * done * alive
        terminated = jnp.logical_or(terminated, done)

        return (lambda_sum, cumulative_td, terminated, current_index + 1), None

    initial = (
        jnp.zeros_like(values),
        jnp.zeros_like(values),
        jnp.zeros_like(dones),
        0,
    )
    (lambda_sum, cumulative_td, terminated, current_index), _ = jax.lax.scan(
        accumulate,
        initial,
        (deltas, dones),
        unroll=16,
    )

    horizon = (current_index - start_index)[..., None]
    clipped_td = jnp.minimum(cumulative_td, 0.0)
    advantages = (1 - lambd) * (
        lambda_sum
        + lambd**horizon / (1 - lambd) * clipped_td * (1 - terminated)
    )
    return advantages, advantages + values


def solved_mna_score(mna_advantages: chex.Array, solved: chex.Array) -> chex.Array:
    """Aggregate MNA per level and gate out levels never solved by the agent."""
    return -jnp.minimum(mna_advantages, 0.0).sum(axis=0) * solved
