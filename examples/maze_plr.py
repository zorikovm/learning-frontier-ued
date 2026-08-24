import json
import time
from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import core, struct
from flax.training.train_state import TrainState as BaseTrainState
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import optax
import distrax
import os
import orbax.checkpoint as ocp
import wandb
from jaxued.environments.underspecified_env import EnvParams, EnvState, Observation, UnderspecifiedEnv
from jaxued.linen import ResetRNN
from jaxued.environments import Maze, MazeRenderer
from jaxued.environments.maze import Level, make_level_generator, make_level_mutator_minimax
from jaxued.level_sampler import LevelSampler
from jaxued.teacher_scores import compute_mna_advantages, solved_mna_score
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper
import chex
from enum import IntEnum

class UpdateState(IntEnum):
    DR = 0
    REPLAY = 1

class TrainState(BaseTrainState):
    sampler: core.FrozenDict[str, chex.ArrayTree] = struct.field(pytree_node=True)
    update_state: UpdateState = struct.field(pytree_node=True)
    # === Below is used for logging ===
    num_dr_updates: int
    num_replay_updates: int
    num_mutation_updates: int
    dr_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    replay_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)
    mutation_last_level_batch: chex.ArrayTree = struct.field(pytree_node=True)

# region PPO helper functions
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float): 
        lambd (float): 
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
    def compute_gae_at_timestep(carry, x):
        gae, next_value = carry
        value, reward, done = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lambd * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        compute_gae_at_timestep,
        (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values

def sample_trajectories_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    num_envs: int,
    max_episode_length: int,
) -> Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]:
    """This samples trajectories from the environment using the agent specified by the `train_state`.

    Args:

        rng (chex.PRNGKey): Singleton 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): Singleton
        init_hstate (chex.ArrayTree): This is the init RNN hidden state, has to have shape (NUM_ENVS, ...)
        init_obs (Observation): The initial observation, shape (NUM_ENVS, ...)
        init_env_state (EnvState): The initial env state (NUM_ENVS, ...)
        num_envs (int): The number of envs that are vmapped over.
        max_episode_length (int): The maximum episode length, i.e., the number of steps to do the rollouts for.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState, chex.ArrayTree, Observation, EnvState, chex.Array], Tuple[Observation, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, dict]]: (rng, train_state, hstate, last_obs, last_env_state, last_value), traj, where traj is (obs, action, reward, done, log_prob, value, info). The first element in the tuple consists of arrays that have shapes (NUM_ENVS, ...) (except `rng` and and `train_state` which are singleton). The second element in the tuple is of shape (NUM_STEPS, NUM_ENVS, ...), and it contains the trajectory.
    """
    def sample_step(carry, _):
        rng, train_state, hstate, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, last_done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action)
        log_prob = pi.log_prob(action)
        value, action, log_prob = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        next_obs, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_envs), env_state, action, env_params)

        carry = (rng, train_state, hstate, next_obs, env_state, done)
        return carry, (obs, action, reward, done, log_prob, value, info)

    (rng, train_state, hstate, last_obs, last_env_state, last_done), traj = jax.lax.scan(
        sample_step,
        (
            rng,
            train_state,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_envs, dtype=bool),
        ),
        None,
        length=max_episode_length,
    )

    x = jax.tree_util.tree_map(lambda x: x[None, ...], (last_obs, last_done))
    _, _, last_value = train_state.apply_fn(train_state.params, x, hstate)

    return (rng, train_state, hstate, last_obs, last_env_state, last_value.squeeze(0)), traj

def evaluate_rnn(
    rng: chex.PRNGKey,
    env: UnderspecifiedEnv,
    env_params: EnvParams,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    init_obs: Observation,
    init_env_state: EnvState,
    max_episode_length: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """This runs the RNN on the environment, given an initial state and observation, and returns (states, rewards, episode_lengths)

    Args:
        rng (chex.PRNGKey): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): Shape (num_levels, )
        init_obs (Observation): Shape (num_levels, )
        init_env_state (EnvState): Shape (num_levels, )
        max_episode_length (int): 

    Returns:
        Tuple[chex.Array, chex.Array, chex.Array]: (States, rewards, episode lengths) ((NUM_STEPS, NUM_LEVELS), (NUM_STEPS, NUM_LEVELS), (NUM_LEVELS,)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]
    
    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        
        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)
    
    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (
            rng,
            init_hstate,
            init_obs,
            init_env_state,
            jnp.zeros(num_levels, dtype=bool),
            jnp.ones(num_levels, dtype=bool),
            jnp.zeros(num_levels, dtype=jnp.int32),
        ),
        None,
        length=max_episode_length,
    )

    return states, rewards, episode_lengths

def update_actor_critic_rnn(
    rng: chex.PRNGKey,
    train_state: TrainState,
    init_hstate: chex.ArrayTree,
    batch: chex.ArrayTree,
    num_envs: int,
    n_steps: int,
    n_minibatch: int,
    n_epochs: int,
    clip_eps: float,
    entropy_coeff: float,
    critic_coeff: float,
    update_grad: bool=True,
) -> Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]:
    """This function takes in a rollout, and PPO hyperparameters, and updates the train state.

    Args:
        rng (chex.PRNGKey): 
        train_state (TrainState): 
        init_hstate (chex.ArrayTree): 
        batch (chex.ArrayTree): obs, actions, dones, log_probs, values, targets, advantages
        num_envs (int): 
        n_steps (int): 
        n_minibatch (int): 
        n_epochs (int): 
        clip_eps (float): 
        entropy_coeff (float): 
        critic_coeff (float): 
        update_grad (bool, optional): If False, the train state does not actually get updated. Defaults to True.

    Returns:
        Tuple[Tuple[chex.PRNGKey, TrainState], chex.ArrayTree]: It returns a new rng, the updated train_state, and the losses. The losses have structure (loss, (l_vf, l_clip, entropy))
    """
    obs, actions, dones, log_probs, values, targets, advantages = batch
    last_dones = jnp.roll(dones, 1, axis=0).at[0].set(False)
    batch = obs, actions, last_dones, log_probs, values, targets, advantages
    
    def update_epoch(carry, _):
        def update_minibatch(train_state, minibatch):
            init_hstate, obs, actions, last_dones, log_probs, values, targets, advantages = minibatch
            
            def loss_fn(params):
                _, pi, values_pred = train_state.apply_fn(params, (obs, last_dones), init_hstate)
                log_probs_pred = pi.log_prob(actions)
                entropy = pi.entropy().mean()

                ratio = jnp.exp(log_probs_pred - log_probs)
                A = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
                l_clip = (-jnp.minimum(ratio * A, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * A)).mean()

                values_pred_clipped = values + (values_pred - values).clip(-clip_eps, clip_eps)
                l_vf = 0.5 * jnp.maximum((values_pred - targets) ** 2, (values_pred_clipped - targets) ** 2).mean()

                loss = l_clip + critic_coeff * l_vf - entropy_coeff * entropy

                return loss, (l_vf, l_clip, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            loss, grads = grad_fn(train_state.params)
            if update_grad:
                train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        rng, train_state = carry
        rng, rng_perm = jax.random.split(rng)
        permutation = jax.random.permutation(rng_perm, num_envs)
        minibatches = (
            jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0)
                .reshape(n_minibatch, -1, *x.shape[1:]),
                init_hstate,
            ),
            *jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1)
                .reshape(x.shape[0], n_minibatch, -1, *x.shape[2:])
                .swapaxes(0, 1),
                batch,
            ),
        )
        train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
        return (rng, train_state), losses

    return jax.lax.scan(update_epoch, (rng, train_state), None, n_epochs)

class ActorCritic(nn.Module):
    """This is an actor critic class that uses an LSTM
    """
    action_dim: Sequence[int]
    
    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs
        
        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)
        
        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)
        
        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    
    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))
# endregion

# region checkpointing
def setup_checkpointing(config: dict, train_state: TrainState, env: UnderspecifiedEnv, env_params: EnvParams) -> ocp.CheckpointManager:
    """This takes in the train state and config, and returns an orbax checkpoint manager.
        It also saves the config in `checkpoints/run_name/seed/config.json`

    Args:
        config (dict): 
        train_state (TrainState): 
        env (UnderspecifiedEnv): 
        env_params (EnvParams): 

    Returns:
        ocp.CheckpointManager: 
    """
    overall_save_dir = os.path.join(os.getcwd(), "checkpoints", f"{config['run_name']}", str(config['seed']))
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # save the config
    with open(os.path.join(overall_save_dir, 'config.json'), 'w+') as f:
        f.write(json.dumps(config.as_dict(), indent=True))
    
    checkpoint_manager = ocp.CheckpointManager(
        os.path.join(overall_save_dir, 'models'),
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=config['checkpoint_save_interval'],
            max_to_keep=config['max_number_of_checkpoints'],
        )
    )
    return checkpoint_manager
#endregion

def train_state_to_log_dict(train_state: TrainState, level_sampler: LevelSampler) -> dict:
    """To prevent the entire (large) train_state to be copied to the CPU when doing logging, this function returns all of the important information in a dictionary format.

        Anything in the `log` key will be logged to wandb.
    
    Args:
        train_state (TrainState): 
        level_sampler (LevelSampler): 

    Returns:
        dict: 
    """
    sampler = train_state.sampler
    idx = jnp.arange(level_sampler.capacity) < sampler["size"]
    finite_idx = idx & jnp.isfinite(sampler["scores"])
    s = jnp.maximum(finite_idx.sum(), 1)
    finite_scores = jnp.where(finite_idx, sampler["scores"], 0.0)
    return {
        "log":{
            "level_sampler/size": sampler["size"],
            "level_sampler/episode_count": sampler["episode_count"],
            "level_sampler/max_score": jnp.where(finite_idx, sampler["scores"], -jnp.inf).max(),
            "level_sampler/weighted_score": (finite_scores * level_sampler.level_weights(sampler)).sum(),
            "level_sampler/mean_score": finite_scores.sum() / s,
            "level_sampler/finite_score_fraction": finite_idx.sum() / jnp.maximum(idx.sum(), 1),
        },
        "info": {
            "num_dr_updates": train_state.num_dr_updates,
            "num_replay_updates": train_state.num_replay_updates,
            "num_mutation_updates": train_state.num_mutation_updates,
        }
    }

def compute_score(config, dones, values, max_returns, advantages):
    if config['score_function'] == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif config['score_function'] == "pvl":
        return positive_value_loss(dones, advantages)
    elif config['score_function'] in ("mna", "mna_frontier", "mna_frontier_lp"):
        return solved_mna_score(advantages, max_returns > 0)
    elif config['score_function'] == "frontier":
        raise ValueError("frontier score must be supplied by update_frontier_history")
    elif config['score_function'] == "frontier_lp":
        raise ValueError("frontier_lp score must be supplied by update_frontier_history")
    else:
        raise ValueError(f"Unknown score function: {config['score_function']}")

def compute_score_advantages(config, last_value, values, rewards, dones, advantages):
    """Return teacher advantages without changing PPO's ordinary GAE."""
    if config["score_function"] in ("mna", "mna_frontier", "mna_frontier_lp"):
        score_advantages, _ = compute_mna_advantages(
            config["gamma"],
            config["gae_lambda"],
            last_value,
            values,
            rewards,
            dones,
        )
        return score_advantages
    return advantages

def initial_level_history(config, batch_size):
    """Teacher state for genuinely new levels.

    The 0.5 prior deliberately represents ignorance, not observed competence.
    A first rollout moves the estimate toward its measured success rate.
    """
    return {
        "max_return": jnp.full(batch_size, -jnp.inf),
        "success_ema": jnp.full(batch_size, config["frontier_prior"]),
        "teacher_visits": jnp.zeros(batch_size, dtype=jnp.int32),
        "successful_episodes": jnp.zeros(batch_size, dtype=jnp.float32),
        "completed_episodes": jnp.zeros(batch_size, dtype=jnp.float32),
    }

def update_frontier_history(config, old_history, dones, rewards):
    """Update cheap per-level competence history from the existing rollout."""
    completed = dones.sum(axis=0).astype(jnp.float32)
    successful = (rewards > 0).sum(axis=0).astype(jnp.float32)
    observed_success = successful / jnp.maximum(completed, 1.0)
    has_observation = completed > 0

    old_p = old_history["success_ema"]
    new_p = jnp.where(
        has_observation,
        old_p + config["frontier_ema_alpha"] * (observed_success - old_p),
        old_p,
    )
    # Scale p(1-p) to [0, 1] for interpretable logging. Rank replay is
    # invariant to this constant, but the scale makes progress coefficients
    # easier to read.
    frontier = 4.0 * new_p * (1.0 - new_p)
    positive_progress = jnp.where(has_observation, jnp.maximum(new_p - old_p, 0.0), 0.0)
    current_max_return = compute_max_returns(dones, rewards)

    history = {
        "max_return": jnp.maximum(old_history["max_return"], current_max_return),
        "success_ema": new_p,
        "teacher_visits": old_history["teacher_visits"] + has_observation.astype(jnp.int32),
        "successful_episodes": old_history["successful_episodes"] + successful,
        "completed_episodes": old_history["completed_episodes"] + completed,
    }
    return history, observed_success, completed, frontier, positive_progress

def frontier_score(config, frontier, positive_progress):
    if config["score_function"] == "frontier":
        return frontier
    if config["score_function"] == "frontier_lp":
        return frontier + config["learning_progress_coeff"] * positive_progress
    return None

def combine_with_frontier(config, base_score, frontier, positive_progress):
    """Optional learnability gate applied after a trajectory-based base score."""
    if config["score_function"] == "mna_frontier":
        return base_score * frontier
    if config["score_function"] == "mna_frontier_lp":
        return base_score * (
            frontier + config["learning_progress_coeff"] * positive_progress
        )
    return base_score

def teacher_rollout_metrics(
    update_kind,
    scores,
    dones,
    values,
    max_returns,
    ppo_advantages,
    observed_success,
    completed,
    frontier,
    positive_progress,
    mutation_edits,
    levels,
    causal_diagnostics,
):
    """Diagnostics shared by every branch of the JAX switch."""
    metrics = {
        "teacher_update_kind": jnp.asarray(update_kind, dtype=jnp.int32),
        "teacher_scores": scores,
        "teacher_maxmc": max_mc(dones, values, max_returns),
        "teacher_pvl": positive_value_loss(dones, ppo_advantages),
        "teacher_success_rate": observed_success,
        "teacher_completed_episodes": completed,
        "teacher_frontier": frontier,
        "teacher_positive_progress": positive_progress,
        "teacher_mutation_edits": jnp.asarray(mutation_edits, dtype=jnp.float32),
    }
    # Full identities enable next-visit causal analysis, but carrying geometry
    # through every scanned update is unnecessary for long multi-seed runs.
    if causal_diagnostics:
        metrics.update({
            "teacher_wall_map": levels.wall_map,
            "teacher_goal_pos": levels.goal_pos,
            "teacher_agent_pos": levels.agent_pos,
            "teacher_agent_dir": levels.agent_dir,
        })
    return metrics

def main(config=None, project="JAXUED_TEST"):
    tags = []
    if not config["exploratory_grad_updates"]:
        tags.append("robust")
    if config["use_accel"]:
        tags.append("ACCEL")
    else:
        tags.append("PLR")
    run = wandb.init(config=config, project=project, group=config["run_name"], tags=tags)
    config = wandb.config
    local_run_name = config["run_name"] or config["group_name"]
    local_result_dir = os.path.join(config["local_results_dir"], local_run_name, str(config["seed"]))
    os.makedirs(local_result_dir, exist_ok=True)
    with open(os.path.join(local_result_dir, "config.json"), "w") as f:
        json.dump(config.as_dict(), f, indent=2)
    
    wandb.define_metric("num_updates")
    wandb.define_metric("num_env_steps")
    wandb.define_metric("solve_rate/*", step_metric="num_updates")
    wandb.define_metric("level_sampler/*", step_metric="num_updates")
    wandb.define_metric("agent/*", step_metric="num_updates")
    wandb.define_metric("return/*", step_metric="num_updates")
    wandb.define_metric("eval_ep_lengths/*", step_metric="num_updates")

    def log_eval(stats, train_state_info):
        print(f"Logging update: {stats['update_count']}")
        
        # generic stats
        env_steps = stats["update_count"] * config["num_train_envs"] * config["num_steps"]
        log_dict = {
            "num_updates": stats["update_count"],
            "num_env_steps": env_steps,
            "sps": env_steps / stats['time_delta'],
            "block_sps": config["eval_freq"] * config["num_train_envs"] * config["num_steps"] / stats['time_delta'],
            "wall_clock_block_seconds": stats["time_delta"],
        }
        
        # evaluation performance
        solve_rates = stats['eval_solve_rates']
        returns     = stats["eval_returns"]
        log_dict.update({f"solve_rate/{name}": solve_rate for name, solve_rate in zip(config["eval_levels"], solve_rates)})
        log_dict.update({"solve_rate/mean": solve_rates.mean()})
        log_dict.update({f"return/{name}": ret for name, ret in zip(config["eval_levels"], returns)})
        log_dict.update({"return/mean": returns.mean()})
        log_dict.update({"eval_ep_lengths/mean": stats['eval_ep_lengths'].mean()})

        validation_solve_rates = np.asarray(stats["validation_solve_rates"])
        validation_returns = np.asarray(stats["validation_returns"])
        log_dict.update({
            "validation/solve_rate_mean": validation_solve_rates.mean(),
            "validation/return_mean": validation_returns.mean(),
        })
        for group_idx, n_walls in enumerate(config["validation_wall_counts"]):
            start = group_idx * validation_per_group
            end = start + validation_per_group
            log_dict[f"validation/solve_rate_walls_{n_walls}"] = validation_solve_rates[start:end].mean()
            log_dict[f"validation/return_walls_{n_walls}"] = validation_returns[start:end].mean()
        
        # level sampler
        log_dict.update(train_state_info["log"])

        # Teacher diagnostics are computed after the training block and cannot
        # influence PPO, replay RNG, or level generation.
        teacher_scores = np.asarray(stats["teacher_scores"]).reshape(-1)
        success_rate = np.asarray(stats["teacher_success_rate"]).reshape(-1)
        completed = np.asarray(stats["teacher_completed_episodes"]).reshape(-1)
        maxmc_scores = np.asarray(stats["teacher_maxmc"]).reshape(-1)
        pvl_scores = np.asarray(stats["teacher_pvl"]).reshape(-1)

        def finite_corr(x, y, extra_mask=None):
            mask = np.isfinite(x) & np.isfinite(y)
            if extra_mask is not None:
                mask &= extra_mask
            if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                return np.nan
            return np.corrcoef(x[mask], y[mask])[0, 1]

        observed_mask = completed > 0
        finite_teacher = np.isfinite(teacher_scores) & observed_mask
        if finite_teacher.any():
            valid_indices = np.flatnonzero(finite_teacher)
            top_count = max(1, int(np.ceil(0.1 * valid_indices.size)))
            top_indices = valid_indices[np.argsort(teacher_scores[valid_indices])[-top_count:]]
            top_success = success_rate[top_indices].mean()
        else:
            top_success = np.nan

        update_kinds = np.asarray(stats["teacher_update_kind"]).reshape(-1)
        mutation_mask = update_kinds == 2
        mutation_edit_counts = np.asarray(stats["teacher_mutation_edits"]).reshape(-1)
        log_dict.update({
            "teacher/score_success_corr": finite_corr(teacher_scores, success_rate, observed_mask),
            "teacher/maxmc_success_corr": finite_corr(maxmc_scores, success_rate, observed_mask),
            "teacher/pvl_success_corr": finite_corr(pvl_scores, success_rate, observed_mask),
            "teacher/top_decile_success_rate": top_success,
            "teacher/mean_observed_success_rate": success_rate[observed_mask].mean() if observed_mask.any() else np.nan,
            "teacher/mean_frontier": np.asarray(stats["teacher_frontier"]).mean(),
            "teacher/mean_positive_progress": np.asarray(stats["teacher_positive_progress"]).mean(),
            "teacher/dr_fraction": (update_kinds == 0).mean(),
            "teacher/replay_fraction": (update_kinds == 1).mean(),
            "teacher/mutation_fraction": mutation_mask.mean(),
            "teacher/mean_mutation_edits": mutation_edit_counts[mutation_mask].mean() if mutation_mask.any() else 0.0,
        })

        if config["diagnostic_logging"]:
            diagnostic_payload = {
                "teacher_scores": np.asarray(stats["teacher_scores"]),
                "maxmc": np.asarray(stats["teacher_maxmc"]),
                "pvl": np.asarray(stats["teacher_pvl"]),
                "success_rate": np.asarray(stats["teacher_success_rate"]),
                "completed_episodes": np.asarray(stats["teacher_completed_episodes"]),
                "frontier": np.asarray(stats["teacher_frontier"]),
                "positive_progress": np.asarray(stats["teacher_positive_progress"]),
                "mutation_edits": np.asarray(stats["teacher_mutation_edits"]),
                "update_kind": np.asarray(stats["teacher_update_kind"]),
            }
            if config["causal_diagnostics"]:
                diagnostic_payload.update({
                    "wall_map": np.asarray(stats["teacher_wall_map"]),
                    "goal_pos": np.asarray(stats["teacher_goal_pos"]),
                    "agent_pos": np.asarray(stats["teacher_agent_pos"]),
                    "agent_dir": np.asarray(stats["teacher_agent_dir"]),
                })
            np.savez_compressed(
                os.path.join(local_result_dir, f"teacher_diagnostics_{int(stats['update_count'])}.npz"),
                **diagnostic_payload,
            )

        local_log_dict = {
            key: np.asarray(value).item() if np.asarray(value).ndim == 0 else np.asarray(value).tolist()
            for key, value in log_dict.items()
        }
        with open(os.path.join(local_result_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(local_log_dict) + "\n")

        if not config["lightweight_logging"]:
            log_dict.update({"images/highest_scoring_level": wandb.Image(np.array(stats["highest_scoring_level"]), caption="Highest scoring level")})
            log_dict.update({"images/highest_weighted_level": wandb.Image(np.array(stats["highest_weighted_level"]), caption="Highest weighted level")})

            for s in ['dr', 'replay', 'mutation']:
                if train_state_info['info'][f'num_{s}_updates'] > 0:
                    log_dict.update({f"images/{s}_levels": [wandb.Image(np.array(image)) for image in stats[f"{s}_levels"]]})

            for i, level_name in enumerate(config["eval_levels"]):
                frames, episode_length = stats["eval_animation"][0][:, i], stats["eval_animation"][1][i]
                frames = np.array(frames[:episode_length])
                log_dict.update({f"animations/{level_name}": wandb.Video(frames, fps=4)})
        
        wandb.log(log_dict)

    def save_sampler_snapshot(train_state, update_count):
        sampler = train_state.sampler
        size = int(np.asarray(sampler["size"]))
        levels = sampler["levels"]
        extras = sampler["levels_extra"]
        payload = {
            "size": np.asarray(size),
            "scores": np.asarray(sampler["scores"][:size]),
            "timestamps": np.asarray(sampler["timestamps"][:size]),
            "weights": np.asarray(level_sampler.level_weights(sampler)[:size]),
            "wall_map": np.asarray(levels.wall_map[:size]),
            "goal_pos": np.asarray(levels.goal_pos[:size]),
            "agent_pos": np.asarray(levels.agent_pos[:size]),
            "agent_dir": np.asarray(levels.agent_dir[:size]),
        }
        payload.update({f"extra_{key}": np.asarray(value[:size]) for key, value in extras.items()})
        np.savez_compressed(
            os.path.join(local_result_dir, f"sampler_{int(update_count)}.npz"),
            **payload,
        )
    
    # Setup the environment
    env = Maze(max_height=13, max_width=13, agent_view_size=config["agent_view_size"], normalize_obs=True)
    eval_env = env
    sample_random_level = make_level_generator(env.max_height, env.max_width, config["n_walls"])
    if config["validation_num_levels"] % len(config["validation_wall_counts"]) != 0:
        raise ValueError("validation_num_levels must be divisible by validation_wall_counts")
    validation_per_group = config["validation_num_levels"] // len(config["validation_wall_counts"])
    validation_batches = []
    validation_key = jax.random.PRNGKey(config["validation_level_seed"])
    for group_idx, n_walls in enumerate(config["validation_wall_counts"]):
        validation_generator = make_level_generator(env.max_height, env.max_width, n_walls)
        group_key = jax.random.fold_in(validation_key, group_idx)
        validation_batches.append(
            jax.vmap(validation_generator)(jax.random.split(group_key, validation_per_group))
        )
    validation_levels = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *validation_batches
    )
    env_renderer = MazeRenderer(env, tile_size=8)
    env = AutoReplayWrapper(env)
    env_params = env.default_params
    mutate_level = make_level_mutator_minimax(100)

    # And the level sampler    
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"], "k": config['topk_k']},
        duplicate_check=config['buffer_duplicate_check'],
    )

    def history_for_candidate_levels(sampler, levels):
        """Reuse history for duplicate candidates; otherwise return the prior."""
        history = initial_level_history(config, config["num_train_envs"])
        if config["score_function"] not in (
            "frontier", "frontier_lp", "mna_frontier", "mna_frontier_lp"
        ):
            return history

        level_inds = jax.vmap(lambda level: level_sampler.find(sampler, level))(levels)
        exists = level_inds >= 0
        stored = level_sampler.get_levels_extra(sampler, jnp.maximum(level_inds, 0))
        return jax.tree_util.tree_map(
            lambda default, old: jnp.where(exists, old, default),
            history,
            stored,
        )
    
    @jax.jit
    def create_train_state(rng) -> TrainState:
        # Creates the train state
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (config["num_minibatches"] * config["epoch_ppo"]))
                / config["num_updates"]
            )
            return config["lr"] * frac
        obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
        obs = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
            obs,
        )
        init_x = (obs, jnp.zeros((256, config["num_train_envs"])))
        network = ActorCritic(env.action_space(env_params).n)
        network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
            # optax.adam(learning_rate=config["lr"], eps=1e-5),
        )
        pholder_level = sample_random_level(jax.random.PRNGKey(0))
        pholder_history = jax.tree_util.tree_map(
            lambda x: x[0], initial_level_history(config, 1)
        )
        sampler = level_sampler.initialize(pholder_level, pholder_history)
        pholder_level_batch = jax.tree_util.tree_map(lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level)
        return TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
            sampler=sampler,
            update_state=0,
            num_dr_updates=0,
            num_replay_updates=0,
            num_mutation_updates=0,
            dr_last_level_batch=pholder_level_batch,
            replay_last_level_batch=pholder_level_batch,
            mutation_last_level_batch=pholder_level_batch,
        )

    def train_step(carry: Tuple[chex.PRNGKey, TrainState], _):
        """
            This is the main training loop. It basically calls either `on_new_levels`, `on_replay_levels`, or `on_mutate_levels` at every step.
        """
        def on_new_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                Samples new (randomly-generated) levels and evaluates the policy on these. It also then adds the levels to the level buffer if they have high-enough scores.
                The agent is updated on these trajectories iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            
            # Reset
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            new_levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, config["num_train_envs"]))
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), new_levels, env_params)
            # Rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            score_advantages = compute_score_advantages(config, last_value, values, rewards, dones, advantages)
            old_history = history_for_candidate_levels(sampler, new_levels)
            history, observed_success, completed, frontier, positive_progress = update_frontier_history(
                config, old_history, dones, rewards
            )
            max_returns = history["max_return"]
            scores = frontier_score(config, frontier, positive_progress)
            if scores is None:
                scores = compute_score(config, dones, values, max_returns, score_advantages)
            scores = combine_with_frontier(config, scores, frontier, positive_progress)
            sampler, _ = level_sampler.insert_batch(sampler, new_levels, scores, history)
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": new_levels.wall_map.sum() / config["num_train_envs"],
                **teacher_rollout_metrics(
                    UpdateState.DR,
                    scores,
                    dones,
                    values,
                    max_returns,
                    advantages,
                    observed_success,
                    completed,
                    frontier,
                    positive_progress,
                    0.0,
                    new_levels,
                    config["causal_diagnostics"],
                ),
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_dr_updates=train_state.num_dr_updates + 1,
                dr_last_level_batch=new_levels,
            )
            return (rng, train_state), metrics
        
        def on_replay_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This samples levels from the level buffer, and updates the policy on them.
            """
            sampler = train_state.sampler
            
            # Collect trajectories on replay levels
            rng, rng_levels, rng_reset = jax.random.split(rng, 3)
            sampler, (level_inds, levels) = level_sampler.sample_replay_levels(sampler, rng_levels, config["num_train_envs"])
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), levels, env_params)
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            score_advantages = compute_score_advantages(config, last_value, values, rewards, dones, advantages)
            old_history = level_sampler.get_levels_extra(sampler, level_inds)
            history, observed_success, completed, frontier, positive_progress = update_frontier_history(
                config, old_history, dones, rewards
            )
            max_returns = history["max_return"]
            scores = frontier_score(config, frontier, positive_progress)
            if scores is None:
                scores = compute_score(config, dones, values, max_returns, score_advantages)
            scores = combine_with_frontier(config, scores, frontier, positive_progress)
            sampler = level_sampler.update_batch(sampler, level_inds, scores, history)
            
            # Update the policy using trajectories collected from replay levels
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=True,
            )
                            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": levels.wall_map.sum() / config["num_train_envs"],
                **teacher_rollout_metrics(
                    UpdateState.REPLAY,
                    scores,
                    dones,
                    values,
                    max_returns,
                    advantages,
                    observed_success,
                    completed,
                    frontier,
                    positive_progress,
                    0.0,
                    levels,
                    config["causal_diagnostics"],
                ),
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.REPLAY,
                num_replay_updates=train_state.num_replay_updates + 1,
                replay_last_level_batch=levels,
            )
            return (rng, train_state), metrics
        
        def on_mutate_levels(rng: chex.PRNGKey, train_state: TrainState):
            """
                This mutates the previous batch of replay levels and potentially adds them to the level buffer.
                This also updates the policy iff `config["exploratory_grad_updates"]` is True.
            """
            sampler = train_state.sampler
            rng, rng_mutate, rng_reset = jax.random.split(rng, 3)
            
            # mutate
            parent_levels = train_state.replay_last_level_batch
            if config["adaptive_mutation"]:
                parent_inds = jax.vmap(lambda level: level_sampler.find(sampler, level))(parent_levels)
                parent_history = level_sampler.get_levels_extra(sampler, jnp.maximum(parent_inds, 0))
                frontier_distance = 2.0 * jnp.abs(parent_history["success_ema"] - 0.5)
                mutation_edits = 1 + jnp.rint(
                    (config["num_edits"] - 1) * frontier_distance
                ).astype(jnp.int32)
            else:
                mutation_edits = jnp.full(
                    config["num_train_envs"], config["num_edits"], dtype=jnp.int32
                )
            child_levels = jax.vmap(mutate_level, (0, 0, 0))(
                jax.random.split(rng_mutate, config["num_train_envs"]),
                parent_levels,
                mutation_edits,
            )
            init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(jax.random.split(rng_reset, config["num_train_envs"]), child_levels, env_params)

            # rollout
            (
                (rng, train_state, hstate, last_obs, last_env_state, last_value),
                (obs, actions, rewards, dones, log_probs, values, info),
            ) = sample_trajectories_rnn(
                rng,
                env,
                env_params,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                init_obs,
                init_env_state,
                config["num_train_envs"],
                config["num_steps"],
            )
            advantages, targets = compute_gae(config["gamma"], config["gae_lambda"], last_value, values, rewards, dones)
            score_advantages = compute_score_advantages(config, last_value, values, rewards, dones, advantages)
            old_history = history_for_candidate_levels(sampler, child_levels)
            history, observed_success, completed, frontier, positive_progress = update_frontier_history(
                config, old_history, dones, rewards
            )
            max_returns = history["max_return"]
            scores = frontier_score(config, frontier, positive_progress)
            if scores is None:
                scores = compute_score(config, dones, values, max_returns, score_advantages)
            scores = combine_with_frontier(config, scores, frontier, positive_progress)
            sampler, _ = level_sampler.insert_batch(sampler, child_levels, scores, history)
            
            # Update: train_state only modified if exploratory_grad_updates is on
            (rng, train_state), losses = update_actor_critic_rnn(
                rng,
                train_state,
                ActorCritic.initialize_carry((config["num_train_envs"],)),
                (obs, actions, dones, log_probs, values, targets, advantages),
                config["num_train_envs"],
                config["num_steps"],
                config["num_minibatches"],
                config["epoch_ppo"],
                config["clip_eps"],
                config["entropy_coeff"],
                config["critic_coeff"],
                update_grad=config["exploratory_grad_updates"],
            )
            
            metrics = {
                "losses": jax.tree_util.tree_map(lambda x: x.mean(), losses),
                "mean_num_blocks": child_levels.wall_map.sum() / config["num_train_envs"],
                **teacher_rollout_metrics(
                    2,
                    scores,
                    dones,
                    values,
                    max_returns,
                    advantages,
                    observed_success,
                    completed,
                    frontier,
                    positive_progress,
                    mutation_edits.mean(),
                    child_levels,
                    config["causal_diagnostics"],
                ),
            }
            
            train_state = train_state.replace(
                sampler=sampler,
                update_state=UpdateState.DR,
                num_mutation_updates=train_state.num_mutation_updates + 1,
                mutation_last_level_batch=child_levels,
            )
            return (rng, train_state), metrics
    
        rng, train_state = carry
        rng, rng_replay = jax.random.split(rng)
        
        # The train step makes a decision on which branch to take, either on_new, on_replay or on_mutate.
        # on_mutate is only called if the replay branch has been taken before (as it uses `train_state.update_state`).
        if config["use_accel"]:
            s = train_state.update_state
            branch = (1 - s) * level_sampler.sample_replay_decision(train_state.sampler, rng_replay) + 2 * s
        else:
            branch = level_sampler.sample_replay_decision(train_state.sampler, rng_replay).astype(int)
        
        return jax.lax.switch(
            branch,
            [
                on_new_levels,
                on_replay_levels,
                on_mutate_levels,
            ],
            rng, train_state
        )
    
    def eval(rng: chex.PRNGKey, train_state: TrainState):
        """
        This evaluates the current policy on the set of evaluation levels specified by config["eval_levels"].
        It returns (states, cum_rewards, episode_lengths), with shapes (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)
        """
        rng, rng_reset = jax.random.split(rng)
        levels = Level.load_prefabs(config["eval_levels"])
        num_levels = len(config["eval_levels"])
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(jax.random.split(rng_reset, num_levels), levels, env_params)
        states, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        cum_rewards = (rewards * mask).sum(axis=0)
        return states, cum_rewards, episode_lengths # (num_steps, num_eval_levels, ...), (num_eval_levels,), (num_eval_levels,)

    def eval_validation(rng: chex.PRNGKey, train_state: TrainState):
        """Evaluate fixed random levels that are never visible to the teacher."""
        num_levels = config["validation_num_levels"]
        rng, rng_reset = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), validation_levels, env_params
        )
        _, rewards, episode_lengths = evaluate_rnn(
            rng,
            eval_env,
            env_params,
            train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs,
            init_env_state,
            env_params.max_steps_in_episode,
        )
        mask = jnp.arange(env_params.max_steps_in_episode)[..., None] < episode_lengths
        return (rewards * mask).sum(axis=0)
    
    @jax.jit
    def train_and_eval_step(runner_state, _):
        """
            This function runs the train_step for a certain number of iterations, and then evaluates the policy.
            It returns the updated train state, and a dictionary of metrics.
        """
        # Train
        (rng, train_state), metrics = jax.lax.scan(train_step, runner_state, None, config["eval_freq"])

        # Eval
        rng, rng_eval = jax.random.split(rng)
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, config["eval_num_attempts"]), train_state)
        update_count = train_state.num_dr_updates + train_state.num_replay_updates + train_state.num_mutation_updates
        # The independent key ensures validation does not consume the training
        # RNG stream between evaluation blocks.
        validation_eval_key = jax.random.fold_in(
            jax.random.PRNGKey(config["validation_eval_seed"]), update_count
        )
        validation_returns = jax.vmap(eval_validation, (0, None))(
            jax.random.split(validation_eval_key, config["validation_num_attempts"]),
            train_state,
        )
        
        # Collect Metrics
        eval_solve_rates = jnp.where(cum_rewards > 0, 1., 0.).mean(axis=0) # (num_eval_levels,)
        eval_returns = cum_rewards.mean(axis=0) # (num_eval_levels,)
        
        # just grab the first run
        states, episode_lengths = jax.tree_util.tree_map(lambda x: x[0], (states, episode_lengths)) # (num_steps, num_eval_levels, ...), (num_eval_levels,)
        metrics["update_count"] = update_count
        metrics["eval_returns"] = eval_returns
        metrics["eval_solve_rates"] = eval_solve_rates
        metrics["eval_ep_lengths"]  = episode_lengths
        metrics["validation_solve_rates"] = (validation_returns > 0).mean(axis=0)
        metrics["validation_returns"] = validation_returns.mean(axis=0)
        if not config["lightweight_logging"]:
            images = jax.vmap(jax.vmap(env_renderer.render_state, (0, None)), (0, None))(states, env_params) # (num_steps, num_eval_levels, ...)
            frames = images.transpose(0, 1, 4, 2, 3) # WandB expects color channel before image dimensions when dealing with animations for some reason
            metrics["eval_animation"] = (frames, episode_lengths)
            metrics["dr_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.dr_last_level_batch, env_params)
            metrics["replay_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.replay_last_level_batch, env_params)
            metrics["mutation_levels"] = jax.vmap(env_renderer.render_level, (0, None))(train_state.mutation_last_level_batch, env_params)

            highest_scoring_level = level_sampler.get_levels(train_state.sampler, train_state.sampler["scores"].argmax())
            highest_weighted_level = level_sampler.get_levels(train_state.sampler, level_sampler.level_weights(train_state.sampler).argmax())
            metrics["highest_scoring_level"] = env_renderer.render_level(highest_scoring_level, env_params)
            metrics["highest_weighted_level"] = env_renderer.render_level(highest_weighted_level, env_params)
        
        return (rng, train_state), metrics
    
    def eval_checkpoint(og_config):
        """
            This function is what is used to evaluate a saved checkpoint *after* training. It first loads the checkpoint and then runs evaluation.
            It saves the states, cum_rewards and episode_lengths to a .npz file in the `results/run_name/seed` directory.
        """
        rng_init, rng_eval = jax.random.split(jax.random.PRNGKey(10000))
        def load(rng_init, checkpoint_directory: str):
            with open(os.path.join(checkpoint_directory, 'config.json')) as f: config = json.load(f)
            checkpoint_manager = ocp.CheckpointManager(os.path.join(os.getcwd(), checkpoint_directory, 'models'), item_handlers=ocp.StandardCheckpointHandler())

            train_state_og: TrainState = create_train_state(rng_init)
            step = checkpoint_manager.latest_step() if og_config['checkpoint_to_eval'] == -1 else og_config['checkpoint_to_eval']

            loaded_checkpoint = checkpoint_manager.restore(step)
            params = loaded_checkpoint['params']
            train_state = train_state_og.replace(params=params)
            return train_state, config, step
        
        train_state, config, step = load(rng_init, og_config['checkpoint_directory'])
        states, cum_rewards, episode_lengths = jax.vmap(eval, (0, None))(jax.random.split(rng_eval, og_config["eval_num_attempts"]), train_state)
        save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')
        os.makedirs(save_loc, exist_ok=True)
        np.savez_compressed(os.path.join(save_loc, 'results.npz'), states=np.asarray(states), cum_rewards=np.asarray(cum_rewards), episode_lengths=np.asarray(episode_lengths), levels=config['eval_levels'])
        solve_rates = np.asarray(cum_rewards > 0).mean(axis=0)
        mean_returns = np.asarray(cum_rewards).mean(axis=0)
        evaluation = {
            "checkpoint_step": int(step),
            "solve_rate_mean": float(solve_rates.mean()),
            "solve_rate_per_level": {
                name: float(value) for name, value in zip(config['eval_levels'], solve_rates)
            },
            "return_per_level": {
                name: float(value) for name, value in zip(config['eval_levels'], mean_returns)
            },
        }
        with open(os.path.join(save_loc, 'evaluation.json'), 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(json.dumps(evaluation, indent=2))
        return states, cum_rewards, episode_lengths

    if config['mode'] == 'eval':
        return eval_checkpoint(config) # evaluate and exit early

    # Set up the train states
    rng = jax.random.PRNGKey(config["seed"])
    rng_init, rng_train = jax.random.split(rng)
    
    train_state = create_train_state(rng_init)
    runner_state = (rng_train, train_state)
    
    # And run the train_eval_sep function for the specified number of updates
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager = setup_checkpointing(config, train_state, env, env_params)
    for eval_step in range(config["num_updates"] // config["eval_freq"]):
        start_time = time.time()
        runner_state, metrics = train_and_eval_step(runner_state, None)
        curr_time = time.time()
        metrics['time_delta'] = curr_time - start_time
        log_eval(metrics, train_state_to_log_dict(runner_state[1], level_sampler))
        update_count = int(np.asarray(metrics["update_count"]))
        if config["diagnostic_logging"] and (
            update_count == config["num_updates"]
            or update_count % config["diagnostic_snapshot_interval"] == 0
        ):
            save_sampler_snapshot(runner_state[1], update_count)
        if config["checkpoint_save_interval"] > 0:
            checkpoint_manager.save(eval_step, args=ocp.args.StandardSave(runner_state[1]))
            checkpoint_manager.wait_until_finished()
    if config["checkpoint_save_interval"] > 0:
        checkpoint_manager.save(
            config["num_updates"],
            args=ocp.args.StandardSave(runner_state[1]),
            force=True,
        )
        checkpoint_manager.wait_until_finished()
    return runner_state[1]

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="JAXUED_TEST")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    # === Train vs Eval ===
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--checkpoint_directory", type=str, default=None)
    parser.add_argument("--checkpoint_to_eval", type=int, default=-1)
    # === CHECKPOINTING ===
    parser.add_argument("--checkpoint_save_interval", type=int, default=2)
    parser.add_argument("--max_number_of_checkpoints", type=int, default=60)
    # === EVAL ===
    parser.add_argument("--eval_freq", type=int, default=250)
    parser.add_argument("--eval_num_attempts", type=int, default=10)
    parser.add_argument("--lightweight_logging", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local_results_dir", type=str, default="results")
    parser.add_argument("--diagnostic_logging", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--causal_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--diagnostic_snapshot_interval", type=int, default=1000)
    parser.add_argument("--validation_num_levels", type=int, default=64)
    parser.add_argument("--validation_num_attempts", type=int, default=5)
    parser.add_argument("--validation_level_seed", type=int, default=20260823)
    parser.add_argument("--validation_eval_seed", type=int, default=20260824)
    parser.add_argument("--validation_wall_counts", nargs='+', type=int, default=[15, 25, 35, 45])
    parser.add_argument("--eval_levels", nargs='+', default=[
        "SixteenRooms",
        "SixteenRooms2",
        "Labyrinth",
        "LabyrinthFlipped",
        "Labyrinth2",
        "StandardMaze",
        "StandardMaze2",
        "StandardMaze3",
    ])
    group = parser.add_argument_group('Training params')
    # === PPO === 
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--max_grad_norm", type=float, default=0.5)
    mut_group = group.add_mutually_exclusive_group()
    mut_group.add_argument("--num_updates", type=int, default=30000)
    mut_group.add_argument("--num_env_steps", type=int, default=None)
    group.add_argument("--num_steps", type=int, default=256)
    group.add_argument("--num_train_envs", type=int, default=32)
    group.add_argument("--num_minibatches", type=int, default=1)
    group.add_argument("--gamma", type=float, default=0.995)
    group.add_argument("--epoch_ppo", type=int, default=5)
    group.add_argument("--clip_eps", type=float, default=0.2)
    group.add_argument("--gae_lambda", type=float, default=0.98)
    group.add_argument("--entropy_coeff", type=float, default=1e-3)
    group.add_argument("--critic_coeff", type=float, default=0.5)
    # === PLR ===
    group.add_argument("--score_function", type=str, default="MaxMC", choices=["MaxMC", "pvl", "mna", "frontier", "frontier_lp", "mna_frontier", "mna_frontier_lp"])
    group.add_argument("--frontier_prior", type=float, default=0.5)
    group.add_argument("--frontier_ema_alpha", type=float, default=0.3)
    group.add_argument("--learning_progress_coeff", type=float, default=1.0)
    group.add_argument("--exploratory_grad_updates", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--level_buffer_capacity", type=int, default=4000)
    group.add_argument("--replay_prob", type=float, default=0.8)
    group.add_argument("--staleness_coeff", type=float, default=0.3)
    group.add_argument("--temperature", type=float, default=0.3)
    group.add_argument("--topk_k", type=int, default=4)
    group.add_argument("--minimum_fill_ratio", type=float, default=0.5)
    group.add_argument("--prioritization", type=str, default="rank", choices=["rank", "topk"])
    group.add_argument("--buffer_duplicate_check", action=argparse.BooleanOptionalAction, default=True)
    # === ACCEL ===
    group.add_argument("--use_accel", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--num_edits", type=int, default=5)
    group.add_argument("--adaptive_mutation", action=argparse.BooleanOptionalAction, default=False)
    # === ENV CONFIG ===
    group.add_argument("--agent_view_size", type=int, default=5)
    # === DR CONFIG ===
    group.add_argument("--n_walls", type=int, default=25)
    
    config = vars(parser.parse_args())
    if config["num_env_steps"] is not None:
        config["num_updates"] = config["num_env_steps"] // (config["num_train_envs"] * config["num_steps"])
    config["group_name"] = ''.join([str(config[key]) for key in sorted([a.dest for a in parser._action_groups[2]._group_actions])])
    
    if config['mode'] == 'eval':
        os.environ['WANDB_MODE'] = 'disabled'
    
    # wandb.login()
    main(config, project=config["project"])
