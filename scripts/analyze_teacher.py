#!/usr/bin/env python3
"""Analyze training-level teacher diagnostics without touching public dev levels."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


def correlation(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def shortest_path(walls, start, goal):
    start = tuple(int(v) for v in start)
    goal = tuple(int(v) for v in goal)
    queue = deque([(start, 0)])
    seen = {start}
    height, width = walls.shape
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == goal:
            return distance
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < width and 0 <= ny < height and not walls[ny, nx] and (nx, ny) not in seen:
                seen.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    return np.inf


def quantile_success(scores, success, observed, bins=5):
    mask = np.isfinite(scores) & observed
    order = np.flatnonzero(mask)[np.argsort(scores[mask])]
    chunks = np.array_split(order, bins)
    return [
        {
            "bin": idx,
            "count": int(chunk.size),
            "score_mean": float(scores[chunk].mean()) if chunk.size else None,
            "success_mean": float(success[chunk].mean()) if chunk.size else None,
        }
        for idx, chunk in enumerate(chunks)
    ]


def next_visit_analysis(diag):
    if "wall_map" not in diag:
        return None
    scores = diag["teacher_scores"]
    success = diag["success_rate"]
    completed = diag["completed_episodes"]
    wall_map = diag["wall_map"]
    goal_pos = diag["goal_pos"]
    agent_pos = diag["agent_pos"]
    agent_dir = diag["agent_dir"]
    update_kind = diag["update_kind"]

    previous = {}
    prior_scores = []
    next_success = []
    progress = []
    gaps = []
    prior_kinds = []
    for update in range(scores.shape[0]):
        current = {}
        for env_idx in range(scores.shape[1]):
            if completed[update, env_idx] <= 0:
                continue
            identity = b"".join((
                wall_map[update, env_idx].tobytes(),
                goal_pos[update, env_idx].tobytes(),
                agent_pos[update, env_idx].tobytes(),
                agent_dir[update, env_idx].tobytes(),
            ))
            current.setdefault(identity, []).append(env_idx)
        for identity, env_indices in current.items():
            current_success = float(success[update, env_indices].mean())
            current_score = float(scores[update, env_indices].mean())
            if identity in previous:
                previous_update, previous_score, previous_success, previous_kind = previous[identity]
                if np.isfinite(previous_score):
                    prior_scores.append(previous_score)
                    next_success.append(current_success)
                    progress.append(current_success - previous_success)
                    gaps.append(update - previous_update)
                    prior_kinds.append(previous_kind)
            previous[identity] = (
                update,
                current_score,
                current_success,
                int(update_kind[update]),
            )
    prior_scores = np.asarray(prior_scores)
    next_success = np.asarray(next_success)
    progress = np.asarray(progress)
    prior_kinds = np.asarray(prior_kinds)

    def summarize(mask):
        return {
            "count": int(mask.sum()),
            "score_next_success_correlation": correlation(prior_scores[mask], next_success[mask]),
            "score_subsequent_progress_correlation": correlation(prior_scores[mask], progress[mask]),
            "mean_subsequent_progress": float(progress[mask].mean()) if mask.any() else None,
        }

    report = {
        "num_next_visits": int(prior_scores.size),
        "score_next_success_correlation": correlation(prior_scores, next_success),
        "score_subsequent_progress_correlation": correlation(prior_scores, progress),
        "mean_subsequent_progress": float(progress.mean()) if progress.size else None,
        "mean_visit_gap_updates": float(np.mean(gaps)) if gaps else None,
    }
    report["by_prior_update_kind"] = {
        "new_no_ppo": summarize(prior_kinds == 0),
        "replay_with_ppo": summarize(prior_kinds == 1),
        "mutation_no_ppo": summarize(prior_kinds == 2),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--sampler", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    diag = np.load(args.diagnostics)
    scores = diag["teacher_scores"].reshape(-1)
    success = diag["success_rate"].reshape(-1)
    completed = diag["completed_episodes"].reshape(-1)
    maxmc = diag["maxmc"].reshape(-1)
    pvl = diag["pvl"].reshape(-1)
    observed = completed > 0

    report = {
        "num_rollout_levels": int(scores.size),
        "observed_fraction": float(observed.mean()),
        "mean_success_rate": float(success[observed].mean()) if observed.any() else None,
        "score_success_correlation": correlation(scores[observed], success[observed]),
        "maxmc_success_correlation": correlation(maxmc[observed], success[observed]),
        "pvl_success_correlation": correlation(pvl[observed], success[observed]),
        "score_quantiles": quantile_success(scores, success, observed),
        "next_visit": next_visit_analysis(diag),
    }

    if args.sampler:
        sampler = np.load(args.sampler)
        walls = sampler["wall_map"]
        agent = sampler["agent_pos"]
        goal = sampler["goal_pos"]
        sampler_scores = sampler["scores"]
        wall_count = walls.sum(axis=(1, 2)).astype(float)
        manhattan = np.abs(agent.astype(float) - goal.astype(float)).sum(axis=1)
        path_length = np.asarray([
            shortest_path(wall, start, target)
            for wall, start, target in zip(walls, agent, goal)
        ])
        finite_path = np.isfinite(path_length)
        top_count = max(1, int(np.ceil(0.1 * sampler_scores.size)))
        top = np.argsort(sampler_scores)[-top_count:]
        report["sampler"] = {
            "size": int(sampler_scores.size),
            "solvable_fraction": float(finite_path.mean()),
            "score_wall_count_correlation": correlation(sampler_scores, wall_count),
            "score_manhattan_correlation": correlation(sampler_scores, manhattan),
            "score_shortest_path_correlation": correlation(
                sampler_scores[finite_path], path_length[finite_path]
            ),
            "overall_mean_wall_count": float(wall_count.mean()),
            "top_decile_mean_wall_count": float(wall_count[top].mean()),
            "overall_mean_shortest_path": float(path_length[finite_path].mean()) if finite_path.any() else None,
            "top_decile_solvable_fraction": float(finite_path[top].mean()),
            "top_decile_mean_shortest_path": float(path_length[top][finite_path[top]].mean()) if finite_path[top].any() else None,
        }
        if "extra_success_ema" in sampler:
            report["sampler"]["score_success_ema_correlation"] = correlation(
                sampler_scores, sampler["extra_success_ema"]
            )
            report["sampler"]["top_decile_success_ema"] = float(
                sampler["extra_success_ema"][top].mean()
            )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
