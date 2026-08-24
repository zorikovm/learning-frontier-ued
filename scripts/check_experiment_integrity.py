#!/usr/bin/env python3
"""Fail if locked student code or PPO defaults differ from pinned upstream."""

import ast
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "0f8f1284677375b889e4f13a32c9617cd009f8c4"
EXAMPLE = "examples/maze_plr.py"
LOCKED_NODES = {"ActorCritic", "compute_gae", "update_actor_critic_rnn"}
LOCKED_DEFAULTS = {
    "--lr": 1e-4,
    "--max_grad_norm": 0.5,
    "--num_updates": 30000,
    "--num_steps": 256,
    "--num_train_envs": 32,
    "--num_minibatches": 1,
    "--gamma": 0.995,
    "--epoch_ppo": 5,
    "--clip_eps": 0.2,
    "--gae_lambda": 0.98,
    "--entropy_coeff": 1e-3,
    "--critic_coeff": 0.5,
}


def pinned_source():
    return subprocess.check_output(
        ["git", "show", f"{PINNED_COMMIT}:{EXAMPLE}"], cwd=ROOT, text=True
    )


def node_map(source):
    tree = ast.parse(source)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in LOCKED_NODES
    }


def parser_defaults(source):
    defaults = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        for keyword in node.keywords:
            if keyword.arg == "default":
                try:
                    defaults[flag] = ast.literal_eval(keyword.value)
                except (ValueError, TypeError):
                    pass
    return defaults


def main():
    current = (ROOT / EXAMPLE).read_text()
    pinned = pinned_source()
    if node_map(current) != node_map(pinned):
        raise SystemExit("Locked ActorCritic/PPO function AST differs from pinned upstream")

    current_defaults = parser_defaults(current)
    for flag, expected in LOCKED_DEFAULTS.items():
        if current_defaults.get(flag) != expected:
            raise SystemExit(
                f"Locked default changed: {flag}={current_defaults.get(flag)!r}, expected {expected!r}"
            )
    print(f"OK: locked student AST and PPO defaults match {PINNED_COMMIT}")


if __name__ == "__main__":
    main()
