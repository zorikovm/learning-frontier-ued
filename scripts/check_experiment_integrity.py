#!/usr/bin/env python3
"""Fail if locked student code or PPO defaults differ from pinned upstream."""

import ast
import copy
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINNED_REPOSITORY = "DramaCow/jaxued"
PINNED_COMMIT = "0f8f1284677375b889e4f13a32c9617cd009f8c4"
EXAMPLE = "examples/maze_plr.py"

# SHA256 of a version-independent normalized AST representation.
# These hashes were taken from examples/maze_plr.py at PINNED_COMMIT.
LOCKED_NODE_HASHES = {
    "ActorCritic": "dca59ce11f0a44aa259a835a4016f7442d6d69c8677afc2a01f28ce2ddb528a6",
    "compute_gae": "c48fb770eb1a58d7e7d42e58868f040e4cca897bc3f515d9cf229d6330aceb9c",
    "update_actor_critic_rnn": "f8f92c0296e23bffdcbe440771b6b2923de00724247f8e2e8147e515605545f0",
}
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


class StripDocstrings(ast.NodeTransformer):
    """Remove docstrings so documentation edits do not fail the integrity check."""

    def _visit_body_node(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr):
            value = node.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = node.body[1:]
        return node

    def visit_FunctionDef(self, node):
        return self._visit_body_node(node)

    def visit_AsyncFunctionDef(self, node):
        return self._visit_body_node(node)

    def visit_ClassDef(self, node):
        return self._visit_body_node(node)


def canonical_ast(value):
    """Convert AST to a stable tuple, ignoring fields added by newer Python versions."""
    if isinstance(value, ast.AST):
        fields = []
        for name, child in ast.iter_fields(value):
            # Python 3.12 added type_params to FunctionDef/ClassDef. It is empty
            # for this Python 3.11 code, so ignoring it makes the representation
            # identical across supported Python versions.
            if name == "type_params":
                continue
            fields.append((name, canonical_ast(child)))
        return (value.__class__.__name__, tuple(fields))
    if isinstance(value, list):
        return tuple(canonical_ast(item) for item in value)
    return value


def node_hashes(source):
    tree = ast.parse(source)
    hashes = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if node.name not in LOCKED_NODE_HASHES:
            continue
        normalized = StripDocstrings().visit(copy.deepcopy(node))
        dumped = repr(canonical_ast(normalized))
        hashes[node.name] = hashlib.sha256(dumped.encode()).hexdigest()
    return hashes


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
    current_hashes = node_hashes(current)

    missing = LOCKED_NODE_HASHES.keys() - current_hashes.keys()
    if missing:
        raise SystemExit(f"Locked student nodes are missing: {', '.join(sorted(missing))}")

    for name, expected in LOCKED_NODE_HASHES.items():
        actual = current_hashes[name]
        if actual != expected:
            raise SystemExit(
                f"Locked student node changed: {name} (expected {expected}, got {actual})"
            )

    current_defaults = parser_defaults(current)
    for flag, expected in LOCKED_DEFAULTS.items():
        if current_defaults.get(flag) != expected:
            raise SystemExit(
                f"Locked default changed: {flag}={current_defaults.get(flag)!r}, expected {expected!r}"
            )

    print(
        "OK: locked student AST and PPO defaults match "
        f"{PINNED_REPOSITORY}@{PINNED_COMMIT}"
    )


if __name__ == "__main__":
    main()
