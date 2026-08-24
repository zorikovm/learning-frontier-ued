#!/usr/bin/env python3
"""Aggregate the latest JSONL record from every local experiment."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


EVAL_LEVELS = [
    "SixteenRooms",
    "SixteenRooms2",
    "Labyrinth",
    "LabyrinthFlipped",
    "Labyrinth2",
    "StandardMaze",
    "StandardMaze2",
    "StandardMaze3",
]

METHOD_ALIASES = {
    "diag_plr_maxmc": "baseline_plr_maxmc",
    "plr_mna": "mna_plr",
    "plr_mna_frontier": "mna_frontier_plr",
    "plr_mna_frontier_lp": "mna_frontier_lp_plr",
    "plr_mna_frontier_lp025": "mna_frontier_lp025_plr",
}


def load_latest(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else None


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def canonical_method(run_name):
    method = re.sub(r"_u\d+_s\d+$", "", run_name)
    return METHOD_ALIASES.get(method, method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.results_dir.glob("*/[0-9]*/metrics.jsonl")):
        metric = load_latest(metrics_path)
        if metric is None:
            continue
        run_name = metrics_path.parents[1].name
        if run_name.startswith("smoke_"):
            continue
        method = canonical_method(run_name)
        seed = int(metrics_path.parent.name)
        row = {
            "method": method,
            "run_name": run_name,
            "seed": seed,
            "num_updates": metric.get("num_updates"),
            "num_env_steps": metric.get("num_env_steps"),
            "wall_clock_seconds": metric.get("wall_clock_block_seconds"),
            "solve_rate": metric.get("solve_rate/mean"),
            "validation_solve_rate": metric.get("validation/solve_rate_mean"),
        }
        row.update({level: metric.get(f"solve_rate/{level}") for level in EVAL_LEVELS})
        rows.append(row)

    output_csv = args.results_dir / "summary.csv"
    fieldnames = list(rows[0]) if rows else ["method", "seed"]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    unique = {}
    for row in rows:
        key = (row["method"], row["num_updates"], row["seed"])
        previous = unique.get(key)
        # A diagnostic rerun can share method/budget/seed with an earlier
        # smoke baseline. Prefer the row with fixed-validation evidence.
        if previous is None or (
            previous["validation_solve_rate"] is None
            and row["validation_solve_rate"] is not None
        ):
            unique[key] = row

    groups = {}
    for row in unique.values():
        groups.setdefault((row["method"], row["num_updates"]), []).append(row)

    markdown = [
        "# Результаты",
        "",
        "Короткие запуски использовались для проверки гипотез. Полных результатов на 30000 updates пока нет.",
        "",
        "| Метод | Seeds | Updates | Public, среднее ± std | Проверочная выборка, среднее ± std |",
        "|---|---:|---:|---:|---:|",
    ]
    for (method, updates), method_rows in sorted(groups.items()):
        solve = [r["solve_rate"] for r in method_rows if r["solve_rate"] is not None]
        validation = [r["validation_solve_rate"] for r in method_rows if r["validation_solve_rate"] is not None]
        solve_mean, solve_std = mean_std(solve)
        val_mean, val_std = mean_std(validation)
        val_text = "N/A" if not validation else f"{val_mean:.4f} ± {val_std:.4f}"
        markdown.append(
            f"| {method} | {len(method_rows)} | {updates} | "
            f"{solve_mean:.4f} ± {solve_std:.4f} | {val_text} |"
        )
    markdown.extend([
        "",
        "## Public результаты по уровням",
        "",
        "| Метод | Updates | " + " | ".join(EVAL_LEVELS) + " |",
        "|---|---:|" + "---:|" * len(EVAL_LEVELS),
    ])
    for (method, updates), method_rows in sorted(groups.items()):
        values = []
        for level in EVAL_LEVELS:
            level_values = [r[level] for r in method_rows if r[level] is not None]
            mean, std = mean_std(level_values)
            values.append(f"{mean:.3f} ± {std:.3f}")
        markdown.append(f"| {method} | {updates} | " + " | ".join(values) + " |")
    output_md = args.results_dir / "SUMMARY.md"
    output_md.write_text("\n".join(markdown) + "\n")
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()
