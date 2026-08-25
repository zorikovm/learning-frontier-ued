#!/usr/bin/env python3
"""Collect training and standalone checkpoint evaluation results."""

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


def load_rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def canonical_method(run_name):
    method = re.sub(r"_u\d+_s\d+$", "", run_name)
    return METHOD_ALIASES.get(method, method)


def fmt(values, digits=4):
    if not values:
        return "N/A"
    mean, std = mean_std(values)
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.results_dir.glob("*/[0-9]*/metrics.jsonl")):
        metric_rows = load_rows(metrics_path)
        if not metric_rows:
            continue
        metric = metric_rows[-1]
        run_name = metrics_path.parents[1].name
        if run_name.startswith("smoke_"):
            continue
        evaluation_path = metrics_path.with_name("evaluation.json")
        evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
        method = canonical_method(run_name)
        seed = int(metrics_path.parent.name)
        row = {
            "method": method,
            "run_name": run_name,
            "seed": seed,
            "num_updates": metric.get("num_updates"),
            "num_env_steps": metric.get("num_env_steps"),
            "wall_clock_seconds": sum(
                row.get("wall_clock_block_seconds", 0.0) for row in metric_rows
            ),
            "solve_rate": metric.get("solve_rate/mean"),
            "checkpoint_solve_rate": evaluation.get("solve_rate_mean"),
            "validation_solve_rate": metric.get("validation/solve_rate_mean"),
        }
        row.update({level: metric.get(f"solve_rate/{level}") for level in EVAL_LEVELS})
        checkpoint_levels = evaluation.get("solve_rate_per_level", {})
        row.update({f"checkpoint_{level}": checkpoint_levels.get(level) for level in EVAL_LEVELS})
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
        if previous is None or (
            previous["validation_solve_rate"] is None
            and row["validation_solve_rate"] is not None
        ):
            unique[key] = row

    groups = {}
    for row in unique.values():
        groups.setdefault((row["method"], row["num_updates"]), []).append(row)

    has_full = any(updates == 30000 for _, updates in groups)
    intro = (
        "Полные результаты приведены вместе с короткими запусками, "
        "на которых отсеивались слабые идеи."
        if has_full
        else "Доступны только короткие запуски для проверки гипотез."
    )
    markdown = [
        "# Результаты",
        "",
        intro,
        "",
        "| Метод | Запуски | Обновления | Во время обучения | Отдельная оценка контрольной точки | Проверочная выборка |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for (method, updates), method_rows in sorted(groups.items()):
        train = [r["solve_rate"] for r in method_rows if r["solve_rate"] is not None]
        checkpoint = [r["checkpoint_solve_rate"] for r in method_rows if r["checkpoint_solve_rate"] is not None]
        validation = [r["validation_solve_rate"] for r in method_rows if r["validation_solve_rate"] is not None]
        markdown.append(
            f"| {method} | {len(method_rows)} | {updates} | {fmt(train)} | "
            f"{fmt(checkpoint)} | {fmt(validation)} |"
        )

    checkpoint_groups = [
        (key, value)
        for key, value in sorted(groups.items())
        if any(r["checkpoint_solve_rate"] is not None for r in value)
    ]
    if checkpoint_groups:
        markdown.extend([
            "",
            "## Отдельная оценка контрольной точки по уровням",
            "",
            "| Метод | Seeds | " + " | ".join(EVAL_LEVELS) + " |",
            "|---|---:|" + "---:|" * len(EVAL_LEVELS),
        ])
        for (method, _), method_rows in checkpoint_groups:
            values = []
            for level in EVAL_LEVELS:
                level_values = [
                    r[f"checkpoint_{level}"]
                    for r in method_rows
                    if r[f"checkpoint_{level}"] is not None
                ]
                values.append(fmt(level_values, digits=3))
            markdown.append(
                f"| {method} | {len(method_rows)} | " + " | ".join(values) + " |"
            )

    output_md = args.results_dir / "SUMMARY.md"
    output_md.write_text("\n".join(markdown) + "\n")
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()
