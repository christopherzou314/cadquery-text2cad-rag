"""Generate the reproducible final report for the 60-case benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = (
    ROOT / "experiments" / "cadquery_benchmark_v1" / "results" / "full_b2"
)
DEFAULT_CASES_PATH = (
    ROOT / "experiments" / "cadquery_benchmark_v1" / "full_cases.json"
)
CONDITIONS = ("baseline", "lightweight_rag", "full_rag")
BUDGETS = (0, 1, 2)
METRICS = ("execution", "constraint", "end_to_end")
CONDITION_LABELS = {
    "baseline": "Baseline",
    "lightweight_rag": "Lightweight RAG",
    "full_rag": "Full RAG",
}
CONDITION_COLORS = {
    "baseline": "#2563EB",
    "lightweight_rag": "#059669",
    "full_rag": "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    records = json.loads(
        (results_dir / "records.json").read_text(encoding="utf-8")
    )
    case_document = json.loads(args.cases.read_text(encoding="utf-8"))
    metadata = json.loads(
        (results_dir / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    cases = {case["id"]: case for case in case_document["cases"]}

    summary = build_summary(records, cases, metadata)
    _write_outputs(results_dir, summary)
    _write_manual_template(results_dir / "manual_scoring.csv", records, cases)
    _write_figures(results_dir, summary)
    (results_dir / "final_report.md").write_text(
        _english_report(summary), encoding="utf-8"
    )
    (results_dir / "final_report_zh.md").write_text(
        _chinese_report(summary), encoding="utf-8"
    )

    print(f"Final report generated from {len(records)} records.")
    print(f"English: {results_dir / 'final_report.md'}")
    print(f"Chinese: {results_dir / 'final_report_zh.md'}")


def build_summary(
    records: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rows = list(records.values())
    _validate_inputs(rows, cases, metadata)

    for row in rows:
        row["_category"] = cases[row["prompt_id"]]["category"]

    cad_rows = [row for row in rows if _is_cad(row)]
    clarification_rows = [row for row in rows if not _is_cad(row)]
    category_order = list(metadata["validation_summary"]["category_counts"])

    overall = _aggregate(rows)
    by_condition = {
        condition: _aggregate(
            [row for row in rows if row["condition"] == condition]
        )
        for condition in CONDITIONS
    }
    by_seed = {
        str(seed): {
            condition: _aggregate(
                [
                    row
                    for row in rows
                    if int(row["seed_requested"]) == seed
                    and row["condition"] == condition
                ]
            )
            for condition in CONDITIONS
        }
        for seed in sorted({int(row["seed_requested"]) for row in rows})
    }
    by_category = {
        category: {
            condition: _aggregate(
                [
                    row
                    for row in rows
                    if row["_category"] == category
                    and row["condition"] == condition
                ]
            )
            for condition in CONDITIONS
        }
        for category in category_order
    }

    pairwise = _pairwise_comparisons(cad_rows)
    failures = _failure_analysis(cad_rows)
    call_usage = _call_usage(rows)
    repair_analysis = _repair_analysis(cad_rows)

    manual_rows = sum(
        len(cases[row["prompt_id"]].get("manual_checks", [])) for row in rows
    )
    clarification_responses = sum(
        bool(row.get("clarification_response")) for row in clarification_rows
    )

    return {
        "benchmark_id": metadata["benchmark_id"],
        "benchmark_version": metadata["benchmark_version"],
        "evaluation_schema_version": metadata["evaluation_schema_version"],
        "model": metadata["model"],
        "temperature": metadata["temperature"],
        "seeds": metadata["seeds"],
        "repair_budget": metadata["max_repairs"],
        "conditions": list(CONDITIONS),
        "category_order": category_order,
        "counts": {
            "cases": len(cases),
            "categories": len(category_order),
            "runs": len(rows),
            "completed": sum(row.get("status") == "completed" for row in rows),
            "failed": sum(row.get("status") == "failed" for row in rows),
            "cad_runs": len(cad_rows),
            "clarification_runs": len(clarification_rows),
            "clarification_responses": clarification_responses,
            "clarification_evaluated": sum(
                row.get("clarification_success") is not None
                for row in clarification_rows
            ),
            "manual_scoring_rows": manual_rows,
        },
        "pricing": metadata["pricing"],
        "overall": overall,
        "by_condition": by_condition,
        "by_seed": by_seed,
        "by_category": by_category,
        "pairwise": pairwise,
        "failures": failures,
        "repair_analysis": repair_analysis,
        "call_usage": call_usage,
        "interpretation_boundary": [
            "Automatic results cover execution, exports, geometry validity, and registered hard constraints only.",
            "Manual semantic and visual checks are not included in success rates.",
            "Clarification responses are captured but remain unscored pending manual or VLM evaluation.",
            "Failed provider calls without usage metadata are absent from token and cost totals.",
        ],
    }


def _validate_inputs(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    expected = len(cases) * len(metadata["conditions"]) * len(metadata["seeds"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} records, found {len(rows)}")
    incomplete = [row for row in rows if row.get("status") != "completed"]
    if incomplete:
        raise ValueError(f"Cannot finalize report with {len(incomplete)} incomplete runs")
    missing_cases = {row["prompt_id"] for row in rows} - set(cases)
    if missing_cases:
        raise ValueError(f"Records reference unknown cases: {sorted(missing_cases)}")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cad_rows = [row for row in rows if _is_cad(row)]
    clarification_rows = [row for row in rows if not _is_cad(row)]
    curves: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        curves[metric] = {}
        for budget in BUDGETS:
            values = [_pass_at(row, metric, budget) for row in cad_rows]
            passed = sum(value is True for value in values)
            ci_low, ci_high = _wilson_interval(passed, len(values))
            curves[metric][str(budget)] = {
                "passed": passed,
                "total": len(values),
                "rate": _rate(passed, len(values)),
                "wilson_95_low": ci_low,
                "wilson_95_high": ci_high,
            }

    latencies = [
        float(row["end_to_end_seconds"])
        for row in rows
        if row.get("end_to_end_seconds") is not None
    ]
    cad_cost = sum(float(row.get("estimated_cost") or 0) for row in cad_rows)
    final_successes = curves["end_to_end"]["2"]["passed"]
    return {
        "run_count": len(rows),
        "cad_run_count": len(cad_rows),
        "clarification_run_count": len(clarification_rows),
        "clarification_response_count": sum(
            bool(row.get("clarification_response")) for row in clarification_rows
        ),
        "pass_curves": curves,
        "repairs_used": sum(int(row.get("repairs_used") or 0) for row in cad_rows),
        "runs_with_repairs": sum(
            int(row.get("repairs_used") or 0) > 0 for row in cad_rows
        ),
        "execution_repairs": sum(
            int(row.get("execution_repairs") or 0) for row in cad_rows
        ),
        "constraint_repairs": sum(
            int(row.get("constraint_repairs") or 0) for row in cad_rows
        ),
        "mean_repairs_per_cad_run": _mean(
            [float(row.get("repairs_used") or 0) for row in cad_rows]
        ),
        "tokens": {
            "prompt": sum(int(row.get("prompt_tokens") or 0) for row in rows),
            "completion": sum(
                int(row.get("completion_tokens") or 0) for row in rows
            ),
            "total": sum(int(row.get("total_tokens") or 0) for row in rows),
        },
        "cost": {
            "total": round(
                sum(float(row.get("estimated_cost") or 0) for row in rows), 8
            ),
            "cad_only": round(cad_cost, 8),
            "currency": _single_value(rows, "cost_currency"),
            "per_final_cad_success": round(cad_cost / final_successes, 8)
            if final_successes
            else None,
        },
        "latency_seconds": {
            "mean": _mean(latencies),
            "median": _median(latencies),
            "p95": _percentile(latencies, 0.95),
        },
        "transport_retries": sum(
            int(row.get("llm_transport_retries") or 0) for row in rows
        ),
        "runs_with_transport_retries": sum(
            int(row.get("llm_transport_retries") or 0) > 0 for row in rows
        ),
        "parameterized_code_count": sum(
            row.get("parameterized_code") is True for row in cad_rows
        ),
        "constraint_groups": {
            "registered": sum(
                int(row.get("constraint_groups_registered") or 0)
                for row in cad_rows
            ),
            "evaluated": sum(
                int(row.get("constraint_groups_evaluated") or 0)
                for row in cad_rows
            ),
            "passed": sum(
                int(row.get("constraint_groups_passed") or 0)
                for row in cad_rows
            ),
        },
    }


def _pairwise_comparisons(cad_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in cad_rows:
        key = (row["prompt_id"], int(row["seed_requested"]))
        indexed[row["condition"]][key] = row

    comparisons = []
    for comparison in ("lightweight_rag", "full_rag"):
        shared = sorted(set(indexed["baseline"]) & set(indexed[comparison]))
        for budget in BUDGETS:
            both_pass = baseline_only = comparison_only = both_fail = 0
            for key in shared:
                baseline_pass = _pass_at(
                    indexed["baseline"][key], "end_to_end", budget
                )
                comparison_pass = _pass_at(
                    indexed[comparison][key], "end_to_end", budget
                )
                if baseline_pass and comparison_pass:
                    both_pass += 1
                elif baseline_pass:
                    baseline_only += 1
                elif comparison_pass:
                    comparison_only += 1
                else:
                    both_fail += 1
            baseline_rate = (both_pass + baseline_only) / len(shared)
            comparison_rate = (both_pass + comparison_only) / len(shared)
            comparisons.append(
                {
                    "comparison": f"{comparison}_vs_baseline",
                    "budget": budget,
                    "paired_runs": len(shared),
                    "baseline_passed": both_pass + baseline_only,
                    "comparison_passed": both_pass + comparison_only,
                    "difference_percentage_points": round(
                        (comparison_rate - baseline_rate) * 100, 4
                    ),
                    "both_pass": both_pass,
                    "baseline_only": baseline_only,
                    "comparison_only": comparison_only,
                    "both_fail": both_fail,
                    "mcnemar_exact_p": _mcnemar_exact_p(
                        baseline_only, comparison_only
                    ),
                }
            )
    return comparisons


def _mcnemar_exact_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = min(left_only, right_only)
    probability = sum(
        math.comb(discordant, value) for value in range(tail + 1)
    ) / (2**discordant)
    return round(min(1.0, 2 * probability), 8)


def _failure_analysis(cad_rows: list[dict[str, Any]]) -> dict[str, Any]:
    initial_failures = [
        row for row in cad_rows if not _pass_at(row, "end_to_end", 0)
    ]
    final_failures = [
        row for row in cad_rows if not _pass_at(row, "end_to_end", 2)
    ]
    failure_stage = Counter()
    constraint_types = Counter()
    constraint_ids = Counter()
    failure_rows = []

    for row in final_failures:
        if not _pass_at(row, "execution", 2):
            stage = "execution_or_export"
            failure_stage[stage] += 1
            failed_ids: list[str] = []
            failed_types: list[str] = []
        else:
            evaluation = row.get("evaluation") or {}
            constraints = evaluation.get("constraints") or []
            failed = [
                item
                for item in constraints
                if item.get("hard") is not False
                and (item.get("evaluated") is not True or item.get("passed") is not True)
            ]
            if failed:
                stage = "hard_constraint"
                failure_stage[stage] += 1
                failed_ids = [str(item.get("id", "unknown")) for item in failed]
                failed_types = [str(item.get("type", "unknown")) for item in failed]
                constraint_ids.update(failed_ids)
                constraint_types.update(failed_types)
            else:
                stage = "geometry"
                failure_stage[stage] += 1
                failed_ids = []
                failed_types = []
        failure_rows.append(
            {
                "run_key": (
                    f"{row['prompt_id']}:{row['condition']}:"
                    f"seed_{row['seed_requested']}"
                ),
                "prompt_id": row["prompt_id"],
                "category": row["_category"],
                "condition": row["condition"],
                "seed": row["seed_requested"],
                "stage": stage,
                "failed_constraint_ids": failed_ids,
                "failed_constraint_types": failed_types,
                "error": _sanitize_local_paths(row.get("error") or ""),
            }
        )

    recovered = len(initial_failures) - len(final_failures)
    return {
        "initial_failure_count": len(initial_failures),
        "final_failure_count": len(final_failures),
        "recovered_count": recovered,
        "conditional_recovery_rate": _rate(recovered, len(initial_failures)),
        "final_failure_stage_counts": dict(failure_stage),
        "failed_constraint_type_counts": dict(constraint_types.most_common()),
        "failed_constraint_id_counts": dict(constraint_ids.most_common()),
        "final_failure_rows": failure_rows,
    }


def _repair_analysis(cad_rows: list[dict[str, Any]]) -> dict[str, Any]:
    repaired = [row for row in cad_rows if int(row.get("repairs_used") or 0) > 0]
    repair_counts = Counter(int(row.get("repairs_used") or 0) for row in cad_rows)
    initial_failures = [
        row for row in cad_rows if not _pass_at(row, "end_to_end", 0)
    ]
    recovered_at_1 = sum(
        _pass_at(row, "end_to_end", 1) for row in initial_failures
    )
    recovered_at_2 = sum(
        _pass_at(row, "end_to_end", 2) for row in initial_failures
    )
    return {
        "runs_with_repair": len(repaired),
        "repair_count_distribution": {
            str(key): repair_counts[key] for key in sorted(repair_counts)
        },
        "total_repairs": sum(int(row.get("repairs_used") or 0) for row in cad_rows),
        "execution_repairs": sum(
            int(row.get("execution_repairs") or 0) for row in cad_rows
        ),
        "constraint_repairs": sum(
            int(row.get("constraint_repairs") or 0) for row in cad_rows
        ),
        "mean_repairs_all_cad_runs": _mean(
            [float(row.get("repairs_used") or 0) for row in cad_rows]
        ),
        "mean_repairs_repaired_runs": _mean(
            [float(row.get("repairs_used") or 0) for row in repaired]
        ),
        "initial_failures": len(initial_failures),
        "recovered_by_b1": recovered_at_1,
        "recovered_by_b2": recovered_at_2,
        "additional_recovery_at_b2": recovered_at_2 - recovered_at_1,
    }


def _call_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "latency_seconds": 0.0}
    )
    for row in rows:
        for call in (row.get("telemetry") or {}).get("llm_calls", []):
            kind = str(call.get("call_kind") or "unknown")
            totals[kind]["calls"] += 1
            totals[kind]["tokens"] += int(
                (call.get("usage") or {}).get("total_tokens") or 0
            )
            totals[kind]["cost"] += float(call.get("estimated_cost") or 0)
            totals[kind]["latency_seconds"] += float(
                call.get("latency_seconds") or 0
            )
    return {
        kind: {
            "calls": int(values["calls"]),
            "tokens": int(values["tokens"]),
            "cost": round(float(values["cost"]), 8),
            "latency_seconds": round(float(values["latency_seconds"]), 6),
        }
        for kind, values in sorted(totals.items())
    }


def _pass_at(row: dict[str, Any], metric: str, budget: int) -> bool:
    key = f"{metric}_pass"
    for point in (row.get("telemetry") or {}).get("pass_by_repair_budget", []):
        if int(point["repair_budget"]) == budget:
            return point.get(key) is True
    raise ValueError(
        f"Missing {metric}@{budget} for {row.get('prompt_id')} "
        f"{row.get('condition')} seed {row.get('seed_requested')}"
    )


def _is_cad(row: dict[str, Any]) -> bool:
    return row.get("task_type") == "cad_generation"


def _write_outputs(results_dir: Path, summary: dict[str, Any]) -> None:
    (results_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    condition_rows = []
    for condition in CONDITIONS:
        item = summary["by_condition"][condition]
        condition_rows.append(_flat_summary_row(condition, item))
    _write_csv(results_dir / "final_condition_summary.csv", condition_rows)

    category_rows = []
    for category in summary["category_order"]:
        for condition in CONDITIONS:
            item = summary["by_category"][category][condition]
            category_rows.append(
                {"category": category, **_flat_summary_row(condition, item)}
            )
    _write_csv(results_dir / "final_category_summary.csv", category_rows)

    seed_rows = []
    for seed, condition_data in summary["by_seed"].items():
        for condition in CONDITIONS:
            seed_rows.append(
                {"seed": seed, **_flat_summary_row(condition, condition_data[condition])}
            )
    _write_csv(results_dir / "final_seed_summary.csv", seed_rows)
    _write_csv(results_dir / "final_pairwise_comparison.csv", summary["pairwise"])
    _write_csv(
        results_dir / "final_failure_runs.csv",
        [
            {
                **row,
                "failed_constraint_ids": ";".join(row["failed_constraint_ids"]),
                "failed_constraint_types": ";".join(
                    row["failed_constraint_types"]
                ),
            }
            for row in summary["failures"]["final_failure_rows"]
        ],
    )


def _flat_summary_row(condition: str, item: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "condition": condition,
        "run_count": item["run_count"],
        "cad_run_count": item["cad_run_count"],
        "clarification_run_count": item["clarification_run_count"],
    }
    for metric in METRICS:
        for budget in BUDGETS:
            point = item["pass_curves"][metric][str(budget)]
            row[f"{metric}_pass_at_{budget}"] = point["passed"]
            row[f"{metric}_rate_at_{budget}"] = point["rate"]
    row.update(
        {
            "repairs_used": item["repairs_used"],
            "execution_repairs": item["execution_repairs"],
            "constraint_repairs": item["constraint_repairs"],
            "total_tokens": item["tokens"]["total"],
            "estimated_cost": item["cost"]["total"],
            "median_latency_seconds": item["latency_seconds"]["median"],
            "p95_latency_seconds": item["latency_seconds"]["p95"],
            "transport_retries": item["transport_retries"],
        }
    )
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_manual_template(
    path: Path,
    records: dict[str, dict[str, Any]],
    cases: dict[str, dict[str, Any]],
) -> None:
    rows = []
    for run_key, record in records.items():
        for check in cases[record["prompt_id"]].get("manual_checks", []):
            rows.append(
                {
                    "run_key": run_key,
                    "prompt_id": record["prompt_id"],
                    "condition": record["condition"],
                    "seed": record["seed_requested"],
                    "manual_check_id": check["id"],
                    "criterion": check["criterion"],
                    "scale": check["scale"],
                    "required": check["required"],
                    "score": "",
                    "notes": "",
                }
            )
    _write_csv(path, rows)


def _write_figures(results_dir: Path, summary: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/text2cad-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for condition in CONDITIONS:
        values = [
            100
            * summary["by_condition"][condition]["pass_curves"]["end_to_end"][
                str(budget)
            ]["rate"]
            for budget in BUDGETS
        ]
        ax.plot(
            BUDGETS,
            values,
            marker="o",
            linewidth=2.2,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
    ax.set_title("End-to-End Success Across the Unified Repair Budget")
    ax.set_xlabel("Repair budget B")
    ax.set_ylabel("EndToEndPass (%)")
    ax.set_xticks(BUDGETS)
    ax.set_ylim(60, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "end_to_end_pass_curve.png", bbox_inches="tight")
    plt.close(fig)

    cad_categories = [
        category
        for category in summary["category_order"]
        if summary["by_category"][category]["baseline"]["cad_run_count"]
    ]
    x = np.arange(len(cad_categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for index, condition in enumerate(CONDITIONS):
        values = [
            100
            * summary["by_category"][category][condition]["pass_curves"][
                "end_to_end"
            ]["2"]["rate"]
            for category in cad_categories
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
    ax.set_title("EndToEndPass@2 by CAD Category")
    ax.set_ylabel("Success (%)")
    ax.set_xticks(x, [name.replace("_", "\n") for name in cad_categories])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "category_end_to_end_at_2.png", bbox_inches="tight")
    plt.close(fig)

    seeds = list(summary["by_seed"])
    x = np.arange(len(seeds))
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for index, condition in enumerate(CONDITIONS):
        values = [
            100
            * summary["by_seed"][seed][condition]["pass_curves"]["end_to_end"][
                "2"
            ]["rate"]
            for seed in seeds
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
        )
    ax.set_title("EndToEndPass@2 Across Seeds")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Success (%)")
    ax.set_xticks(x, seeds)
    ax.set_ylim(75, 100)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figures / "seed_stability_at_2.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    labels = [CONDITION_LABELS[condition] for condition in CONDITIONS]
    colors = [CONDITION_COLORS[condition] for condition in CONDITIONS]
    tokens = [summary["by_condition"][c]["tokens"]["total"] / 1e6 for c in CONDITIONS]
    costs = [summary["by_condition"][c]["cost"]["total"] for c in CONDITIONS]
    latency = [
        summary["by_condition"][c]["latency_seconds"]["median"]
        for c in CONDITIONS
    ]
    for ax, values, title, ylabel in zip(
        axes,
        (tokens, costs, latency),
        ("Total tokens", "Estimated cost", "Median latency"),
        ("Million tokens", "USD", "Seconds"),
    ):
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Resource Use by Generation Condition", y=1.02)
    fig.tight_layout()
    fig.savefig(figures / "resource_use_by_condition.png", bbox_inches="tight")
    plt.close(fig)


def _english_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    overall = summary["overall"]
    failures = summary["failures"]
    repair = summary["repair_analysis"]
    e0 = overall["pass_curves"]["end_to_end"]["0"]
    e2 = overall["pass_curves"]["end_to_end"]["2"]
    gain = 100 * (e2["rate"] - e0["rate"])

    lines = [
        "# CadQuery Benchmark v1: Final Experimental Report",
        "",
        "## Executive Summary",
        "",
        (
            f"This held-out evaluation contains {counts['cases']} prompts across "
            f"{counts['categories']} categories, three generation conditions, and "
            f"three seeds. All {counts['runs']} scheduled API tasks completed: "
            f"{counts['cad_runs']} CAD-generation runs and "
            f"{counts['clarification_runs']} clarification-only runs."
        ),
        "",
        (
            f"Across CAD runs, EndToEndPass increased from {e0['passed']}/"
            f"{e0['total']} ({_pct(e0['rate'])}) at B=0 to {e2['passed']}/"
            f"{e2['total']} ({_pct(e2['rate'])}) at B=2, a gain of "
            f"{gain:.2f} percentage points. The final ExecutionPass@2 was "
            f"{_pct(overall['pass_curves']['execution']['2']['rate'])}, so hard "
            "constraints rather than Python/CadQuery execution were the larger "
            "remaining source of failure."
        ),
        "",
        (
            "Baseline achieved the highest EndToEndPass@2 "
            f"({_pct(summary['by_condition']['baseline']['pass_curves']['end_to_end']['2']['rate'])}), "
            "followed by lightweight RAG "
            f"({_pct(summary['by_condition']['lightweight_rag']['pass_curves']['end_to_end']['2']['rate'])}) "
            "and full RAG "
            f"({_pct(summary['by_condition']['full_rag']['pass_curves']['end_to_end']['2']['rate'])}). "
            "Under this retrieval and prompting configuration, adding reference "
            "material did not improve the primary automatic metric."
        ),
        "",
        "![End-to-end pass curve](figures/end_to_end_pass_curve.png)",
        "",
        "## Experimental Design",
        "",
        f"- Model: `{summary['model']}` with temperature {summary['temperature']}.",
        f"- Test set: B013-B072, {counts['cases']} held-out cases, six per category.",
        f"- Seeds: {', '.join(str(seed) for seed in summary['seeds'])}.",
        "- Conditions: baseline, lightweight curated CadQuery references, and full-documentation RAG.",
        "- Unified repair budget: B=2. Execution and hard-constraint repairs consume the same budget; unchanged transport retries do not.",
        "- Automatic hard constraints: bounding-box dimensions/bounds, STEP solid count, and cylindrical hole patterns.",
        "- Conflicting-description cases request clarification and skip CAD execution and repair.",
        "",
        "## Metric Definitions",
        "",
        "- **ExecutionPass@B:** within budget B, code executes and required STEP/STL artifacts are exported.",
        "- **ConstraintPass@B:** within budget B, every registered automatic hard constraint is evaluated and passed.",
        "- **EndToEndPass@B:** execution, exports, valid non-degenerate geometry, and all registered hard constraints pass.",
        "- **B=0** is the initial generation; B=1 and B=2 permit one and two model revisions, respectively.",
        "",
        "## Main Results",
        "",
        _overall_curve_table(overall),
        "",
        _condition_table(summary),
        "",
        (
            f"The B=2 EndToEndPass estimate has a run-level Wilson 95% interval "
            f"of {_pct(e2['wilson_95_low'])} to {_pct(e2['wilson_95_high'])}. "
            "This interval is descriptive because runs from the same prompt are "
            "not fully independent."
        ),
        "",
        "## Paired Condition Comparisons",
        "",
        _pairwise_table(summary),
        "",
        "The exact McNemar tests pair conditions by prompt and seed. They test run-level discordance and do not establish generalization beyond this benchmark.",
        "",
        "## Repair Effectiveness",
        "",
        (
            f"There were {failures['initial_failure_count']} initial end-to-end "
            f"failures. The closed loop recovered {repair['recovered_by_b1']} by "
            f"B=1 and {repair['recovered_by_b2']} by B=2; the second repair added "
            f"{repair['additional_recovery_at_b2']} recoveries. Conditional "
            f"recovery at B=2 was {_pct(failures['conditional_recovery_rate'])}."
        ),
        "",
        (
            f"The system used {repair['total_repairs']} model revisions: "
            f"{repair['execution_repairs']} execution-triggered and "
            f"{repair['constraint_repairs']} constraint-triggered. Mean repair "
            f"count was {repair['mean_repairs_all_cad_runs']:.3f} per CAD run "
            f"and {repair['mean_repairs_repaired_runs']:.3f} among repaired runs."
        ),
        "",
        "## Category and Seed Results",
        "",
        "![Category results](figures/category_end_to_end_at_2.png)",
        "",
        _category_table(summary),
        "",
        "![Seed stability](figures/seed_stability_at_2.png)",
        "",
        _seed_table(summary),
        "",
        "## Remaining Automatic Failures",
        "",
        (
            f"At B=2, {failures['final_failure_count']} of {counts['cad_runs']} "
            "CAD runs remained unsuccessful. The final-stage breakdown was "
            f"{_format_counts(failures['final_failure_stage_counts'])}."
        ),
        "",
        _constraint_failure_table(summary),
        "",
        "The complete run-level list is stored in `final_failure_runs.csv`.",
        "",
        "## Resource Use and Reliability",
        "",
        "![Resource use](figures/resource_use_by_condition.png)",
        "",
        _resource_table(summary),
        "",
        _call_usage_table(summary),
        "",
        (
            f"Total recorded usage was {overall['tokens']['total']:,} tokens and "
            f"{overall['cost']['currency']} {overall['cost']['total']:.4f}. "
            f"There were {overall['transport_retries']} transport retries across "
            f"{overall['runs_with_transport_retries']} recorded tasks. Network "
            "retries did not consume the model repair budget."
        ),
        "",
        "## Clarification and Deferred Evaluation",
        "",
        (
            f"All {counts['clarification_responses']}/{counts['clarification_runs']} "
            "conflicting-description tasks produced a clarification response. "
            "Their correctness remains unscored. The generated "
            f"`manual_scoring.csv` contains {counts['manual_scoring_rows']} rows "
            "reserved for later human or VLM assessment. None of these pending "
            "scores are included in the automatic success rates."
        ),
        "",
        "## Limitations",
        "",
        "- Automatic evaluation does not yet measure recognizability, semantic similarity, slot dimensions, wall thickness, or general visual quality unless encoded by a registered v2 constraint.",
        "- Clarification quality is pending; capturing a response is not equivalent to answering correctly.",
        "- The study uses one model, one provider, three seeds, and one retrieval configuration.",
        "- Full RAG retrieves top-k excerpts; it does not place every documentation page into every request.",
        "- Run-level confidence intervals understate prompt-level dependence; paired tests partially address this but do not replace replication on new case sets.",
        "- Usage totals exclude failed provider requests that returned no usage metadata, including some job-level retries completed later by resume.",
        "",
        "## Conclusion",
        "",
        (
            "The principal positive result is the constraint-aware repair loop: "
            f"it raised EndToEndPass from {_pct(e0['rate'])} to {_pct(e2['rate'])}. "
            "The majority of revisions were triggered by measured geometric "
            "constraint failures, supporting the value of execution-plus-geometry "
            "feedback over traceback-only repair. In contrast, neither RAG mode "
            "outperformed baseline on the primary automatic metric in this study. "
            "The next evaluation layer should score semantic and visual quality "
            "with blinded human review or a validated VLM rubric while expanding "
            "the hard-constraint registry."
        ),
        "",
        "## Reproducibility",
        "",
        "The report was generated directly from `records.json` with:",
        "",
        "```bash",
        "python scripts/generate_final_benchmark_report.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def _chinese_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    overall = summary["overall"]
    failures = summary["failures"]
    repair = summary["repair_analysis"]
    e0 = overall["pass_curves"]["end_to_end"]["0"]
    e2 = overall["pass_curves"]["end_to_end"]["2"]
    gain = 100 * (e2["rate"] - e0["rate"])

    lines = [
        "# CadQuery Benchmark v1 最终实验报告",
        "",
        "## 摘要",
        "",
        (
            f"本次留出测试包含 {counts['cases']} 个案例，覆盖 "
            f"{counts['categories']} 个类别、3 种生成模式和 3 个 seed。"
            f"计划中的 {counts['runs']} 个 API 任务已全部完成，其中包括 "
            f"{counts['cad_runs']} 个 CAD 生成任务和 "
            f"{counts['clarification_runs']} 个冲突澄清任务。"
        ),
        "",
        (
            f"在 CAD 任务上，EndToEndPass 从 B=0 的 {e0['passed']}/"
            f"{e0['total']}（{_pct(e0['rate'])}）提升到 B=2 的 "
            f"{e2['passed']}/{e2['total']}（{_pct(e2['rate'])}），提高 "
            f"{gain:.2f} 个百分点。最终 ExecutionPass@2 为 "
            f"{_pct(overall['pass_curves']['execution']['2']['rate'])}，说明"
            "剩余失败主要来自硬约束，而不是 Python/CadQuery 执行。"
        ),
        "",
        (
            "Baseline 的 EndToEndPass@2 最高，为 "
            f"{_pct(summary['by_condition']['baseline']['pass_curves']['end_to_end']['2']['rate'])}；"
            "Lightweight RAG 为 "
            f"{_pct(summary['by_condition']['lightweight_rag']['pass_curves']['end_to_end']['2']['rate'])}；"
            "Full RAG 为 "
            f"{_pct(summary['by_condition']['full_rag']['pass_curves']['end_to_end']['2']['rate'])}。"
            "在当前检索和提示配置下，加入参考资料没有提高主要自动指标。"
        ),
        "",
        "![端到端成功率曲线](figures/end_to_end_pass_curve.png)",
        "",
        "## 实验设计",
        "",
        f"- 模型：`{summary['model']}`，temperature={summary['temperature']}。",
        f"- 测试集：B013-B072，共 {counts['cases']} 个留出案例，每类 6 个。",
        f"- Seed：{', '.join(str(seed) for seed in summary['seeds'])}。",
        "- 模式：Baseline、轻量级 CadQuery 参考库、完整官方文档 RAG。",
        "- 统一修复预算：B=2。执行错误修复和硬约束修复共享预算；未修改请求的网络重试不计入预算。",
        "- 自动硬约束：包围盒尺寸/边界、STEP 实体数量、圆柱孔阵列。",
        "- 冲突描述案例只请求澄清，不执行 CAD，也不进行代码修复。",
        "",
        "## 指标定义",
        "",
        "- **ExecutionPass@B**：预算 B 内代码成功执行，并导出要求的 STEP/STL。",
        "- **ConstraintPass@B**：预算 B 内全部已注册自动硬约束均完成评价并通过。",
        "- **EndToEndPass@B**：执行、导出、非退化有效几何和全部自动硬约束同时通过。",
        "- **B=0** 表示首次生成；B=1/B=2 分别允许模型修改代码 1/2 次。",
        "",
        "## 主要结果",
        "",
        _overall_curve_table(overall),
        "",
        _condition_table(summary),
        "",
        (
            f"B=2 EndToEndPass 的运行级 Wilson 95% 区间为 "
            f"{_pct(e2['wilson_95_low'])} 至 {_pct(e2['wilson_95_high'])}。"
            "由于同一 prompt 的多次运行并非完全独立，该区间只作为描述性结果。"
        ),
        "",
        "## 成对模式比较",
        "",
        _pairwise_table(summary),
        "",
        "McNemar 精确检验按相同 prompt 和 seed 配对，只反映本测试集中的运行级差异，不能单独证明对其他任务的泛化。",
        "",
        "## 修复闭环效果",
        "",
        (
            f"首次生成共有 {failures['initial_failure_count']} 个端到端失败。"
            f"B=1 时累计修复 {repair['recovered_by_b1']} 个，B=2 时累计修复 "
            f"{repair['recovered_by_b2']} 个；第二次修复额外恢复 "
            f"{repair['additional_recovery_at_b2']} 个。B=2 条件失败恢复率为 "
            f"{_pct(failures['conditional_recovery_rate'])}。"
        ),
        "",
        (
            f"系统共使用 {repair['total_repairs']} 次模型修改，其中执行错误触发 "
            f"{repair['execution_repairs']} 次，硬约束错误触发 "
            f"{repair['constraint_repairs']} 次。所有 CAD 任务平均修复 "
            f"{repair['mean_repairs_all_cad_runs']:.3f} 次；只统计发生过修复的任务时，"
            f"平均为 {repair['mean_repairs_repaired_runs']:.3f} 次。"
        ),
        "",
        "## 类别与 Seed 分析",
        "",
        "![类别结果](figures/category_end_to_end_at_2.png)",
        "",
        _category_table(summary),
        "",
        "![Seed 稳定性](figures/seed_stability_at_2.png)",
        "",
        _seed_table(summary),
        "",
        "## 剩余自动失败",
        "",
        (
            f"B=2 后仍有 {failures['final_failure_count']}/{counts['cad_runs']} "
            f"个 CAD 任务失败。最终失败阶段为："
            f"{_format_counts(failures['final_failure_stage_counts'])}。"
        ),
        "",
        _constraint_failure_table(summary),
        "",
        "完整的运行级失败列表保存在 `final_failure_runs.csv`。",
        "",
        "## 资源消耗与 API 可靠性",
        "",
        "![资源消耗](figures/resource_use_by_condition.png)",
        "",
        _resource_table(summary),
        "",
        _call_usage_table(summary),
        "",
        (
            f"记录到的总用量为 {overall['tokens']['total']:,} tokens，估算费用 "
            f"{overall['cost']['currency']} {overall['cost']['total']:.4f}。"
            f"共有 {overall['transport_retries']} 次网络重试，涉及 "
            f"{overall['runs_with_transport_retries']} 个任务；网络重试不消耗模型修复预算。"
        ),
        "",
        "## 澄清任务与待补评价",
        "",
        (
            f"{counts['clarification_responses']}/{counts['clarification_runs']} "
            "个冲突描述任务均生成了澄清回复，但回复是否正确仍未评分。"
            f"`manual_scoring.csv` 已保留 {counts['manual_scoring_rows']} 行人工或 VLM "
            "评价位置；这些待评结果没有被计入任何自动成功率。"
        ),
        "",
        "## 局限性",
        "",
        "- 除非已经注册为 v2 硬约束，当前自动评价不能测量可识别性、语义相似度、槽尺寸、薄壁厚度或整体视觉质量。",
        "- 捕获到澄清回复不等于回复正确，冲突处理能力仍需人工或 VLM 评价。",
        "- 实验只使用一个模型、一个 API 提供方、三个 seed 和一套检索配置。",
        "- Full RAG 每次检索 top-k 片段，并不是把全部官方文档放入每个请求。",
        "- 运行级置信区间会低估 prompt 内部相关性；成对检验有所缓解，但仍需新的独立案例集复现。",
        "- API 未返回 usage 的失败请求不会进入 token 和费用总计，包括部分后续通过 resume 补齐的任务级失败。",
        "",
        "## 结论",
        "",
        (
            "本实验最明确的正向结果是硬约束反馈修复闭环：EndToEndPass 从 "
            f"{_pct(e0['rate'])} 提升到 {_pct(e2['rate'])}。大多数代码修改由"
            "几何硬约束失败触发，说明“执行 + 几何测量反馈”相比只依赖 traceback "
            "具有实际价值。另一方面，两种 RAG 模式均未在主要自动指标上超过 Baseline。"
            "下一阶段应在扩展硬约束注册表的同时，引入盲评人工评价或经过验证的 VLM "
            "评分体系，补充语义和视觉质量证据。"
        ),
        "",
        "## 复现方式",
        "",
        "本报告直接由 `records.json` 生成：",
        "",
        "```bash",
        "python scripts/generate_final_benchmark_report.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def _overall_curve_table(item: dict[str, Any]) -> str:
    lines = [
        "| Metric | @0 | @1 | @2 |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "execution": "ExecutionPass",
        "constraint": "ConstraintPass",
        "end_to_end": "EndToEndPass",
    }
    for metric in METRICS:
        cells = []
        for budget in BUDGETS:
            point = item["pass_curves"][metric][str(budget)]
            cells.append(
                f"{point['passed']}/{point['total']} ({_pct(point['rate'])})"
            )
        lines.append(f"| {labels[metric]} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _condition_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Condition | CAD n | E2E@0 | E2E@1 | E2E@2 | Repairs | Tokens | Cost | Median latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["by_condition"][condition]
        curve = item["pass_curves"]["end_to_end"]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {item['cad_run_count']} | "
            f"{_pct(curve['0']['rate'])} | {_pct(curve['1']['rate'])} | "
            f"{_pct(curve['2']['rate'])} | {item['repairs_used']} | "
            f"{item['tokens']['total']:,} | {item['cost']['currency']} "
            f"{item['cost']['total']:.4f} | {item['latency_seconds']['median']:.2f}s |"
        )
    return "\n".join(lines)


def _pairwise_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Comparison | B | Baseline | Comparison | Difference | Baseline only | Comparison only | Exact p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["pairwise"]:
        label = CONDITION_LABELS[row["comparison"].removesuffix("_vs_baseline")]
        n = row["paired_runs"]
        lines.append(
            f"| {label} vs Baseline | {row['budget']} | "
            f"{row['baseline_passed']}/{n} | {row['comparison_passed']}/{n} | "
            f"{row['difference_percentage_points']:+.2f} pp | "
            f"{row['baseline_only']} | {row['comparison_only']} | "
            f"{row['mcnemar_exact_p']:.4f} |"
        )
    return "\n".join(lines)


def _category_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Category | Baseline E2E@2 | Lightweight RAG | Full RAG |",
        "|---|---:|---:|---:|",
    ]
    for category in summary["category_order"]:
        baseline = summary["by_category"][category]["baseline"]
        if not baseline["cad_run_count"]:
            lines.append(f"| {category} | clarification only | clarification only | clarification only |")
            continue
        cells = []
        for condition in CONDITIONS:
            item = summary["by_category"][category][condition]
            point = item["pass_curves"]["end_to_end"]["2"]
            cells.append(f"{point['passed']}/{point['total']} ({_pct(point['rate'])})")
        lines.append(f"| {category} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _seed_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Seed | Baseline E2E@2 | Lightweight RAG | Full RAG |",
        "|---:|---:|---:|---:|",
    ]
    for seed, data in summary["by_seed"].items():
        cells = []
        for condition in CONDITIONS:
            point = data[condition]["pass_curves"]["end_to_end"]["2"]
            cells.append(f"{point['passed']}/{point['total']} ({_pct(point['rate'])})")
        lines.append(f"| {seed} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _constraint_failure_table(summary: dict[str, Any]) -> str:
    counts = summary["failures"]["failed_constraint_type_counts"]
    lines = [
        "| Failed hard-constraint type at B=2 | Failed groups |",
        "|---|---:|",
    ]
    if not counts:
        lines.append("| None | 0 |")
    else:
        for constraint_type, count in counts.items():
            lines.append(f"| {constraint_type} | {count} |")
    return "\n".join(lines)


def _resource_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Condition | Total tokens | Cost | Mean latency | Median latency | P95 latency | Transport retries |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = summary["by_condition"][condition]
        lines.append(
            f"| {CONDITION_LABELS[condition]} | {item['tokens']['total']:,} | "
            f"{item['cost']['currency']} {item['cost']['total']:.4f} | "
            f"{item['latency_seconds']['mean']:.2f}s | "
            f"{item['latency_seconds']['median']:.2f}s | "
            f"{item['latency_seconds']['p95']:.2f}s | {item['transport_retries']} |"
        )
    return "\n".join(lines)


def _call_usage_table(summary: dict[str, Any]) -> str:
    lines = [
        "| LLM call type | Calls | Tokens | Cost | Total LLM latency |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind, item in summary["call_usage"].items():
        lines.append(
            f"| {kind} | {item['calls']} | {item['tokens']:,} | "
            f"USD {item['cost']:.4f} | {item['latency_seconds']:.2f}s |"
        )
    return "\n".join(lines)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return round(center - margin, 8), round(center + margin, 8)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return round(statistics.median(values), 6) if values else None


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 6)


def _single_value(rows: list[dict[str, Any]], field: str) -> Any:
    values = {row.get(field) for row in rows if row.get(field) is not None}
    return next(iter(values)) if len(values) == 1 else None


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _format_counts(values: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items()) or "none"


def _sanitize_local_paths(value: str) -> str:
    return value.replace(str(ROOT), "<PROJECT_ROOT>").replace(
        str(Path.home()), "<HOME>"
    )


if __name__ == "__main__":
    main()
