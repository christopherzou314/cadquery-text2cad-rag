"""Evaluate existing CadQuery experiment artifacts without calling the LLM API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text2cad.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    evaluate_cad_run,
    flatten_evaluation,
    legacy_dimension_constraint,
)


EXPERIMENT_DIR = ROOT / "experiments" / "cadquery_rag"
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
PROMPT_PATH = EXPERIMENT_DIR / "prompt_set.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    records_path = results_dir / "records.json"
    prompts = {
        prompt["id"]: prompt
        for prompt in json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    }
    records = json.loads(records_path.read_text(encoding="utf-8"))
    metadata = _load_json(results_dir / "experiment_metadata.json")
    repair_budget = int(metadata.get("max_repairs", 0))

    rows = []
    for key, record in records.items():
        prompt = prompts[record["prompt_id"]]
        constraints = prompt.get("constraints") or [
            legacy_dimension_constraint(prompt["target_dimensions_mm"])
        ]
        evaluation = evaluate_cad_run(
            attempt_returncodes=_attempt_returncodes(record),
            step_path=Path(record["step_path"]),
            stl_path=Path(record["stl_path"]),
            constraints=constraints,
            repair_budget=repair_budget,
        )
        run_dir = Path(record["run_dir"])
        evaluation_path = run_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record["constraints"] = constraints
        record["evaluation_path"] = str(evaluation_path)
        record["evaluation"] = evaluation
        record.update(flatten_evaluation(evaluation))
        agent_summary = _load_json(Path(record["run_dir"]) / "agent_run.json")
        if agent_summary.get("schema_version") == "2.0":
            record["telemetry"] = agent_summary
            record.update(_telemetry_fields(agent_summary))
        rows.append(_csv_row(key, record))

    records_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(results_dir / "evaluation_results.csv", rows)
    summary = _aggregate(rows, repair_budget)
    (results_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Evaluated {len(rows)} existing runs.")
    print(f"Detailed CSV: {results_dir / 'evaluation_results.csv'}")
    print(f"Summary JSON: {results_dir / 'evaluation_summary.json'}")


def _attempt_returncodes(record: dict) -> list[int]:
    # Several original runs hit an Assembly export harness bug. Their unchanged
    # generated code was re-executed successfully after the harness was fixed.
    if record.get("harness_rechecked") and record.get("returncode") is not None:
        return [int(record["returncode"])]
    agent_summary = Path(record["run_dir"]) / "agent_run.json"
    if agent_summary.exists():
        attempts = _load_json(agent_summary).get("attempts", [])
        return [int(attempt["returncode"]) for attempt in attempts]
    if record.get("returncode") is not None:
        return [int(record["returncode"])]
    return []


def _csv_row(key: str, record: dict) -> dict:
    return {
        "run_key": key,
        "prompt_id": record["prompt_id"],
        "difficulty": record["difficulty"],
        "category": record["category"],
        "condition": record["condition"],
        "seed_requested": record.get("seed_requested"),
        "seed_sent_to_provider": record.get("seed_sent_to_provider"),
        "repair_budget": record["evaluation"]["execution"]["repair_budget"],
        "first_generation_success": record["first_generation_success"],
        "success_within_repair_budget": record["success_within_repair_budget"],
        "repairs_used": record["repairs_used"],
        "step_export_success": record["step_export_success"],
        "stl_export_success": record["stl_export_success"],
        "geometry_valid": record["geometry_valid"],
        "step_geometry_valid": record["step_geometry_valid"],
        "step_solid_count": record["step_solid_count"],
        "cylindrical_hole_count": record["cylindrical_hole_count"],
        "dimension_constraint_satisfied": record["dimension_constraint_satisfied"],
        "constraint_groups_registered": record["constraint_groups_registered"],
        "constraint_groups_evaluated": record["constraint_groups_evaluated"],
        "constraint_evaluation_coverage": record["constraint_evaluation_coverage"],
        "constraint_groups_passed": record["constraint_groups_passed"],
        "constraint_groups_total": record["constraint_groups_total"],
        "constraint_group_pass_rate": record["constraint_group_pass_rate"],
        "all_evaluated_hard_constraints_satisfied": record[
            "all_evaluated_hard_constraints_satisfied"
        ],
        "all_registered_hard_constraints_satisfied": record[
            "all_registered_hard_constraints_satisfied"
        ],
        "task_success_v1": record["task_success_v1"],
        "task_success_v2": record["task_success_v2"],
        "actual_dimensions_mm": json.dumps(record["actual_dimensions_mm"]),
        "failed_constraint_ids": json.dumps(record["failed_constraint_ids"]),
        "unevaluated_constraint_ids": json.dumps(
            record["unevaluated_constraint_ids"]
        ),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "llm_call_count": record.get("llm_call_count"),
        "llm_transport_retries": record.get("llm_transport_retries"),
        "prompt_tokens": record.get("prompt_tokens"),
        "completion_tokens": record.get("completion_tokens"),
        "total_tokens": record.get("total_tokens"),
        "cached_prompt_tokens": record.get("cached_prompt_tokens"),
        "reasoning_tokens": record.get("reasoning_tokens"),
        "text_completion_tokens": record.get("text_completion_tokens"),
        "token_usage_complete": record.get("token_usage_complete"),
        "estimated_cost": record.get("estimated_cost"),
        "cost_currency": record.get("cost_currency"),
        "rag_retrieval_seconds": record.get("rag_retrieval_seconds"),
        "initial_generation_seconds": record.get("initial_generation_seconds"),
        "repair_generation_seconds": record.get("repair_generation_seconds"),
        "llm_seconds": record.get("llm_seconds"),
        "cad_execution_seconds": record.get("cad_execution_seconds"),
        "end_to_end_seconds": record.get("end_to_end_seconds"),
    }


def _aggregate(rows: list[dict], repair_budget: int) -> dict:
    conditions = sorted({row["condition"] for row in rows})
    by_condition = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        constraints_passed = sum(row["constraint_groups_passed"] for row in selected)
        constraints_total = sum(row["constraint_groups_total"] for row in selected)
        by_condition[condition] = {
            "run_count": len(selected),
            "first_generation_success_count": sum(
                row["first_generation_success"] for row in selected
            ),
            "success_within_repair_budget_count": sum(
                row["success_within_repair_budget"] for row in selected
            ),
            "step_export_success_count": sum(row["step_export_success"] for row in selected),
            "stl_export_success_count": sum(row["stl_export_success"] for row in selected),
            "geometry_valid_count": sum(row["geometry_valid"] for row in selected),
            "task_success_v1_count": sum(row["task_success_v1"] for row in selected),
            "task_success_v2_count": sum(row["task_success_v2"] for row in selected),
            "constraint_groups_passed": constraints_passed,
            "constraint_groups_total": constraints_total,
            "constraint_group_pass_rate": (
                round(constraints_passed / constraints_total, 6)
                if constraints_total
                else None
            ),
            "mean_repairs_used": round(
                sum(row["repairs_used"] for row in selected) / len(selected), 6
            ),
            "mean_elapsed_seconds": round(
                sum(float(row["elapsed_seconds"]) for row in selected) / len(selected),
                6,
            ),
            "runs_with_complete_token_usage": sum(
                row.get("token_usage_complete") is True for row in selected
            ),
            "total_prompt_tokens": _sum_present(selected, "prompt_tokens"),
            "total_completion_tokens": _sum_present(selected, "completion_tokens"),
            "total_tokens": _sum_present(selected, "total_tokens"),
            "estimated_total_cost": _sum_present(selected, "estimated_cost"),
            "mean_llm_seconds": _mean_present(selected, "llm_seconds"),
            "mean_cad_execution_seconds": _mean_present(
                selected, "cad_execution_seconds"
            ),
            "mean_end_to_end_seconds": _mean_present(selected, "end_to_end_seconds"),
        }
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repair_budget": repair_budget,
        "run_count": len(rows),
        "metric_scope": (
            "Execution, STEP/STL export, non-degenerate STL geometry, STEP solid "
            "count, and registered dimension, position, and cylindrical-hole constraints."
        ),
        "by_condition": by_condition,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sum_present(rows: list[dict], field: str) -> float | int | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return sum(values) if values else None


def _mean_present(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _telemetry_fields(telemetry: dict) -> dict:
    usage = telemetry["token_usage"]
    cost = telemetry["cost"]
    timing = telemetry["timing"]
    return {
        "llm_call_count": usage["call_count"],
        "llm_transport_retries": sum(
            call["transport_retry_count"] for call in telemetry["llm_calls"]
        ),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "cached_prompt_tokens": usage["cached_prompt_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "text_completion_tokens": usage["text_completion_tokens"],
        "token_usage_complete": usage["available_for_all_calls"],
        "estimated_cost": cost["estimated_total"],
        "cost_currency": cost["currency"],
        "rag_retrieval_seconds": timing["rag_retrieval_seconds"],
        "initial_generation_seconds": timing["initial_generation_seconds"],
        "repair_generation_seconds": timing["repair_generation_seconds"],
        "llm_seconds": timing["llm_seconds"],
        "cad_execution_seconds": timing["cad_execution_seconds"],
        "end_to_end_seconds": timing["end_to_end_seconds"],
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
