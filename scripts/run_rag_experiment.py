"""Run the reproducible baseline versus lightweight-RAG CadQuery experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text2cad.agent import generate_clarification, generate_execute_repair
from src.text2cad.env import load_dotenv
from src.text2cad.evaluation import evaluate_cad_run, flatten_evaluation
from src.text2cad.renderer import render_stl_to_png
from src.text2cad.runner import execute_cadquery_code


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "cadquery_rag" / "results"
PROMPT_PATH = ROOT / "experiments" / "cadquery_rag" / "prompt_set.json"
CONDITIONS = ("baseline", "lightweight_rag", "full_rag")
MAX_REPAIRS = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--limit", type=int, help="Only run the first N prompts.")
    parser.add_argument("--resume", action="store_true", help="Skip completed condition/prompt pairs.")
    parser.add_argument("--seed", type=int, help="Experiment seed to record for every job.")
    parser.add_argument(
        "--send-seed",
        action="store_true",
        help="Send --seed to an API that explicitly supports the seed field.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=MAX_REPAIRS,
        help="Unified execution-and-constraint repair budget for each program.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    prompts = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    if args.limit:
        prompts = prompts[: args.limit]

    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    records_path = experiment_dir / "records.json"
    records = _load_records(records_path) if args.resume else {}

    metadata_path = experiment_dir / "experiment_metadata.json"
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.resume and metadata_path.exists()
        else {}
    )
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("OPENAI_MODEL", "not set"),
        "base_url_host": _base_url_host(os.getenv("OPENAI_BASE_URL", "")),
        "conditions": list(CONDITIONS),
        "rag_library": "knowledge/cadquery_reference.json (original 8 entries)",
        "rag_top_k": 3,
        "max_repairs": args.max_repairs,
        "seed_requested": args.seed,
        "seed_sent_to_provider": args.seed is not None and args.send_seed,
        "api_retries": 3,
        "temperature": 0.1,
        "pricing": {
            "input_cost_per_1m_tokens": os.getenv("LLM_INPUT_COST_PER_1M_TOKENS"),
            "output_cost_per_1m_tokens": os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS"),
            "currency": os.getenv("LLM_COST_CURRENCY", "CNY"),
        },
        "prompt_count": len(prompts),
        "order": "Alternating condition order by prompt ID",
    }
    if existing_metadata:
        metadata["started_at_utc"] = existing_metadata["started_at_utc"]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    jobs = []
    for index, prompt in enumerate(prompts):
        order = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        jobs.extend((prompt, condition) for condition in order)

    for job_index, (prompt, condition) in enumerate(jobs, start=1):
        key = _record_key(prompt["id"], condition, args.seed)
        if args.resume and key in records and records[key].get("status") == "completed":
            print(f"[{job_index}/{len(jobs)}] Skipping completed {key}", flush=True)
            continue
        print(f"\n[{job_index}/{len(jobs)}] Running {key}", flush=True)
        records[key] = _run_one(
            prompt,
            condition,
            experiment_dir,
            seed=args.seed,
            send_seed=args.send_seed,
            max_repairs=args.max_repairs,
        )
        records_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_csv(experiment_dir / "results.csv", records)

    _recheck_assembly_export_failures(records, experiment_dir, args.max_repairs)
    records_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(experiment_dir / "results.csv", records)
    print(f"\nExperiment complete: {experiment_dir}", flush=True)


def _run_one(
    prompt: dict,
    condition: str,
    experiment_dir: Path,
    *,
    seed: int | None,
    send_seed: bool,
    max_repairs: int,
) -> dict:
    pair_dir = experiment_dir / "raw" / prompt["id"] / condition
    if seed is not None:
        pair_dir = pair_dir / f"seed_{seed}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    log_path = pair_dir / "progress.log"
    started = time.monotonic()

    def progress(message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    record = {
        "prompt_id": prompt["id"],
        "title": prompt.get("title"),
        "difficulty": prompt["difficulty"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "target_dimensions_mm": prompt.get("target_dimensions_mm"),
        "constraints": prompt.get("constraints", []),
        "manual_checks": prompt.get("manual_checks", []),
        "expected_behavior": prompt.get("expected_behavior", "generate"),
        "task_type": (
            "clarification"
            if prompt.get("expected_behavior") == "clarify_before_generation"
            else "cad_generation"
        ),
        "cad_evaluation_eligible": (
            prompt.get("expected_behavior") != "clarify_before_generation"
        ),
        "parameterization_required": prompt.get("parameterization_required", True),
        "tags": prompt.get("tags", []),
        "condition": condition,
        "seed_requested": seed,
        "seed_sent_to_provider": seed is not None and send_seed,
        "configured_repair_budget": max_repairs,
        "repair_budget": max_repairs,
        "status": "failed",
        "compile_success": False,
        "visible_geometry": False,
        "dimension_reasonable": False,
        "elapsed_seconds": None,
        "error": "",
    }
    before = set(pair_dir.glob("run_*"))
    if prompt.get("expected_behavior") == "clarify_before_generation":
        record.update(
            {
                "repair_budget": 0,
                "compile_success": None,
                "visible_geometry": None,
                "dimension_reasonable": None,
                "clarification_evaluation_status": "pending",
                "clarification_success": None,
            }
        )
        try:
            result = generate_clarification(
                description=prompt["prompt"],
                output_root=pair_dir,
                rag_mode={
                    "baseline": "off",
                    "lightweight_rag": "lightweight",
                    "full_rag": "full",
                }[condition],
                rag_top_k=3,
                api_timeout=300,
                api_max_retries=3,
                progress_callback=progress,
                seed=seed,
                send_seed=send_seed,
                run_context={
                    "prompt_id": prompt["id"],
                    "difficulty": prompt["difficulty"],
                    "category": prompt["category"],
                    "condition": condition,
                    "expected_behavior": "clarify_before_generation",
                },
            )
            record["run_dir"] = str(result.run_dir)
            record["reference_ids"] = list(result.reference_ids)
            record["attempt_count"] = 0
            record["response_count"] = 1
            record["clarification_response"] = result.response
            record["clarification_response_path"] = str(result.response_path)
            record["clarification_response_characters"] = len(result.response)
            record["telemetry"] = result.telemetry
            record.update(_flatten_telemetry(result.telemetry))
            record["status"] = "completed"
        except Exception as exc:
            new_dirs = sorted(set(pair_dir.glob("run_*")) - before)
            if new_dirs:
                run_dir = new_dirs[-1]
                record["run_dir"] = str(run_dir)
                summary_path = run_dir / "agent_run.json"
                if summary_path.exists():
                    telemetry = json.loads(summary_path.read_text(encoding="utf-8"))
                    record["telemetry"] = telemetry
                    record.update(_flatten_telemetry(telemetry))
            record["clarification_evaluation_status"] = "unavailable"
            record["error"] = f"{type(exc).__name__}: {exc}"
            progress(f"Clarification job failed: {record['error']}")
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 2)
        return record

    try:
        result = generate_execute_repair(
            description=prompt["prompt"],
            max_repairs=max_repairs,
            output_root=pair_dir,
            python_executable=os.getenv("CADQUERY_PYTHON"),
            rag_mode={
                "baseline": "off",
                "lightweight_rag": "lightweight",
                "full_rag": "full",
            }[condition],
            rag_top_k=3,
            api_timeout=300,
            api_max_retries=3,
            execution_timeout=180,
            progress_callback=progress,
            seed=seed,
            send_seed=send_seed,
            run_context={
                "prompt_id": prompt["id"],
                "difficulty": prompt["difficulty"],
                "category": prompt["category"],
                "condition": condition,
                "expected_behavior": prompt.get("expected_behavior", "generate"),
            },
            constraints=prompt.get("constraints", []),
        )
        record["run_dir"] = str(result.run_dir)
        record["reference_ids"] = list(result.reference_ids)
        record["attempt_count"] = len(result.attempts)
        record["telemetry"] = result.telemetry
        record.update(_flatten_telemetry(result.telemetry))
        final = result.final_attempt
        record["compile_success"] = final.ok
        record["returncode"] = final.returncode
        record["error"] = final.stderr[-4000:] if final.stderr else ""
        record["code_path"] = str(final.code_path)
        record["step_path"] = str(final.step_path)
        record["stl_path"] = str(final.stl_path)
        record.update(_code_metrics(final.code_path))
        evaluation = result.final_evaluation
        evaluation_path = result.run_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record["evaluation_path"] = str(evaluation_path)
        record["evaluation"] = evaluation
        record.update(flatten_evaluation(evaluation))
        if evaluation["geometry"]["geometry_valid"]:
            preview_path = pair_dir / "preview.png"
            render_stl_to_png(final.stl_path, preview_path)
            record["preview_path"] = str(preview_path)
        record["status"] = "completed"
    except Exception as exc:
        new_dirs = sorted(set(pair_dir.glob("run_*")) - before)
        if new_dirs:
            run_dir = new_dirs[-1]
            record["run_dir"] = str(run_dir)
            agent_summary_path = run_dir / "agent_run.json"
            attempt_returncodes = []
            step_path = run_dir / "model.step"
            stl_path = run_dir / "model.stl"
            if agent_summary_path.exists():
                telemetry = json.loads(agent_summary_path.read_text(encoding="utf-8"))
                record["telemetry"] = telemetry
                record.update(_flatten_telemetry(telemetry))
                record["attempt_count"] = telemetry["attempt_count"]
                attempt_returncodes = [
                    attempt["returncode"] for attempt in telemetry["attempts"]
                ]
                if telemetry["attempts"]:
                    final_attempt = telemetry["attempts"][-1]
                    step_path = Path(final_attempt["step_path"])
                    stl_path = Path(final_attempt["stl_path"])
            if "evaluation" not in record:
                evaluation = evaluate_cad_run(
                    attempt_returncodes=attempt_returncodes,
                    step_path=step_path,
                    stl_path=stl_path,
                    constraints=prompt.get("constraints", []),
                    repair_budget=max_repairs,
                )
                evaluation_path = run_dir / "evaluation.json"
                evaluation_path.write_text(
                    json.dumps(evaluation, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                record["evaluation_path"] = str(evaluation_path)
                record["evaluation"] = evaluation
                record.update(flatten_evaluation(evaluation))
        record["error"] = f"{type(exc).__name__}: {exc}"
        progress(f"Experiment job failed: {record['error']}")
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return record


def _code_metrics(code_path: Path) -> dict:
    code = code_path.read_text(encoding="utf-8")
    lines = [line for line in code.splitlines() if line.strip()]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "code_lines": len(lines),
            "named_numeric_parameters": 0,
            "has_function": False,
            "has_comments": "#" in code,
            "parameterized_code": False,
        }
    parameter_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, (ast.Constant, ast.UnaryOp)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id != "result":
                parameter_names.add(target.id)
    return {
        "code_lines": len(lines),
        "named_numeric_parameters": len(parameter_names),
        "parameter_names": sorted(parameter_names),
        "has_function": any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)),
        "has_comments": "#" in code,
        "parameterized_code": len(parameter_names) >= 1,
    }


def _recheck_assembly_export_failures(
    records: dict, experiment_dir: Path, repair_budget: int
) -> None:
    """Recheck results affected by the original Assembly export harness bug."""
    for key, record in records.items():
        error = record.get("error", "")
        if "compound: 0 methods found" not in error:
            continue
        print(f"Rechecking {key} with Assembly-to-Compound export...", flush=True)
        code_path = Path(record["code_path"])
        run_dir = Path(record["run_dir"])
        record["initial_harness_error"] = error
        execution = execute_cadquery_code(
            code_path.read_text(encoding="utf-8"),
            description=record["prompt"],
            run_dir=run_dir,
            python_executable=os.getenv("CADQUERY_PYTHON"),
            timeout=180,
            attempt=0,
        )
        record["compile_success"] = execution.ok
        record["returncode"] = execution.returncode
        record["error"] = execution.stderr[-4000:] if execution.stderr else ""
        if not execution.ok:
            continue
        evaluation = evaluate_cad_run(
            attempt_returncodes=[execution.returncode],
            step_path=execution.step_path,
            stl_path=execution.stl_path,
            constraints=record.get("constraints", []),
            repair_budget=repair_budget,
        )
        evaluation_path = run_dir / "evaluation.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record["evaluation_path"] = str(evaluation_path)
        record["evaluation"] = evaluation
        record.update(flatten_evaluation(evaluation))
        preview_path = run_dir.parent / "preview.png"
        render_stl_to_png(execution.stl_path, preview_path)
        record["preview_path"] = str(preview_path)
        record["harness_rechecked"] = True


def _load_records(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, records: dict) -> None:
    rows = list(records.values())
    fields = [
        "prompt_id", "title", "difficulty", "category", "condition", "status",
        "task_type", "cad_evaluation_eligible",
        "expected_behavior", "parameterization_required",
        "seed_requested", "seed_sent_to_provider", "configured_repair_budget",
        "repair_budget",
        "total_repair_budget", "execution_repairs", "constraint_repairs",
        "execution_pass_within_repair_budget",
        "constraint_pass_within_repair_budget",
        "end_to_end_pass_within_repair_budget",
        "compile_success", "first_generation_success", "success_within_repair_budget",
        "repairs_used", "step_export_success", "stl_export_success", "geometry_valid",
        "step_geometry_valid", "step_solid_count", "cylindrical_hole_count",
        "visible_geometry", "dimension_constraint_satisfied", "dimension_reasonable",
        "constraint_groups_registered", "constraint_groups_evaluated",
        "constraint_evaluation_coverage", "constraint_groups_passed",
        "constraint_groups_total", "constraint_group_pass_rate",
        "all_evaluated_hard_constraints_satisfied",
        "all_registered_hard_constraints_satisfied", "task_success_v1",
        "task_success_v2",
        "llm_call_count", "llm_transport_retries", "prompt_tokens",
        "completion_tokens", "total_tokens", "cached_prompt_tokens",
        "reasoning_tokens", "text_completion_tokens",
        "token_usage_complete", "estimated_cost", "cost_currency",
        "rag_retrieval_seconds", "initial_generation_seconds",
        "repair_generation_seconds", "llm_seconds", "cad_execution_seconds",
        "end_to_end_seconds",
        "actual_dimensions_mm", "named_numeric_parameters", "parameterized_code",
        "failed_constraint_ids", "unevaluated_constraint_ids",
        "elapsed_seconds", "reference_ids", "error", "code_path", "preview_path",
        "evaluation_path", "clarification_evaluation_status",
        "clarification_success", "clarification_response_characters",
        "clarification_response_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _base_url_host(base_url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(base_url).netloc


def _record_key(prompt_id: str, condition: str, seed: int | None) -> str:
    base = f"{prompt_id}:{condition}"
    return base if seed is None else f"{base}:seed_{seed}"


def _flatten_telemetry(telemetry: dict) -> dict:
    usage = telemetry["token_usage"]
    cost = telemetry["cost"]
    timing = telemetry["timing"]
    return {
        "total_repair_budget": telemetry.get(
            "total_repair_budget", telemetry["max_repairs"]
        ),
        "execution_repairs": telemetry.get("execution_repairs", 0),
        "constraint_repairs": telemetry.get("constraint_repairs", 0),
        "execution_pass_within_repair_budget": telemetry.get(
            "execution_pass_within_repair_budget"
        ),
        "constraint_pass_within_repair_budget": telemetry.get(
            "constraint_pass_within_repair_budget"
        ),
        "end_to_end_pass_within_repair_budget": telemetry.get(
            "end_to_end_pass_within_repair_budget"
        ),
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


if __name__ == "__main__":
    main()
