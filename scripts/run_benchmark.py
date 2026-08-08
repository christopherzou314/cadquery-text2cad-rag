"""Run the seed-controlled CadQuery benchmark with resume support."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_rag_experiment import (  # noqa: E402
    CONDITIONS,
    _base_url_host,
    _load_records,
    _record_key,
    _run_one,
    _write_csv,
)
from scripts.validate_benchmark import (  # noqa: E402
    DEFAULT_CONFIG,
    FULL_CASES,
    validate_benchmark,
)
from src.text2cad.env import load_dotenv  # noqa: E402


DEFAULT_RESULTS_DIR = (
    ROOT / "experiments" / "cadquery_benchmark_v1" / "results" / "full_b2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=FULL_CASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, help="Use only the first N cases.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override the configured seed list for the selected benchmark phase.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        help="Override the three configured generation conditions.",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        help="Run at most N ordered jobs; useful for an API smoke test.",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        help="Override the configured repair budget for the selected phase.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs already marked completed and retry failed jobs.",
    )
    parser.add_argument(
        "--do-not-send-seed",
        action="store_true",
        help="Record seeds without sending them to the provider.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env")
    cases_path = args.cases.resolve()
    config_path = args.config.resolve()
    errors, validation_summary = validate_benchmark(cases_path, config_path)
    if errors:
        for error in errors:
            print(f"Benchmark validation error: {error}", file=sys.stderr)
        raise SystemExit(1)

    case_document = json.loads(cases_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = case_document["cases"]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        cases = cases[: args.limit]
    phase = case_document["phase"]
    seeds = args.seeds or config["seeds"]
    conditions = tuple(args.conditions or config["conditions"])
    max_repairs = (
        args.max_repairs
        if args.max_repairs is not None
        else int(config["repair_budget"])
    )
    if max_repairs < 0:
        raise SystemExit("--max-repairs cannot be negative")
    send_seed = not args.do_not_send_seed

    experiment_dir = args.experiment_dir.resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    records_path = experiment_dir / "records.json"
    records = _load_records(records_path) if args.resume else {}
    jobs = _ordered_jobs(cases, seeds, conditions)
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            raise SystemExit("--max-jobs must be at least 1")
        jobs = jobs[: args.max_jobs]

    metadata_path = experiment_dir / "experiment_metadata.json"
    existing_metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.resume and metadata_path.exists()
        else {}
    )
    metadata = {
        "benchmark_id": config["benchmark_id"],
        "benchmark_version": config["benchmark_version"],
        "evaluation_schema_version": config["evaluation_schema_version"],
        "phase": phase,
        "started_at_utc": existing_metadata.get(
            "started_at_utc", datetime.now(timezone.utc).isoformat()
        ),
        "last_updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": os.getenv("OPENAI_MODEL", "not set"),
        "base_url_host": _base_url_host(os.getenv("OPENAI_BASE_URL", "")),
        "case_ids": [case["id"] for case in cases],
        "conditions": list(conditions),
        "seeds": seeds,
        "seed_sent_to_provider": send_seed,
        "max_repairs": max_repairs,
        "clarification_protocol": config.get("clarification_protocol"),
        "rag_top_k": 3,
        "api_retries": 3,
        "temperature": 0.1,
        "pricing": {
            "input_cost_per_1m_tokens": os.getenv("LLM_INPUT_COST_PER_1M_TOKENS"),
            "output_cost_per_1m_tokens": os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS"),
            "currency": os.getenv("LLM_COST_CURRENCY", "CNY"),
        },
        "job_count": len(jobs),
        "completed_job_count": sum(
            record.get("status") == "completed" for record in records.values()
        ),
        "validation_summary": validation_summary,
        "condition_order": "balanced rotation by case and seed",
    }
    _write_json_atomic(metadata_path, metadata)

    for job_index, (case, condition, seed) in enumerate(jobs, start=1):
        key = _record_key(case["id"], condition, seed)
        if args.resume and records.get(key, {}).get("status") == "completed":
            print(f"[{job_index}/{len(jobs)}] Skipping completed {key}", flush=True)
            continue
        print(f"\n[{job_index}/{len(jobs)}] Running {key}", flush=True)
        records[key] = _run_one(
            case,
            condition,
            experiment_dir,
            seed=seed,
            send_seed=send_seed,
            max_repairs=max_repairs,
        )
        _write_json_atomic(records_path, records)
        _write_csv(experiment_dir / "results.csv", records)
        metadata["last_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["completed_job_count"] = sum(
            record.get("status") == "completed" for record in records.values()
        )
        _write_json_atomic(metadata_path, metadata)

    print(f"\n{phase.title()} run complete: {experiment_dir}", flush=True)


def _ordered_jobs(
    cases: list[dict], seeds: list[int], conditions: tuple[str, ...]
) -> list[tuple[dict, str, int]]:
    jobs = []
    for seed_index, seed in enumerate(seeds):
        for case_index, case in enumerate(cases):
            rotation = (seed_index + case_index) % len(conditions)
            order = conditions[rotation:] + conditions[:rotation]
            jobs.extend((case, condition, seed) for condition in order)
    return jobs


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


if __name__ == "__main__":
    main()
