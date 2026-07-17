"""Run the reproducible baseline versus lightweight-RAG CadQuery experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text2cad.agent import generate_execute_repair
from src.text2cad.env import load_dotenv
from src.text2cad.renderer import render_stl_to_png
from src.text2cad.runner import execute_cadquery_code


DEFAULT_EXPERIMENT_DIR = ROOT / "experiments" / "cadquery_rag" / "results"
PROMPT_PATH = ROOT / "experiments" / "cadquery_rag" / "prompt_set.json"
CONDITIONS = ("baseline", "lightweight_rag", "full_rag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--limit", type=int, help="Only run the first N prompts.")
    parser.add_argument("--resume", action="store_true", help="Skip completed condition/prompt pairs.")
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
        "max_repairs": 0,
        "api_retries": 3,
        "temperature": 0.1,
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
        key = f"{prompt['id']}:{condition}"
        if args.resume and key in records and records[key].get("status") == "completed":
            print(f"[{job_index}/{len(jobs)}] Skipping completed {key}", flush=True)
            continue
        print(f"\n[{job_index}/{len(jobs)}] Running {key}", flush=True)
        records[key] = _run_one(prompt, condition, experiment_dir)
        records_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _write_csv(experiment_dir / "results.csv", records)

    _recheck_assembly_export_failures(records, experiment_dir)
    records_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(experiment_dir / "results.csv", records)
    print(f"\nExperiment complete: {experiment_dir}", flush=True)


def _run_one(prompt: dict, condition: str, experiment_dir: Path) -> dict:
    pair_dir = experiment_dir / "raw" / prompt["id"] / condition
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
        "difficulty": prompt["difficulty"],
        "category": prompt["category"],
        "prompt": prompt["prompt"],
        "target_dimensions_mm": prompt["target_dimensions_mm"],
        "condition": condition,
        "status": "failed",
        "compile_success": False,
        "visible_geometry": False,
        "dimension_reasonable": False,
        "elapsed_seconds": None,
        "error": "",
    }
    before = set(pair_dir.glob("run_*"))
    try:
        result = generate_execute_repair(
            description=prompt["prompt"],
            max_repairs=0,
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
        )
        record["run_dir"] = str(result.run_dir)
        record["reference_ids"] = list(result.reference_ids)
        record["attempt_count"] = len(result.attempts)
        final = result.final_attempt
        record["compile_success"] = result.success
        record["returncode"] = final.returncode
        record["error"] = final.stderr[-4000:] if final.stderr else ""
        record["code_path"] = str(final.code_path)
        record["step_path"] = str(final.step_path)
        record["stl_path"] = str(final.stl_path)
        record.update(_code_metrics(final.code_path))
        if result.success:
            geometry = _geometry_metrics(final.stl_path, prompt["target_dimensions_mm"])
            record.update(geometry)
            preview_path = pair_dir / "preview.png"
            render_stl_to_png(final.stl_path, preview_path)
            record["preview_path"] = str(preview_path)
        record["status"] = "completed"
    except Exception as exc:
        new_dirs = sorted(set(pair_dir.glob("run_*")) - before)
        if new_dirs:
            record["run_dir"] = str(new_dirs[-1])
        record["error"] = f"{type(exc).__name__}: {exc}"
        progress(f"Experiment job failed: {record['error']}")
    finally:
        record["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return record


def _geometry_metrics(stl_path: Path, target_dimensions: list[float | None]) -> dict:
    import vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    mesh = reader.GetOutput()
    bounds = mesh.GetBounds()
    dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    visible = (
        mesh.GetNumberOfPoints() > 3
        and mesh.GetNumberOfCells() > 1
        and all(math.isfinite(value) for value in dimensions)
        and all(value > 0.01 for value in dimensions)
    )
    relative_errors = [
        None if target is None else abs(actual - target) / target
        for actual, target in zip(dimensions, target_dimensions, strict=True)
    ]
    measured_errors = [value for value in relative_errors if value is not None]
    return {
        "visible_geometry": visible,
        "mesh_points": mesh.GetNumberOfPoints(),
        "mesh_cells": mesh.GetNumberOfCells(),
        "actual_dimensions_mm": [round(value, 3) for value in dimensions],
        "dimension_relative_errors": [
            None if value is None else round(value, 3) for value in relative_errors
        ],
        "dimension_reasonable": max(measured_errors, default=0) <= 0.25,
    }


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


def _recheck_assembly_export_failures(records: dict, experiment_dir: Path) -> None:
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
        record.update(_geometry_metrics(execution.stl_path, record["target_dimensions_mm"]))
        preview_path = experiment_dir / "raw" / record["prompt_id"] / record["condition"] / "preview.png"
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
        "prompt_id", "difficulty", "category", "condition", "status",
        "compile_success", "visible_geometry", "dimension_reasonable",
        "actual_dimensions_mm", "named_numeric_parameters", "parameterized_code",
        "elapsed_seconds", "reference_ids", "error", "code_path", "preview_path",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _base_url_host(base_url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(base_url).netloc


if __name__ == "__main__":
    main()
