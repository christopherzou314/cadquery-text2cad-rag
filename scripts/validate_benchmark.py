"""Validate CadQuery benchmark case files without calling the LLM API."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text2cad.evaluation import EVALUATION_SCHEMA_VERSION  # noqa: E402


BENCHMARK_DIR = ROOT / "experiments" / "cadquery_benchmark_v1"
FULL_CASES = BENCHMARK_DIR / "full_cases.json"
DEFAULT_CASES = FULL_CASES
DEFAULT_CONFIG = BENCHMARK_DIR / "benchmark_config.json"
CATEGORIES = {
    "primitive",
    "holes_and_slots",
    "thin_wall",
    "revolved",
    "complex_boolean",
    "curved_surface",
    "assembly",
    "ambiguous",
    "conflicting",
    "out_of_distribution",
}
DIFFICULTIES = {"easy", "medium", "hard", "complex"}
EXPECTED_BEHAVIORS = {
    "generate",
    "generate_with_reasonable_assumptions",
    "clarify_before_generation",
    "best_effort_generation",
}
CONSTRAINT_TYPES = {
    "bbox_dimensions",
    "bbox_bounds",
    "solid_count",
    "cylindrical_hole_pattern",
}
AXES = {"x", "y", "z"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def validate_benchmark(
    cases_path: Path = DEFAULT_CASES,
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    case_document = _load_json(cases_path, errors, "cases")
    config = _load_json(config_path, errors, "config")
    if errors:
        return errors, {}

    for field in (
        "benchmark_id",
        "benchmark_version",
        "evaluation_schema_version",
        "phase",
        "cases",
    ):
        if field not in case_document:
            errors.append(f"case document: missing required field {field!r}")

    if case_document.get("benchmark_id") != config.get("benchmark_id"):
        errors.append("case document: benchmark_id does not match config")
    if case_document.get("benchmark_version") != config.get("benchmark_version"):
        errors.append("case document: benchmark_version does not match config")
    if case_document.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        errors.append(
            "case document: evaluation_schema_version must equal frozen evaluator "
            f"version {EVALUATION_SCHEMA_VERSION}"
        )

    cases = case_document.get("cases")
    if not isinstance(cases, list):
        errors.append("case document: cases must be a list")
        cases = []
    phase = case_document.get("phase")
    expected_count = config.get("target_case_count")
    if phase != "full":
        errors.append("case document: phase must be 'full'")
    elif len(cases) != expected_count:
        errors.append(
            f"case document: phase {phase!r} requires {expected_count} cases, "
            f"found {len(cases)}"
        )

    category_targets = config.get("category_targets", {})
    if set(category_targets) != CATEGORIES:
        errors.append("config: category_targets must contain exactly the 10 categories")
    if sum(category_targets.values()) != config.get("target_case_count"):
        errors.append("config: category target sum does not equal target_case_count")
    _validate_seeds(config, errors)

    ids: set[str] = set()
    prompts: set[str] = set()
    category_counts: Counter[str] = Counter()
    constraint_counts: Counter[str] = Counter()
    automatic_cases = 0
    manual_cases = 0
    clarification_cases = 0
    for index, case in enumerate(cases):
        label = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label}: must be an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not re.fullmatch(r"B\d{3}", case_id):
            errors.append(f"{label}: id must match B000 format")
            case_id = label
        elif case_id in ids:
            errors.append(f"{label}: duplicate id {case_id!r}")
        ids.add(case_id)
        label = case_id

        prompt = case.get("prompt")
        if not isinstance(prompt, str) or len(prompt.strip()) < 20:
            errors.append(f"{label}: prompt must contain at least 20 characters")
        elif prompt.strip() in prompts:
            errors.append(f"{label}: duplicate prompt")
        else:
            prompts.add(prompt.strip())

        category = case.get("category")
        if category not in CATEGORIES:
            errors.append(f"{label}: unsupported category {category!r}")
        else:
            category_counts[category] += 1
        if case.get("difficulty") not in DIFFICULTIES:
            errors.append(f"{label}: unsupported difficulty")
        behavior = case.get("expected_behavior")
        if behavior not in EXPECTED_BEHAVIORS:
            errors.append(f"{label}: unsupported expected_behavior {behavior!r}")
        if not isinstance(case.get("parameterization_required"), bool):
            errors.append(f"{label}: parameterization_required must be boolean")
        if category == "conflicting" and behavior != "clarify_before_generation":
            errors.append(f"{label}: conflicting cases must request clarification")
        if category == "out_of_distribution" and behavior != "best_effort_generation":
            errors.append(f"{label}: out-of-distribution cases must use best effort")
        if behavior == "clarify_before_generation":
            clarification_cases += 1

        constraints = case.get("constraints")
        if not isinstance(constraints, list):
            errors.append(f"{label}: constraints must be a list")
            constraints = []
        if constraints:
            automatic_cases += 1
        if behavior == "clarify_before_generation" and constraints:
            errors.append(
                f"{label}: clarification-only cases cannot register CAD constraints"
            )
        constraint_ids: set[str] = set()
        for constraint in constraints:
            _validate_constraint(constraint, label, constraint_ids, errors)
            if isinstance(constraint, dict) and constraint.get("type") in CONSTRAINT_TYPES:
                constraint_counts[constraint["type"]] += 1

        manual_checks = case.get("manual_checks")
        if not isinstance(manual_checks, list):
            errors.append(f"{label}: manual_checks must be a list")
            manual_checks = []
        if manual_checks:
            manual_cases += 1
        _validate_manual_checks(manual_checks, label, errors)
        if not constraints and not manual_checks:
            errors.append(f"{label}: requires at least one automatic or manual check")

        tags = case.get("tags")
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            errors.append(f"{label}: tags must be a list of non-empty strings")
        elif len(tags) != len(set(tags)):
            errors.append(f"{label}: tags must be unique")

    case_id_range = config.get("case_id_range", {})
    range_start = case_id_range.get("start")
    range_end = case_id_range.get("end")
    if phase == "full" and (
        not isinstance(range_start, int)
        or isinstance(range_start, bool)
        or not isinstance(range_end, int)
        or isinstance(range_end, bool)
        or range_end < range_start
        or range_end - range_start + 1 != expected_count
    ):
        errors.append(
            "config: case_id_range must define exactly "
            f"{expected_count} ordered IDs"
        )
    elif phase == "full":
        expected_ids = {
            f"B{number:03d}" for number in range(range_start, range_end + 1)
        }
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            unexpected = sorted(ids - expected_ids)
            errors.append(
                f"case document: {phase} phase IDs must be exactly "
                f"B{range_start:03d}-B{range_end:03d}; "
                f"missing={missing}, unexpected={unexpected}"
            )

    if phase == "full":
        for category in sorted(CATEGORIES):
            target = category_targets.get(category)
            actual = category_counts[category]
            if actual != target:
                errors.append(
                    f"case document: full phase category {category!r} requires "
                    f"{target} cases, found {actual}"
                )

    summary = {
        "benchmark_id": case_document.get("benchmark_id"),
        "benchmark_version": case_document.get("benchmark_version"),
        "evaluation_schema_version": case_document.get("evaluation_schema_version"),
        "phase": phase,
        "case_count": len(cases),
        "category_counts": dict(sorted(category_counts.items())),
        "constraint_type_counts": dict(sorted(constraint_counts.items())),
        "cases_with_automatic_constraints": automatic_cases,
        "cases_with_manual_checks": manual_cases,
        "clarification_only_cases": clarification_cases,
        "cad_generation_cases": len(cases) - clarification_cases,
    }
    return errors, summary


def _validate_constraint(
    constraint: Any,
    case_id: str,
    seen_ids: set[str],
    errors: list[str],
) -> None:
    if not isinstance(constraint, dict):
        errors.append(f"{case_id}: each constraint must be an object")
        return
    constraint_id = constraint.get("id")
    if not isinstance(constraint_id, str) or not constraint_id:
        errors.append(f"{case_id}: constraint id must be a non-empty string")
    elif constraint_id in seen_ids:
        errors.append(f"{case_id}: duplicate constraint id {constraint_id!r}")
    else:
        seen_ids.add(constraint_id)
    constraint_type = constraint.get("type")
    if constraint_type not in CONSTRAINT_TYPES:
        errors.append(f"{case_id}/{constraint_id}: unsupported constraint type")
        return
    if not isinstance(constraint.get("hard"), bool):
        errors.append(f"{case_id}/{constraint_id}: hard must be boolean")

    if constraint_type == "bbox_dimensions":
        _validate_axis_values(
            constraint.get("expected_mm"), case_id, constraint_id, errors, positive=True
        )
    elif constraint_type == "bbox_bounds":
        if "expected_min_mm" not in constraint and "expected_max_mm" not in constraint:
            errors.append(f"{case_id}/{constraint_id}: bbox_bounds needs min or max")
        for field in ("expected_min_mm", "expected_max_mm"):
            if field in constraint:
                _validate_axis_values(
                    constraint[field], case_id, constraint_id, errors, positive=False
                )
    elif constraint_type == "solid_count":
        expected = constraint.get("expected")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
            errors.append(f"{case_id}/{constraint_id}: expected must be an integer >= 1")
    elif constraint_type == "cylindrical_hole_pattern":
        count = constraint.get("expected_count")
        diameter = constraint.get("diameter_mm")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            errors.append(f"{case_id}/{constraint_id}: expected_count must be >= 1")
        if not _is_number(diameter) or diameter <= 0:
            errors.append(f"{case_id}/{constraint_id}: diameter_mm must be positive")
        axis = constraint.get("axis")
        axes = {axis} if isinstance(axis, str) else set(axis or [])
        if not axes or not axes.issubset(AXES):
            errors.append(f"{case_id}/{constraint_id}: axis must use x, y, or z")
        if not isinstance(constraint.get("through"), bool):
            errors.append(f"{case_id}/{constraint_id}: through must be boolean")
        pcd = constraint.get("pitch_circle_diameter_mm")
        if pcd is not None and (not _is_number(pcd) or pcd <= 0):
            errors.append(f"{case_id}/{constraint_id}: PCD must be positive")

    for field in (
        "absolute_tolerance_mm",
        "relative_tolerance",
        "diameter_tolerance_mm",
        "position_tolerance_mm",
        "match_tolerance_mm",
    ):
        if field in constraint and (
            not _is_number(constraint[field]) or constraint[field] < 0
        ):
            errors.append(f"{case_id}/{constraint_id}: {field} must be non-negative")


def _validate_axis_values(
    values: Any,
    case_id: str,
    constraint_id: str,
    errors: list[str],
    *,
    positive: bool,
) -> None:
    if not isinstance(values, dict) or not values or not set(values).issubset(AXES):
        errors.append(f"{case_id}/{constraint_id}: expected axes must be non-empty x/y/z")
        return
    for axis, value in values.items():
        if not _is_number(value) or (positive and value <= 0):
            qualifier = "positive" if positive else "numeric"
            errors.append(f"{case_id}/{constraint_id}: axis {axis} must be {qualifier}")


def _validate_manual_checks(
    checks: list[Any], case_id: str, errors: list[str]
) -> None:
    seen: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            errors.append(f"{case_id}: each manual check must be an object")
            continue
        check_id = check.get("id")
        if not isinstance(check_id, str) or not check_id:
            errors.append(f"{case_id}: manual check id must be non-empty")
        elif check_id in seen:
            errors.append(f"{case_id}: duplicate manual check id {check_id!r}")
        else:
            seen.add(check_id)
        if not isinstance(check.get("criterion"), str) or len(check["criterion"]) < 10:
            errors.append(f"{case_id}/{check_id}: criterion is too short")
        if check.get("scale") not in {"binary", "0_to_2", "1_to_5"}:
            errors.append(f"{case_id}/{check_id}: unsupported manual scale")
        if not isinstance(check.get("required"), bool):
            errors.append(f"{case_id}/{check_id}: required must be boolean")


def _validate_seeds(config: dict[str, Any], errors: list[str]) -> None:
    values = config.get("seeds")
    if not isinstance(values, list) or len(values) != 3:
        errors.append("config: seeds must contain exactly 3 values")
    elif any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        errors.append("config: seeds must be integers")
    elif len(values) != len(set(values)):
        errors.append("config: seeds must be unique")


def _load_json(path: Path, errors: list[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{label}: cannot read {path}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}: root JSON value must be an object")
        return {}
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main() -> None:
    args = parse_args()
    errors, summary = validate_benchmark(args.cases.resolve(), args.config.resolve())
    if errors:
        print(f"Benchmark validation failed with {len(errors)} error(s):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Benchmark validation passed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
