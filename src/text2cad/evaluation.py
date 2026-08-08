"""Quantitative evaluation for generated CadQuery artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from .step_features import empty_step_features, read_step_features


AXES = ("x", "y", "z")
EVALUATION_SCHEMA_VERSION = "2.0"


def layered_pass_metrics(evaluation: dict[str, Any]) -> dict[str, bool | None]:
    """Derive execution, constraint, and end-to-end pass states."""
    execution = evaluation["execution"]
    exports = evaluation["exports"]
    summary = evaluation["summary"]
    execution_pass = bool(
        execution["final_success"]
        and exports["step_success"]
        and exports["stl_success"]
    )
    has_hard_constraints = summary["hard_constraints_total"] > 0
    constraint_pass = (
        summary["all_registered_hard_constraints_satisfied"]
        if has_hard_constraints
        else None
    )
    end_to_end_pass = bool(
        summary["artifact_success"]
        and (constraint_pass is True if has_hard_constraints else True)
    )
    return {
        "execution_pass": execution_pass,
        "constraint_pass": constraint_pass,
        "end_to_end_pass": end_to_end_pass,
        "constraint_eligible": has_hard_constraints,
    }


def pass_by_repair_budget(
    attempt_evaluations: list[dict[str, Any]], max_repair_budget: int
) -> list[dict[str, Any]]:
    """Calculate cumulative pass states for every repair budget from 0 through B."""
    if max_repair_budget < 0:
        raise ValueError("max_repair_budget cannot be negative")
    states = [layered_pass_metrics(item) for item in attempt_evaluations]
    constraint_eligible = any(state["constraint_eligible"] for state in states)
    curve = []
    for budget in range(max_repair_budget + 1):
        available = states[: budget + 1]
        curve.append(
            {
                "repair_budget": budget,
                "execution_pass": any(
                    state["execution_pass"] is True for state in available
                ),
                "constraint_pass": (
                    any(state["constraint_pass"] is True for state in available)
                    if constraint_eligible
                    else None
                ),
                "end_to_end_pass": any(
                    state["end_to_end_pass"] is True for state in available
                ),
            }
        )
    return curve


def repair_trigger(evaluation: dict[str, Any]) -> str | None:
    """Classify why another model-generated code attempt is needed."""
    metrics = layered_pass_metrics(evaluation)
    if metrics["end_to_end_pass"]:
        return None
    if not evaluation["summary"]["artifact_success"]:
        return "execution"
    if metrics["constraint_eligible"] and metrics["constraint_pass"] is not True:
        return "constraint"
    return "execution"


def format_repair_feedback(evaluation: dict[str, Any], trigger: str) -> str:
    """Format deterministic execution or hard-constraint feedback for the LLM."""
    if trigger == "execution":
        return _format_artifact_feedback(evaluation)
    if trigger != "constraint":
        raise ValueError(f"Unknown repair trigger: {trigger}")

    failed = [
        item
        for item in evaluation["constraints"]
        if item["hard"] and (not item["evaluated"] or item["passed"] is not True)
    ]
    satisfied = [
        item["id"]
        for item in evaluation["constraints"]
        if item["hard"] and item["evaluated"] and item["passed"] is True
    ]
    lines = ["AUTOMATIC HARD-CONSTRAINT EVALUATION", ""]
    for item in failed:
        status = "FAILED" if item["evaluated"] else "UNEVALUATED"
        lines.append(
            f"{status} constraint: {item['id']} (type={item['type']})"
        )
        if item.get("reason"):
            lines.append(f"Reason: {item['reason']}")
        for check in item.get("checks", []):
            lines.append(_format_constraint_check(check))
        lines.append("")
    if satisfied:
        lines.append("Already satisfied hard constraints: " + ", ".join(satisfied))
    geometry = evaluation.get("geometry", {})
    if geometry.get("geometry_valid"):
        lines.append(
            "Measured bounding box (mm): "
            + _format_axis_values(geometry.get("dimensions_mm") or {})
        )
    lines.extend(
        [
            "Preserve every already satisfied requirement while correcting the failures.",
            "Do not add dummy or disconnected geometry only to manipulate measurements.",
        ]
    )
    return "\n".join(lines).strip()


def evaluate_cad_run(
    *,
    attempt_returncodes: Iterable[int],
    step_path: Path,
    stl_path: Path,
    constraints: list[dict[str, Any]],
    repair_budget: int,
) -> dict[str, Any]:
    """Evaluate execution, exported artifacts, geometry, and registered constraints."""
    returncodes = list(attempt_returncodes)
    first_execution_success = bool(returncodes) and returncodes[0] == 0
    final_execution_success = bool(returncodes) and returncodes[-1] == 0
    step_export_success = _nonempty_file(step_path)
    stl_export_success = _nonempty_file(stl_path)

    geometry = _read_stl_geometry(stl_path) if stl_export_success else _empty_geometry()
    step_geometry = (
        read_step_features(step_path)
        if step_export_success
        else empty_step_features("STEP artifact is unavailable.")
    )
    constraint_results = [
        evaluate_constraint(constraint, geometry, step_geometry)
        for constraint in constraints
    ]
    evaluated = [result for result in constraint_results if result["evaluated"]]
    passed = [result for result in evaluated if result["passed"]]
    registered_hard = [result for result in constraint_results if result["hard"]]
    hard = [result for result in evaluated if result["hard"]]
    hard_passed = [result for result in hard if result["passed"]]
    all_evaluated_hard_passed = (
        all(result["passed"] for result in hard) if hard else None
    )
    all_registered_hard_passed = (
        len(hard) == len(registered_hard)
        and all(result["passed"] for result in hard)
        if registered_hard
        else None
    )
    artifact_success = (
        final_execution_success
        and step_export_success
        and stl_export_success
        and geometry["geometry_valid"]
    )
    v1_results = [
        result
        for result in constraint_results
        if result["type"] in {"bbox_dimensions", "bbox_bounds"}
    ]
    v1_registered_hard = [result for result in v1_results if result["hard"]]
    v1_evaluated_hard = [
        result for result in v1_registered_hard if result["evaluated"]
    ]
    v1_hard_passed = (
        len(v1_evaluated_hard) == len(v1_registered_hard)
        and all(result["passed"] for result in v1_evaluated_hard)
        if v1_registered_hard
        else None
    )
    task_success_v1 = artifact_success and v1_hard_passed is True
    task_success_v2 = artifact_success and all_registered_hard_passed is True

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "execution": {
            "first_generation_success": first_execution_success,
            "final_success": final_execution_success,
            "success_within_repair_budget": any(code == 0 for code in returncodes),
            "attempts_used": len(returncodes),
            "repairs_used": max(0, len(returncodes) - 1),
            "repair_budget": repair_budget,
            "returncodes": returncodes,
        },
        "exports": {
            "step_success": step_export_success,
            "stl_success": stl_export_success,
            "step_size_bytes": _file_size(step_path),
            "stl_size_bytes": _file_size(stl_path),
        },
        "geometry": geometry,
        "step_geometry": step_geometry,
        "constraints": constraint_results,
        "summary": {
            "artifact_success": artifact_success,
            "constraint_groups_registered": len(constraint_results),
            "constraint_groups_evaluated": len(evaluated),
            "constraint_evaluation_coverage": (
                round(len(evaluated) / len(constraint_results), 6)
                if constraint_results
                else None
            ),
            "constraint_groups_passed": len(passed),
            "constraint_groups_total": len(evaluated),
            "constraint_group_pass_rate": (
                round(len(passed) / len(evaluated), 6) if evaluated else None
            ),
            "hard_constraints_passed": len(hard_passed),
            "hard_constraints_evaluated": len(hard),
            "hard_constraints_total": len(registered_hard),
            "all_evaluated_hard_constraints_satisfied": all_evaluated_hard_passed,
            "all_registered_hard_constraints_satisfied": all_registered_hard_passed,
            "task_success_v1": task_success_v1,
            "task_success_v2": task_success_v2,
        },
    }


def flatten_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Expose high-level metrics for tabular experiment records."""
    execution = evaluation["execution"]
    exports = evaluation["exports"]
    geometry = evaluation["geometry"]
    step_geometry = evaluation.get("step_geometry") or empty_step_features()
    summary = evaluation["summary"]
    dimensions = geometry.get("dimensions_mm") or {}
    dimension_result = next(
        (
            item
            for item in evaluation["constraints"]
            if item["type"] == "bbox_dimensions" and item["evaluated"]
        ),
        None,
    )
    failed_constraint_ids = [
        result["id"]
        for result in evaluation["constraints"]
        if result["evaluated"] and not result["passed"]
    ]
    unevaluated_constraint_ids = [
        result["id"]
        for result in evaluation["constraints"]
        if not result["evaluated"]
    ]
    return {
        "first_generation_success": execution["first_generation_success"],
        "success_within_repair_budget": execution["success_within_repair_budget"],
        "repairs_used": execution["repairs_used"],
        "step_export_success": exports["step_success"],
        "stl_export_success": exports["stl_success"],
        "geometry_valid": geometry["geometry_valid"],
        "visible_geometry": geometry["geometry_valid"],
        "mesh_points": geometry["mesh_points"],
        "mesh_cells": geometry["mesh_cells"],
        "step_geometry_valid": step_geometry["step_valid"],
        "step_solid_count": step_geometry["solid_count"],
        "cylindrical_hole_count": step_geometry["cylindrical_hole_count"],
        "actual_dimensions_mm": [dimensions.get(axis) for axis in AXES],
        "dimension_constraint_satisfied": (
            dimension_result["passed"] if dimension_result else None
        ),
        "dimension_reasonable": dimension_result["passed"] if dimension_result else False,
        "constraint_groups_registered": summary["constraint_groups_registered"],
        "constraint_groups_evaluated": summary["constraint_groups_evaluated"],
        "constraint_evaluation_coverage": summary["constraint_evaluation_coverage"],
        "constraint_groups_passed": summary["constraint_groups_passed"],
        "constraint_groups_total": summary["constraint_groups_total"],
        "constraint_group_pass_rate": summary["constraint_group_pass_rate"],
        "all_evaluated_hard_constraints_satisfied": summary[
            "all_evaluated_hard_constraints_satisfied"
        ],
        "all_registered_hard_constraints_satisfied": summary[
            "all_registered_hard_constraints_satisfied"
        ],
        "task_success_v1": summary["task_success_v1"],
        "task_success_v2": summary["task_success_v2"],
        "failed_constraint_ids": failed_constraint_ids,
        "unevaluated_constraint_ids": unevaluated_constraint_ids,
    }


def evaluate_constraint(
    constraint: dict[str, Any],
    geometry: dict[str, Any],
    step_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one supported constraint against measured STL or STEP geometry."""
    constraint_type = constraint.get("type")
    result: dict[str, Any] = {
        "id": constraint.get("id", constraint_type or "unnamed"),
        "type": constraint_type,
        "hard": bool(constraint.get("hard", True)),
        "evaluated": False,
        "passed": None,
        "checks": [],
    }
    if constraint_type in {"bbox_dimensions", "bbox_bounds"} and not geometry.get(
        "geometry_valid"
    ):
        result["reason"] = "STL geometry is unavailable or invalid."
        return result

    if constraint_type in {"solid_count", "cylindrical_hole_pattern"} and not (
        step_geometry or {}
    ).get("step_valid"):
        result["reason"] = "STEP geometry is unavailable or invalid."
        return result

    if constraint_type == "bbox_dimensions":
        expected = constraint.get("expected_mm", {})
        actual = geometry["dimensions_mm"]
        result["checks"] = _axis_checks(
            expected,
            actual,
            constraint,
            value_label="dimension",
        )
    elif constraint_type == "bbox_bounds":
        checks = []
        for bound_name, geometry_key in (
            ("expected_min_mm", "bounds_min_mm"),
            ("expected_max_mm", "bounds_max_mm"),
        ):
            checks.extend(
                _axis_checks(
                    constraint.get(bound_name, {}),
                    geometry[geometry_key],
                    constraint,
                    value_label=bound_name.removeprefix("expected_").removesuffix("_mm"),
                )
            )
        result["checks"] = checks
    elif constraint_type == "solid_count":
        expected = int(constraint["expected"])
        actual = int(step_geometry["solid_count"])
        result["checks"] = [
            {
                "value": "solid_count",
                "expected": expected,
                "actual": actual,
                "passed": actual == expected,
            }
        ]
    elif constraint_type == "cylindrical_hole_pattern":
        hole_result = _cylindrical_hole_checks(constraint, step_geometry)
        result["checks"] = hole_result["checks"]
        result["matched_features"] = hole_result["matched_features"]
        result["candidate_count_before_diameter_filter"] = hole_result[
            "candidate_count_before_diameter_filter"
        ]
    else:
        result["reason"] = f"Unsupported constraint type: {constraint_type!r}."
        return result

    result["evaluated"] = bool(result["checks"])
    result["passed"] = (
        all(check["passed"] for check in result["checks"])
        if result["checks"]
        else None
    )
    if not result["checks"]:
        result["reason"] = "Constraint has no expected axis values."
    return result


def _cylindrical_hole_checks(
    constraint: dict[str, Any], step_geometry: dict[str, Any]
) -> dict[str, Any]:
    expected_diameter = float(constraint["diameter_mm"])
    allowed_axes = constraint.get("axis")
    if isinstance(allowed_axes, str):
        allowed_axes = [allowed_axes]
    allowed_axes = set(allowed_axes or AXES)
    axis_candidates = [
        hole
        for hole in step_geometry["cylindrical_holes"]
        if hole.get("axis") in allowed_axes
    ]
    match_tolerance = float(
        constraint.get(
            "match_tolerance_mm",
            max(2.0, expected_diameter * 0.35),
        )
    )
    matched = [
        hole
        for hole in axis_candidates
        if abs(float(hole["diameter_mm"]) - expected_diameter) <= match_tolerance
    ]
    expected_count = int(constraint["expected_count"])
    checks: list[dict[str, Any]] = [
        {
            "value": "hole_count",
            "expected": expected_count,
            "actual": len(matched),
            "allowed_axes": sorted(allowed_axes),
            "diameter_match_tolerance_mm": match_tolerance,
            "passed": len(matched) == expected_count,
        }
    ]

    diameter_tolerance = float(constraint.get("diameter_tolerance_mm", 0.3))
    for index, hole in enumerate(matched):
        actual = float(hole["diameter_mm"])
        error = abs(actual - expected_diameter)
        checks.append(
            {
                "value": "hole_diameter",
                "feature_index": index,
                "expected_mm": expected_diameter,
                "actual_mm": actual,
                "absolute_error_mm": round(error, 6),
                "tolerance_mm": diameter_tolerance,
                "passed": error <= diameter_tolerance,
            }
        )

    if "through" in constraint:
        expected_through = bool(constraint["through"])
        for index, hole in enumerate(matched):
            actual_through = hole.get("through")
            checks.append(
                {
                    "value": "through_hole",
                    "feature_index": index,
                    "expected": expected_through,
                    "actual": actual_through,
                    "passed": actual_through is expected_through,
                }
            )

    if "pitch_circle_diameter_mm" in constraint:
        expected_pcd = float(constraint["pitch_circle_diameter_mm"])
        pcd_tolerance = float(constraint.get("position_tolerance_mm", 0.5))
        actual_pcd = _pitch_circle_diameter(matched)
        error = abs(actual_pcd - expected_pcd) if actual_pcd is not None else None
        checks.append(
            {
                "value": "pitch_circle_diameter",
                "expected_mm": expected_pcd,
                "actual_mm": round(actual_pcd, 6) if actual_pcd is not None else None,
                "absolute_error_mm": round(error, 6) if error is not None else None,
                "tolerance_mm": pcd_tolerance,
                "passed": error is not None and error <= pcd_tolerance,
            }
        )

    return {
        "checks": checks,
        "matched_features": matched,
        "candidate_count_before_diameter_filter": len(axis_candidates),
    }


def _pitch_circle_diameter(holes: list[dict[str, Any]]) -> float | None:
    if len(holes) < 2:
        return None
    axis = holes[0].get("axis")
    if axis not in AXES or any(hole.get("axis") != axis for hole in holes):
        return None
    plane_axes = [candidate for candidate in AXES if candidate != axis]
    points = [
        tuple(float(hole["center_mm"][plane_axis]) for plane_axis in plane_axes)
        for hole in holes
    ]
    centroid = tuple(sum(values) / len(points) for values in zip(*points, strict=True))
    radii = [
        math.sqrt(sum((value - center) ** 2 for value, center in zip(point, centroid)))
        for point in points
    ]
    return 2 * sum(radii) / len(radii)


def legacy_dimension_constraint(
    target_dimensions_mm: list[float | None],
) -> dict[str, Any]:
    """Convert the original [x, y, z] target into the version-1 schema."""
    expected = {
        axis: target
        for axis, target in zip(AXES, target_dimensions_mm, strict=True)
        if target is not None
    }
    return {
        "id": "overall_dimensions",
        "type": "bbox_dimensions",
        "hard": True,
        "expected_mm": expected,
        "absolute_tolerance_mm": 0.5,
        "relative_tolerance": 0.01,
    }


def _axis_checks(
    expected_by_axis: dict[str, float],
    actual_by_axis: dict[str, float],
    constraint: dict[str, Any],
    *,
    value_label: str,
) -> list[dict[str, Any]]:
    absolute_tolerance = float(constraint.get("absolute_tolerance_mm", 0.5))
    relative_tolerance = float(constraint.get("relative_tolerance", 0.01))
    checks = []
    for axis in AXES:
        if axis not in expected_by_axis:
            continue
        expected = float(expected_by_axis[axis])
        actual = float(actual_by_axis[axis])
        error = abs(actual - expected)
        tolerance = max(absolute_tolerance, abs(expected) * relative_tolerance)
        checks.append(
            {
                "value": value_label,
                "axis": axis,
                "expected_mm": expected,
                "actual_mm": round(actual, 6),
                "absolute_error_mm": round(error, 6),
                "tolerance_mm": round(tolerance, 6),
                "passed": error <= tolerance,
            }
        )
    return checks


def _format_artifact_feedback(evaluation: dict[str, Any]) -> str:
    execution = evaluation["execution"]
    exports = evaluation["exports"]
    geometry = evaluation["geometry"]
    lines = ["CAD EXECUTION/ARTIFACT EVALUATION"]
    if not execution["final_success"]:
        lines.append("The CadQuery program or export process did not complete successfully.")
    if not exports["step_success"]:
        lines.append("STEP export is missing or empty.")
    if not exports["stl_success"]:
        lines.append("STL export is missing or empty.")
    if not geometry["geometry_valid"]:
        lines.append("The exported STL does not contain valid non-degenerate geometry.")
        if geometry.get("error"):
            lines.append(f"Geometry validation error: {geometry['error']}")
    lines.append("Return a complete corrected program that executes and exports valid STEP/STL geometry.")
    return "\n".join(lines)


def _format_constraint_check(check: dict[str, Any]) -> str:
    labels = []
    if check.get("value") is not None:
        labels.append(str(check["value"]))
    if check.get("axis") is not None:
        labels.append(f"axis={check['axis']}")
    if check.get("feature_index") is not None:
        labels.append(f"feature={check['feature_index']}")
    prefix = "Check " + ", ".join(labels) if labels else "Check"
    expected = check.get("expected_mm", check.get("expected"))
    actual = check.get("actual_mm", check.get("actual"))
    parts = [prefix]
    if expected is not None:
        parts.append(f"expected={expected}")
    if actual is not None:
        parts.append(f"actual={actual}")
    if check.get("tolerance_mm") is not None:
        parts.append(f"tolerance_mm={check['tolerance_mm']}")
    if check.get("diameter_match_tolerance_mm") is not None:
        parts.append(
            "diameter_match_tolerance_mm="
            f"{check['diameter_match_tolerance_mm']}"
        )
    if check.get("allowed_axes") is not None:
        parts.append(f"allowed_axes={check['allowed_axes']}")
    if check.get("expected_through") is not None:
        parts.append(f"expected_through={check['expected_through']}")
    return "; ".join(parts) + f"; passed={check.get('passed')}"


def _format_axis_values(values: dict[str, Any]) -> str:
    return ", ".join(f"{axis.upper()}={values[axis]}" for axis in AXES if axis in values)


def _read_stl_geometry(stl_path: Path) -> dict[str, Any]:
    try:
        import vtk

        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        mesh = reader.GetOutput()
        bounds = mesh.GetBounds()
        if bounds is None or len(bounds) != 6:
            return _empty_geometry("STL reader returned no bounds.")

        dimensions = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
        finite = all(math.isfinite(value) for value in bounds)
        valid = (
            mesh.GetNumberOfPoints() >= 4
            and mesh.GetNumberOfCells() >= 4
            and finite
            and all(value > 0.01 for value in dimensions)
        )
        return {
            "geometry_valid": valid,
            "validation_method": "non-empty STL mesh with finite, non-degenerate bounds",
            "mesh_points": mesh.GetNumberOfPoints(),
            "mesh_cells": mesh.GetNumberOfCells(),
            "bounds_min_mm": _axis_dict((bounds[0], bounds[2], bounds[4])),
            "bounds_max_mm": _axis_dict((bounds[1], bounds[3], bounds[5])),
            "dimensions_mm": _axis_dict(dimensions),
            "error": None,
        }
    except Exception as exc:
        return _empty_geometry(f"{type(exc).__name__}: {exc}")


def _empty_geometry(error: str | None = None) -> dict[str, Any]:
    return {
        "geometry_valid": False,
        "validation_method": "non-empty STL mesh with finite, non-degenerate bounds",
        "mesh_points": 0,
        "mesh_cells": 0,
        "bounds_min_mm": None,
        "bounds_max_mm": None,
        "dimensions_mm": None,
        "error": error,
    }


def _axis_dict(values: Iterable[float]) -> dict[str, float]:
    return {
        axis: round(float(value), 6)
        for axis, value in zip(AXES, values, strict=True)
    }


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None
