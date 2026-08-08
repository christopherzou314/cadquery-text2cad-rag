import tempfile
import unittest
from pathlib import Path

import cadquery as cq

from src.text2cad.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    evaluate_cad_run,
    evaluate_constraint,
    flatten_evaluation,
    format_repair_feedback,
    layered_pass_metrics,
    pass_by_repair_budget,
    repair_trigger,
)


TETRAHEDRON_STL = """solid tetrahedron
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 0 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 0 1 0
    endloop
  endfacet
  facet normal 1 1 1
    outer loop
      vertex 1 0 0
      vertex 0 1 0
      vertex 0 0 1
    endloop
  endfacet
endsolid tetrahedron
"""


class EvaluationTests(unittest.TestCase):
    def test_evaluation_schema_version_is_frozen_at_v2(self):
        self.assertEqual(EVALUATION_SCHEMA_VERSION, "2.0")

    def test_constraint_uses_larger_of_absolute_and_relative_tolerance(self):
        geometry = {
            "geometry_valid": True,
            "dimensions_mm": {"x": 101.0, "y": 10.0, "z": 10.0},
            "bounds_min_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
            "bounds_max_mm": {"x": 101.0, "y": 10.0, "z": 10.0},
        }
        result = evaluate_constraint(
            {
                "id": "length",
                "type": "bbox_dimensions",
                "expected_mm": {"x": 100},
                "absolute_tolerance_mm": 0.5,
                "relative_tolerance": 0.01,
            },
            geometry,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["checks"][0]["tolerance_mm"], 1.0)

    def test_run_evaluation_tracks_repairs_exports_and_constraints(self):
        constraints = [
            {
                "id": "dimensions",
                "type": "bbox_dimensions",
                "expected_mm": {"x": 1, "y": 1, "z": 1},
                "absolute_tolerance_mm": 0.01,
                "relative_tolerance": 0,
            },
            {
                "id": "position",
                "type": "bbox_bounds",
                "expected_min_mm": {"z": 0},
                "expected_max_mm": {"z": 1},
                "absolute_tolerance_mm": 0.01,
                "relative_tolerance": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_path = root / "model.step"
            stl_path = root / "model.stl"
            cq.exporters.export(cq.Workplane("XY").box(1, 1, 1).val(), str(step_path))
            stl_path.write_text(TETRAHEDRON_STL, encoding="ascii")

            evaluation = evaluate_cad_run(
                attempt_returncodes=[1, 0],
                step_path=step_path,
                stl_path=stl_path,
                constraints=constraints,
                repair_budget=1,
            )

        self.assertFalse(evaluation["execution"]["first_generation_success"])
        self.assertTrue(evaluation["execution"]["success_within_repair_budget"])
        self.assertEqual(evaluation["execution"]["repairs_used"], 1)
        self.assertTrue(evaluation["exports"]["step_success"])
        self.assertTrue(evaluation["exports"]["stl_success"])
        self.assertTrue(evaluation["geometry"]["geometry_valid"])
        self.assertEqual(evaluation["summary"]["constraint_groups_passed"], 2)
        self.assertEqual(evaluation["summary"]["constraint_groups_total"], 2)
        self.assertTrue(evaluation["summary"]["task_success_v1"])

    def test_failed_hard_constraint_fails_task(self):
        geometry = {
            "geometry_valid": True,
            "dimensions_mm": {"x": 10.0, "y": 10.0, "z": 10.0},
            "bounds_min_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
            "bounds_max_mm": {"x": 10.0, "y": 10.0, "z": 10.0},
        }
        result = evaluate_constraint(
            {
                "id": "wrong_length",
                "type": "bbox_dimensions",
                "hard": True,
                "expected_mm": {"x": 20},
                "absolute_tolerance_mm": 0.5,
                "relative_tolerance": 0.01,
            },
            geometry,
        )

        self.assertTrue(result["evaluated"])
        self.assertFalse(result["passed"])

    def test_failed_constraint_produces_structured_repair_feedback(self):
        geometry = {
            "geometry_valid": True,
            "dimensions_mm": {"x": 12.0, "y": 10.0, "z": 10.0},
            "bounds_min_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
            "bounds_max_mm": {"x": 12.0, "y": 10.0, "z": 10.0},
        }
        constraint = {
            "id": "length",
            "type": "bbox_dimensions",
            "hard": True,
            "expected_mm": {"x": 10},
            "absolute_tolerance_mm": 0.5,
            "relative_tolerance": 0,
        }
        result = evaluate_constraint(constraint, geometry)
        evaluation = {
            "execution": {"final_success": True},
            "exports": {"step_success": True, "stl_success": True},
            "geometry": geometry,
            "constraints": [result],
            "summary": {
                "artifact_success": True,
                "hard_constraints_total": 1,
                "all_registered_hard_constraints_satisfied": False,
            },
        }

        self.assertEqual(repair_trigger(evaluation), "constraint")
        self.assertEqual(
            layered_pass_metrics(evaluation),
            {
                "execution_pass": True,
                "constraint_pass": False,
                "end_to_end_pass": False,
                "constraint_eligible": True,
            },
        )
        feedback = format_repair_feedback(evaluation, "constraint")
        self.assertIn("FAILED constraint: length", feedback)
        self.assertIn("expected=10.0", feedback)
        self.assertIn("actual=12.0", feedback)

    def test_pass_curve_carries_success_forward_without_extra_repairs(self):
        successful = {
            "execution": {"final_success": True},
            "exports": {"step_success": True, "stl_success": True},
            "summary": {
                "artifact_success": True,
                "hard_constraints_total": 1,
                "all_registered_hard_constraints_satisfied": True,
            },
        }

        curve = pass_by_repair_budget([successful], 2)

        self.assertEqual([item["end_to_end_pass"] for item in curve], [True] * 3)

    def test_missing_execution_and_artifacts_are_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = evaluate_cad_run(
                attempt_returncodes=[],
                step_path=root / "missing.step",
                stl_path=root / "missing.stl",
                constraints=[
                    {
                        "id": "dimensions",
                        "type": "bbox_dimensions",
                        "expected_mm": {"x": 10},
                    }
                ],
                repair_budget=0,
            )

        self.assertFalse(evaluation["execution"]["first_generation_success"])
        self.assertFalse(evaluation["exports"]["step_success"])
        self.assertFalse(evaluation["geometry"]["geometry_valid"])
        self.assertEqual(evaluation["summary"]["constraint_groups_registered"], 1)
        self.assertEqual(evaluation["summary"]["constraint_groups_evaluated"], 0)
        self.assertEqual(evaluation["summary"]["constraint_evaluation_coverage"], 0)
        self.assertFalse(evaluation["summary"]["task_success_v1"])
        flattened = flatten_evaluation(evaluation)
        self.assertEqual(flattened["failed_constraint_ids"], [])
        self.assertEqual(flattened["unevaluated_constraint_ids"], ["dimensions"])


if __name__ == "__main__":
    unittest.main()
