import tempfile
import unittest
from pathlib import Path

import cadquery as cq

from src.text2cad.evaluation import evaluate_cad_run, evaluate_constraint
from src.text2cad.step_features import extract_shape_features, read_step_features


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


def valid_stl_geometry():
    return {
        "geometry_valid": True,
        "dimensions_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
        "bounds_min_mm": {"x": 0.0, "y": 0.0, "z": 0.0},
        "bounds_max_mm": {"x": 1.0, "y": 1.0, "z": 1.0},
    }


class StepFeatureTests(unittest.TestCase):
    def test_extracts_four_through_holes_and_pitch_circle(self):
        radius = 23
        plate = (
            cq.Workplane("XY")
            .box(60, 60, 8)
            .faces(">Z")
            .workplane()
            .pushPoints([(radius, 0), (0, radius), (-radius, 0), (0, -radius)])
            .hole(5)
        )
        features = extract_shape_features(plate.val())
        result = evaluate_constraint(
            {
                "id": "bolt_circle",
                "type": "cylindrical_hole_pattern",
                "expected_count": 4,
                "diameter_mm": 5,
                "diameter_tolerance_mm": 0.1,
                "axis": "z",
                "through": True,
                "pitch_circle_diameter_mm": 46,
                "position_tolerance_mm": 0.1,
            },
            valid_stl_geometry(),
            features,
        )

        self.assertEqual(features["solid_count"], 1)
        self.assertEqual(features["cylindrical_hole_count"], 4)
        self.assertTrue(result["evaluated"])
        self.assertTrue(result["passed"])

    def test_blind_hole_fails_through_requirement(self):
        plate = (
            cq.Workplane("XY")
            .box(30, 30, 10)
            .faces(">Z")
            .workplane()
            .hole(6, 4)
        )
        features = extract_shape_features(plate.val())
        result = evaluate_constraint(
            {
                "id": "through_hole",
                "type": "cylindrical_hole_pattern",
                "expected_count": 1,
                "diameter_mm": 6,
                "axis": "z",
                "through": True,
            },
            valid_stl_geometry(),
            features,
        )

        self.assertFalse(features["cylindrical_holes"][0]["through"])
        self.assertFalse(result["passed"])

    def test_wrong_diameter_is_matched_then_fails_tolerance(self):
        plate = (
            cq.Workplane("XY")
            .box(30, 30, 8)
            .faces(">Z")
            .workplane()
            .hole(7)
        )
        features = extract_shape_features(plate.val())
        result = evaluate_constraint(
            {
                "id": "diameter",
                "type": "cylindrical_hole_pattern",
                "expected_count": 1,
                "diameter_mm": 6,
                "diameter_tolerance_mm": 0.3,
                "axis": "z",
                "through": True,
            },
            valid_stl_geometry(),
            features,
        )

        self.assertEqual(len(result["matched_features"]), 1)
        self.assertFalse(result["passed"])

    def test_external_cylinder_is_not_reported_as_a_hole(self):
        features = extract_shape_features(cq.Workplane("XY").cylinder(10, 5).val())
        self.assertEqual(features["cylindrical_hole_count"], 0)

    def test_step_round_trip_and_solid_count(self):
        compound = cq.Compound.makeCompound(
            [
                cq.Workplane("XY").box(10, 10, 10).val(),
                cq.Workplane("XY").transformed(offset=(20, 0, 0)).box(10, 10, 10).val(),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            step_path = Path(directory) / "two_solids.step"
            cq.exporters.export(compound, str(step_path))
            features = read_step_features(step_path)

        self.assertTrue(features["step_valid"])
        self.assertEqual(features["solid_count"], 2)

    def test_v2_can_fail_without_changing_v1_result(self):
        compound = cq.Compound.makeCompound(
            [
                cq.Workplane("XY").box(1, 1, 1).val(),
                cq.Workplane("XY").transformed(offset=(2, 0, 0)).box(1, 1, 1).val(),
            ]
        )
        constraints = [
            {
                "id": "dimensions",
                "type": "bbox_dimensions",
                "expected_mm": {"x": 1, "y": 1, "z": 1},
                "absolute_tolerance_mm": 0.01,
                "relative_tolerance": 0,
            },
            {
                "id": "single_solid",
                "type": "solid_count",
                "expected": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            step_path = root / "model.step"
            stl_path = root / "model.stl"
            cq.exporters.export(compound, str(step_path))
            stl_path.write_text(TETRAHEDRON_STL, encoding="ascii")
            evaluation = evaluate_cad_run(
                attempt_returncodes=[0],
                step_path=step_path,
                stl_path=stl_path,
                constraints=constraints,
                repair_budget=0,
            )

        self.assertTrue(evaluation["summary"]["task_success_v1"])
        self.assertFalse(evaluation["summary"]["task_success_v2"])


if __name__ == "__main__":
    unittest.main()
