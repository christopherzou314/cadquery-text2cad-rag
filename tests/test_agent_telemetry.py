import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.text2cad.agent import generate_clarification, generate_execute_repair
from src.text2cad.llm import LLMError


class AgentTelemetryTests(unittest.TestCase):
    @patch(
        "src.text2cad.agent.generate_clarification_response",
        return_value="The radius and diameter conflict. Which value is mandatory?",
    )
    def test_clarification_run_skips_cad_and_writes_pending_behavior_log(
        self, _generate
    ):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_clarification(
                "A cylinder has radius 30 mm and diameter 40 mm.",
                output_root=Path(directory),
                seed=11,
                send_seed=True,
                run_context={"expected_behavior": "clarify_before_generation"},
            )
            summary = json.loads(
                (result.run_dir / "agent_run.json").read_text(encoding="utf-8")
            )
            saved_response = result.response_path.read_text(encoding="utf-8").strip()

        self.assertEqual(saved_response, result.response)
        self.assertEqual(summary["task_type"], "clarification")
        self.assertEqual(summary["attempt_count"], 0)
        self.assertEqual(summary["max_repairs"], 0)
        self.assertEqual(summary["repairs_used"], 0)
        self.assertIsNone(summary["execution_pass_within_repair_budget"])
        self.assertEqual(summary["clarification_evaluation_status"], "pending")
        self.assertIsNone(summary["clarification_success"])
        self.assertEqual(summary["timing"]["cad_execution_seconds"], 0.0)
        self.assertEqual(summary["attempts"], [])

    def test_mock_run_writes_execution_and_end_to_end_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            result = generate_execute_repair(
                "a small box",
                mock=True,
                max_repairs=0,
                output_root=Path(directory),
                python_executable=sys.executable,
                seed=7,
                run_context={"prompt_id": "TEST-1", "condition": "mock"},
            )
            summary = json.loads(
                (result.run_dir / "agent_run.json").read_text(encoding="utf-8")
            )

        self.assertTrue(result.success)
        self.assertEqual(summary["schema_version"], "3.0")
        self.assertEqual(summary["seed_requested"], 7)
        self.assertFalse(summary["seed_sent_to_provider"])
        self.assertEqual(summary["run_context"]["prompt_id"], "TEST-1")
        self.assertEqual(summary["token_usage"]["call_count"], 0)
        self.assertIsNone(summary["token_usage"]["total_tokens"])
        self.assertGreater(summary["timing"]["cad_execution_seconds"], 0)
        self.assertGreaterEqual(
            summary["timing"]["end_to_end_seconds"],
            summary["timing"]["cad_execution_seconds"],
        )
        self.assertEqual(summary["attempts"][0]["returncode"], 0)
        self.assertTrue(summary["execution_pass_within_repair_budget"])
        self.assertIsNone(summary["constraint_pass_within_repair_budget"])
        self.assertTrue(summary["end_to_end_pass_within_repair_budget"])

    @patch(
        "src.text2cad.agent.repair_cadquery_code",
        side_effect=[
            "import cadquery as cq\nresult = cq.Workplane('XY').box(20, 10, 10)",
            "import cadquery as cq\nresult = cq.Workplane('XY').box(10, 10, 10)",
        ],
    )
    @patch(
        "src.text2cad.agent.generate_cadquery_code",
        return_value="raise RuntimeError('attempt zero failed')",
    )
    def test_unified_budget_repairs_execution_then_constraint(
        self, _generate, repair
    ):
        constraints = [
            {
                "id": "overall_dimensions",
                "type": "bbox_dimensions",
                "hard": True,
                "expected_mm": {"x": 10, "y": 10, "z": 10},
                "absolute_tolerance_mm": 0.1,
                "relative_tolerance": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = generate_execute_repair(
                "a 10 mm cube",
                max_repairs=2,
                constraints=constraints,
                output_root=Path(directory),
                python_executable=sys.executable,
                api_key="test-key",
            )
            first_feedback = (result.run_dir / "repair_feedback_1.txt").read_text(
                encoding="utf-8"
            )
            second_feedback = (result.run_dir / "repair_feedback_2.txt").read_text(
                encoding="utf-8"
            )

        self.assertTrue(result.success)
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(result.repair_types, ("execution", "constraint"))
        self.assertEqual(result.telemetry["repairs_used"], 2)
        self.assertEqual(result.telemetry["execution_repairs"], 1)
        self.assertEqual(result.telemetry["constraint_repairs"], 1)
        self.assertEqual(
            [item["end_to_end_pass"] for item in result.telemetry["pass_by_repair_budget"]],
            [False, False, True],
        )
        self.assertEqual(repair.call_args_list[0].kwargs["repair_type"], "execution")
        self.assertEqual(repair.call_args_list[1].kwargs["repair_type"], "constraint")
        self.assertIn("attempt zero failed", first_feedback)
        self.assertIn("FAILED constraint: overall_dimensions", second_feedback)
        self.assertIn("expected=10.0", second_feedback)
        self.assertIn("actual=20.0", second_feedback)

    @patch(
        "src.text2cad.agent.generate_cadquery_code",
        side_effect=LLMError("provider unavailable"),
    )
    def test_initial_generation_failure_still_writes_agent_log(self, _generate):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(LLMError):
                generate_execute_repair(
                    "a small box",
                    output_root=root,
                    api_key="test-key",
                    max_repairs=0,
                )
            run_dir = next(root.glob("run_*"))
            summary = json.loads(
                (run_dir / "agent_run.json").read_text(encoding="utf-8")
            )

        self.assertFalse(summary["success"])
        self.assertEqual(summary["failure_stage"], "initial_generation")
        self.assertEqual(summary["attempt_count"], 0)
        self.assertIn("provider unavailable", summary["error"])
