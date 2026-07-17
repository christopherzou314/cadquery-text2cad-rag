import sys
import tempfile
import unittest
from pathlib import Path

from src.text2cad.runner import execute_cadquery_code


class RunnerTests(unittest.TestCase):
    def test_assembly_exports_to_step_and_stl(self):
        code = """import cadquery as cq

result = cq.Assembly()
result.add(cq.Workplane("XY").box(20, 10, 5), name="base")
result.add(
    cq.Workplane("XY").cylinder(8, 2).translate((0, 0, 5)),
    name="pin",
)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = execute_cadquery_code(
                code,
                description="assembly export regression test",
                run_dir=Path(directory),
                python_executable=sys.executable,
                timeout=60,
            )

            self.assertTrue(result.ok, result.stderr)
            self.assertGreater(result.step_path.stat().st_size, 0)
            self.assertGreater(result.stl_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
