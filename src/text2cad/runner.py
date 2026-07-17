"""Run generated CadQuery code and export CAD artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CadRunResult:
    run_dir: Path
    code_path: Path
    step_path: Path
    stl_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class CadExecutionResult:
    run_dir: Path
    code_path: Path
    step_path: Path
    stl_path: Path
    metadata_path: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


RUNNER_SCRIPT = r'''
from pathlib import Path
import runpy
import sys

import cadquery as cq
from cadquery import exporters

code_path = Path(sys.argv[1])
step_path = Path(sys.argv[2])
stl_path = Path(sys.argv[3])

namespace = runpy.run_path(str(code_path))
result = namespace.get("result")
if result is None:
    for fallback_name in ("model", "part", "assembly"):
        if fallback_name in namespace:
            result = namespace[fallback_name]
            break

if result is None:
    raise ValueError("Generated code did not define `result`.")

export_result = result.toCompound() if isinstance(result, cq.Assembly) else result
exporters.export(export_result, str(step_path))
exporters.export(export_result, str(stl_path))
'''


def make_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    counter = 1
    while run_dir.exists():
        counter += 1
        run_dir = output_root / f"run_{timestamp}_{counter}"
    run_dir.mkdir(parents=True)
    return run_dir


def run_cadquery_code(
    code: str,
    *,
    description: str,
    output_root: Path = Path("outputs"),
    python_executable: str | None = None,
    timeout: int = 180,
) -> CadRunResult:
    """Write generated code, execute it with CadQuery, and export STEP/STL."""
    run_dir = make_run_dir(output_root).resolve()
    execution = execute_cadquery_code(
        code,
        description=description,
        run_dir=run_dir,
        python_executable=python_executable,
        timeout=timeout,
    )

    if not execution.ok:
        raise RuntimeError(
            "CadQuery execution failed. See "
            f"{execution.metadata_path}\n\nSTDERR:\n{execution.stderr}"
        )

    return CadRunResult(
        run_dir=execution.run_dir,
        code_path=execution.code_path,
        step_path=execution.step_path,
        stl_path=execution.stl_path,
        metadata_path=execution.metadata_path,
    )


def execute_cadquery_code(
    code: str,
    *,
    description: str,
    run_dir: Path,
    python_executable: str | None = None,
    timeout: int = 180,
    attempt: int = 0,
) -> CadExecutionResult:
    """Execute code inside an existing run directory and capture success/failure."""
    python_executable = python_executable or os.getenv("CADQUERY_PYTHON") or sys.executable
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if attempt == 0 else f"_repair_{attempt}"
    code_path = run_dir / f"generated_model{suffix}.py"
    helper_path = run_dir / "_run_cadquery_export.py"
    step_path = run_dir / f"model{suffix}.step"
    stl_path = run_dir / f"model{suffix}.stl"
    metadata_path = run_dir / f"run{suffix}.json"

    code_path.write_text(code.strip() + "\n", encoding="utf-8")
    helper_path.write_text(textwrap.dedent(RUNNER_SCRIPT).strip() + "\n", encoding="utf-8")

    timed_out = False
    try:
        completed = subprocess.run(
            [python_executable, str(helper_path), str(code_path), str(step_path), str(stl_path)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _timeout_output(exc.stdout)
        partial_stderr = _timeout_output(exc.stderr)
        stderr = (
            f"CadQuery execution timed out after {timeout} seconds."
            + (f"\n\nPartial stderr:\n{partial_stderr}" if partial_stderr else "")
        )

    metadata = {
        "description": description,
        "attempt": attempt,
        "python_executable": python_executable,
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "step_path": str(step_path),
        "stl_path": str(stl_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    return CadExecutionResult(
        run_dir=run_dir,
        code_path=code_path,
        step_path=step_path,
        stl_path=stl_path,
        metadata_path=metadata_path,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
