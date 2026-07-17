"""Generate CadQuery code, execute it, and repair failures with traceback feedback."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .llm import generate_cadquery_code, repair_cadquery_code
from .prompts import mock_cadquery_code
from .rag import (
    FULL_LIBRARY_PATH,
    LIGHTWEIGHT_LIBRARY_PATH,
    format_reference_context,
    retrieve_references,
    write_retrieval_log,
)
from .runner import CadExecutionResult, execute_cadquery_code, make_run_dir


@dataclass(frozen=True)
class AgentRunResult:
    success: bool
    run_dir: Path
    attempts: list[CadExecutionResult]
    reference_ids: tuple[str, ...] = ()
    rag_mode: str = "off"

    @property
    def final_attempt(self) -> CadExecutionResult:
        return self.attempts[-1]


def generate_execute_repair(
    description: str,
    *,
    mock: bool = False,
    max_repairs: int = 2,
    output_root: Path = Path("outputs"),
    python_executable: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_timeout: int = 300,
    api_max_retries: int = 3,
    execution_timeout: int = 180,
    progress_callback: Callable[[str], None] | None = None,
    use_rag: bool = False,
    rag_mode: str | None = None,
    rag_top_k: int = 3,
) -> AgentRunResult:
    """Run the basic agent loop: generate, execute, repair from traceback."""
    python_executable = python_executable or os.getenv("CADQUERY_PYTHON") or sys.executable
    run_dir = make_run_dir(output_root).resolve()
    reference_context = None
    reference_ids: tuple[str, ...] = ()
    effective_rag_mode = rag_mode or ("full" if use_rag else "off")
    if effective_rag_mode not in {"off", "lightweight", "full"}:
        raise ValueError(f"Unknown RAG mode: {effective_rag_mode}")

    if effective_rag_mode != "off":
        library_path = (
            LIGHTWEIGHT_LIBRARY_PATH
            if effective_rag_mode == "lightweight"
            else FULL_LIBRARY_PATH
        )
        _report_progress(
            progress_callback,
            f"Retrieving CadQuery references ({effective_rag_mode} RAG)...",
        )
        matches = retrieve_references(
            description,
            top_k=rag_top_k,
            library_path=library_path,
        )
        reference_context = format_reference_context(matches)
        reference_ids = tuple(match.id for match in matches)
        write_retrieval_log(run_dir / "retrieval.json", query=description, matches=matches)
        _report_progress(
            progress_callback,
            "Retrieved references: " + ", ".join(match.title for match in matches),
        )

    if mock:
        _report_progress(progress_callback, "Creating mock CadQuery code...")
        code = mock_cadquery_code(description)
    else:
        _report_progress(progress_callback, "Calling LLM to generate CadQuery code...")
        code = generate_cadquery_code(
            description,
            reference_context=reference_context,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout,
            max_retries=api_max_retries,
            progress_callback=progress_callback,
        )

    attempts: list[CadExecutionResult] = []
    max_attempts = max_repairs + 1

    for attempt_index in range(max_attempts):
        _report_progress(
            progress_callback,
            f"Executing CadQuery code (attempt {attempt_index + 1}/{max_attempts})...",
        )
        execution = execute_cadquery_code(
            code,
            description=description,
            run_dir=run_dir,
            python_executable=python_executable,
            timeout=execution_timeout,
            attempt=attempt_index,
        )
        attempts.append(execution)
        if execution.ok:
            _report_progress(progress_callback, "CadQuery execution succeeded.")
            _write_agent_summary(
                run_dir,
                description,
                attempts,
                effective_rag_mode,
                reference_ids,
            )
            return AgentRunResult(
                success=True,
                run_dir=run_dir,
                attempts=attempts,
                reference_ids=reference_ids,
                rag_mode=effective_rag_mode,
            )

        if mock or attempt_index >= max_repairs:
            break

        _report_progress(
            progress_callback,
            f"CadQuery failed. Asking LLM for repair {attempt_index + 1}/{max_repairs}...",
        )
        code = repair_cadquery_code(
            description,
            code,
            execution.stderr,
            reference_context=reference_context,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout,
            max_retries=api_max_retries,
            progress_callback=progress_callback,
        )

    _write_agent_summary(
        run_dir,
        description,
        attempts,
        effective_rag_mode,
        reference_ids,
    )
    return AgentRunResult(
        success=False,
        run_dir=run_dir,
        attempts=attempts,
        reference_ids=reference_ids,
        rag_mode=effective_rag_mode,
    )


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)


def _write_agent_summary(
    run_dir: Path,
    description: str,
    attempts: list[CadExecutionResult],
    rag_mode: str,
    reference_ids: tuple[str, ...],
) -> None:
    summary = {
        "description": description,
        "success": attempts[-1].ok,
        "attempt_count": len(attempts),
        "generation_setting": (
            "baseline" if rag_mode == "off" else f"reference_assisted_{rag_mode}"
        ),
        "rag_mode": rag_mode,
        "reference_ids": list(reference_ids),
        "attempts": [
            {
                "attempt": index,
                "returncode": attempt.returncode,
                "code_path": str(attempt.code_path),
                "step_path": str(attempt.step_path),
                "stl_path": str(attempt.stl_path),
                "metadata_path": str(attempt.metadata_path),
            }
            for index, attempt in enumerate(attempts)
        ],
    }
    (run_dir / "agent_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
