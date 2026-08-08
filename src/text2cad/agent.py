"""Generate CadQuery code and repair execution or hard-constraint failures."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import (
    evaluate_cad_run,
    format_repair_feedback,
    layered_pass_metrics,
    pass_by_repair_budget,
    repair_trigger,
)
from .llm import (
    generate_cadquery_code,
    generate_clarification_response,
    repair_cadquery_code,
)
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
    telemetry: dict[str, Any] = field(default_factory=dict)
    attempt_evaluations: list[dict[str, Any]] = field(default_factory=list)
    repair_types: tuple[str, ...] = ()
    repair_events: tuple[dict[str, Any], ...] = ()

    @property
    def final_attempt(self) -> CadExecutionResult:
        return self.attempts[-1]

    @property
    def final_evaluation(self) -> dict[str, Any]:
        return self.attempt_evaluations[-1]


@dataclass(frozen=True)
class ClarificationRunResult:
    run_dir: Path
    response: str
    response_path: Path
    reference_ids: tuple[str, ...] = ()
    rag_mode: str = "off"
    telemetry: dict[str, Any] = field(default_factory=dict)


def generate_clarification(
    description: str,
    *,
    output_root: Path = Path("outputs"),
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_timeout: int = 300,
    api_max_retries: int = 3,
    progress_callback: Callable[[str], None] | None = None,
    rag_mode: str = "off",
    rag_top_k: int = 3,
    seed: int | None = None,
    send_seed: bool = False,
    run_context: dict[str, Any] | None = None,
) -> ClarificationRunResult:
    """Request and log clarification text without executing or repairing CAD code."""
    if rag_mode not in {"off", "lightweight", "full"}:
        raise ValueError(f"Unknown RAG mode: {rag_mode}")
    run_dir = make_run_dir(output_root).resolve()
    run_started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    llm_calls: list[dict[str, Any]] = []
    reference_ids: tuple[str, ...] = ()
    rag_retrieval_seconds = 0.0

    try:
        reference_context, reference_ids, rag_retrieval_seconds = (
            _retrieve_reference_context(
                description=description,
                rag_mode=rag_mode,
                rag_top_k=rag_top_k,
                run_dir=run_dir,
                progress_callback=progress_callback,
            )
        )
    except Exception as exc:
        _write_clarification_summary(
            run_dir=run_dir,
            description=description,
            response=None,
            response_path=None,
            rag_mode=rag_mode,
            reference_ids=reference_ids,
            seed=seed,
            send_seed=send_seed,
            llm_calls=llm_calls,
            started_at_utc=started_at_utc,
            run_started=run_started,
            rag_retrieval_seconds=rag_retrieval_seconds,
            run_context=run_context,
            failure_stage="rag_retrieval",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    try:
        _report_progress(progress_callback, "Calling LLM for clarification response...")
        response = generate_clarification_response(
            description,
            reference_context=reference_context,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=api_timeout,
            max_retries=api_max_retries,
            progress_callback=progress_callback,
            seed=seed,
            send_seed=send_seed,
            metrics_callback=_llm_metrics_collector(
                llm_calls, call_kind="clarification", agent_attempt=0
            ),
        )
        if not response.strip():
            raise ValueError("LLM returned an empty clarification response")
    except Exception as exc:
        _write_clarification_summary(
            run_dir=run_dir,
            description=description,
            response=None,
            response_path=None,
            rag_mode=rag_mode,
            reference_ids=reference_ids,
            seed=seed,
            send_seed=send_seed,
            llm_calls=llm_calls,
            started_at_utc=started_at_utc,
            run_started=run_started,
            rag_retrieval_seconds=rag_retrieval_seconds,
            run_context=run_context,
            failure_stage="clarification_generation",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    response_path = run_dir / "clarification_response.txt"
    response_path.write_text(response.rstrip() + "\n", encoding="utf-8")
    telemetry = _write_clarification_summary(
        run_dir=run_dir,
        description=description,
        response=response,
        response_path=response_path,
        rag_mode=rag_mode,
        reference_ids=reference_ids,
        seed=seed,
        send_seed=send_seed,
        llm_calls=llm_calls,
        started_at_utc=started_at_utc,
        run_started=run_started,
        rag_retrieval_seconds=rag_retrieval_seconds,
        run_context=run_context,
    )
    _report_progress(
        progress_callback,
        "Clarification response saved; CAD execution and repair were skipped.",
    )
    return ClarificationRunResult(
        run_dir=run_dir,
        response=response,
        response_path=response_path,
        reference_ids=reference_ids,
        rag_mode=rag_mode,
        telemetry=telemetry,
    )


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
    seed: int | None = None,
    send_seed: bool = False,
    run_context: dict[str, Any] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    initial_code: str | None = None,
) -> AgentRunResult:
    """Generate or reuse code, then execute, evaluate, and repair within one budget."""
    if max_repairs < 0:
        raise ValueError("max_repairs cannot be negative")
    if mock and initial_code is not None:
        raise ValueError("mock and initial_code cannot be used together")
    registered_constraints = list(constraints or [])
    python_executable = python_executable or os.getenv("CADQUERY_PYTHON") or sys.executable
    run_dir = make_run_dir(output_root).resolve()
    run_started = time.monotonic()
    started_at_utc = datetime.now(timezone.utc).isoformat()
    reference_context = None
    reference_ids: tuple[str, ...] = ()
    llm_calls: list[dict[str, Any]] = []
    rag_retrieval_seconds = 0.0
    effective_rag_mode = rag_mode or ("full" if use_rag else "off")
    if effective_rag_mode not in {"off", "lightweight", "full"}:
        raise ValueError(f"Unknown RAG mode: {effective_rag_mode}")

    if effective_rag_mode != "off":
        retrieval_started = time.monotonic()
        try:
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
            write_retrieval_log(
                run_dir / "retrieval.json", query=description, matches=matches
            )
            _report_progress(
                progress_callback,
                "Retrieved references: " + ", ".join(match.title for match in matches),
            )
            rag_retrieval_seconds = round(time.monotonic() - retrieval_started, 6)
        except Exception as exc:
            rag_retrieval_seconds = round(time.monotonic() - retrieval_started, 6)
            _write_agent_summary(
                run_dir=run_dir,
                description=description,
                attempts=[],
                rag_mode=effective_rag_mode,
                reference_ids=reference_ids,
                success=False,
                max_repairs=max_repairs,
                seed=seed,
                send_seed=send_seed,
                llm_calls=llm_calls,
                started_at_utc=started_at_utc,
                run_started=run_started,
                rag_retrieval_seconds=rag_retrieval_seconds,
                run_context=run_context,
                failure_stage="rag_retrieval",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

    try:
        if initial_code is not None:
            _report_progress(progress_callback, "Loading existing attempt 0 code...")
            code = initial_code
        elif mock:
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
                seed=seed,
                send_seed=send_seed,
                metrics_callback=_llm_metrics_collector(
                    llm_calls, call_kind="generation", agent_attempt=0
                ),
            )
    except Exception as exc:
        _write_agent_summary(
            run_dir=run_dir,
            description=description,
            attempts=[],
            rag_mode=effective_rag_mode,
            reference_ids=reference_ids,
            success=False,
            max_repairs=max_repairs,
            seed=seed,
            send_seed=send_seed,
            llm_calls=llm_calls,
            started_at_utc=started_at_utc,
            run_started=run_started,
            rag_retrieval_seconds=rag_retrieval_seconds,
            run_context=run_context,
            failure_stage="initial_generation",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    attempts: list[CadExecutionResult] = []
    attempt_evaluations: list[dict[str, Any]] = []
    repair_types: list[str] = []
    repair_events: list[dict[str, Any]] = []
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
        evaluation = evaluate_cad_run(
            attempt_returncodes=[attempt.returncode for attempt in attempts],
            step_path=execution.step_path,
            stl_path=execution.stl_path,
            constraints=registered_constraints,
            repair_budget=max_repairs,
        )
        attempt_evaluations.append(evaluation)
        evaluation_path = run_dir / f"evaluation_attempt_{attempt_index}.json"
        evaluation_path.write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        trigger = repair_trigger(evaluation)
        if trigger is None:
            metrics = layered_pass_metrics(evaluation)
            _report_progress(
                progress_callback,
                "CadQuery execution, exports, geometry, and registered constraints passed."
                if metrics["constraint_eligible"]
                else "CadQuery execution, exports, and geometry passed.",
            )
            telemetry = _write_agent_summary(
                run_dir=run_dir,
                description=description,
                attempts=attempts,
                rag_mode=effective_rag_mode,
                reference_ids=reference_ids,
                success=True,
                max_repairs=max_repairs,
                seed=seed,
                send_seed=send_seed,
                llm_calls=llm_calls,
                started_at_utc=started_at_utc,
                run_started=run_started,
                rag_retrieval_seconds=rag_retrieval_seconds,
                run_context=run_context,
                attempt_evaluations=attempt_evaluations,
                repair_types=repair_types,
                repair_events=repair_events,
                initial_code_provided=initial_code is not None,
            )
            return AgentRunResult(
                success=True,
                run_dir=run_dir,
                attempts=attempts,
                reference_ids=reference_ids,
                rag_mode=effective_rag_mode,
                telemetry=telemetry,
                attempt_evaluations=attempt_evaluations,
                repair_types=tuple(repair_types),
                repair_events=tuple(repair_events),
            )

        if mock or attempt_index >= max_repairs:
            break

        if trigger == "execution" and execution.stderr:
            feedback = execution.stderr
        else:
            feedback = format_repair_feedback(evaluation, trigger)
        _report_progress(
            progress_callback,
            f"{trigger.title()} failure. Asking LLM for repair "
            f"{attempt_index + 1}/{max_repairs}...",
        )
        feedback_path = run_dir / f"repair_feedback_{attempt_index + 1}.txt"
        feedback_path.write_text(feedback.rstrip() + "\n", encoding="utf-8")
        repair_event = {
            "repair_attempt": attempt_index + 1,
            "type": trigger,
            "feedback_path": str(feedback_path),
            "code_generated": False,
        }
        repair_events.append(repair_event)
        try:
            code = repair_cadquery_code(
                description,
                code,
                feedback,
                reference_context=reference_context,
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=api_timeout,
                max_retries=api_max_retries,
                progress_callback=progress_callback,
                seed=seed,
                send_seed=send_seed,
                metrics_callback=_llm_metrics_collector(
                    llm_calls,
                    call_kind=f"repair_{trigger}",
                    agent_attempt=attempt_index + 1,
                ),
                repair_type=trigger,
            )
            repair_types.append(trigger)
            repair_event["code_generated"] = True
        except Exception as exc:
            _write_agent_summary(
                run_dir=run_dir,
                description=description,
                attempts=attempts,
                rag_mode=effective_rag_mode,
                reference_ids=reference_ids,
                success=False,
                max_repairs=max_repairs,
                seed=seed,
                send_seed=send_seed,
                llm_calls=llm_calls,
                started_at_utc=started_at_utc,
                run_started=run_started,
                rag_retrieval_seconds=rag_retrieval_seconds,
                run_context=run_context,
                failure_stage=f"repair_generation_{attempt_index + 1}",
                error=f"{type(exc).__name__}: {exc}",
                attempt_evaluations=attempt_evaluations,
                repair_types=repair_types,
                repair_events=repair_events,
                initial_code_provided=initial_code is not None,
            )
            raise

    final_trigger = repair_trigger(attempt_evaluations[-1]) if attempt_evaluations else None
    telemetry = _write_agent_summary(
        run_dir=run_dir,
        description=description,
        attempts=attempts,
        rag_mode=effective_rag_mode,
        reference_ids=reference_ids,
        success=False,
        max_repairs=max_repairs,
        seed=seed,
        send_seed=send_seed,
        llm_calls=llm_calls,
        started_at_utc=started_at_utc,
        run_started=run_started,
        rag_retrieval_seconds=rag_retrieval_seconds,
        run_context=run_context,
        failure_stage=(
            "constraint_evaluation" if final_trigger == "constraint" else "cadquery_execution"
        ),
        error=(
            format_repair_feedback(attempt_evaluations[-1], final_trigger)
            if final_trigger and attempt_evaluations
            else (attempts[-1].stderr if attempts else None)
        ),
        attempt_evaluations=attempt_evaluations,
        repair_types=repair_types,
        repair_events=repair_events,
        initial_code_provided=initial_code is not None,
    )
    return AgentRunResult(
        success=False,
        run_dir=run_dir,
        attempts=attempts,
        reference_ids=reference_ids,
        rag_mode=effective_rag_mode,
        telemetry=telemetry,
        attempt_evaluations=attempt_evaluations,
        repair_types=tuple(repair_types),
        repair_events=tuple(repair_events),
    )


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)


def _retrieve_reference_context(
    *,
    description: str,
    rag_mode: str,
    rag_top_k: int,
    run_dir: Path,
    progress_callback: Callable[[str], None] | None,
) -> tuple[str | None, tuple[str, ...], float]:
    if rag_mode == "off":
        return None, (), 0.0

    retrieval_started = time.monotonic()
    library_path = (
        LIGHTWEIGHT_LIBRARY_PATH if rag_mode == "lightweight" else FULL_LIBRARY_PATH
    )
    _report_progress(
        progress_callback,
        f"Retrieving CadQuery references ({rag_mode} RAG)...",
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
    return (
        reference_context,
        reference_ids,
        round(time.monotonic() - retrieval_started, 6),
    )


def _write_clarification_summary(
    *,
    run_dir: Path,
    description: str,
    response: str | None,
    response_path: Path | None,
    rag_mode: str,
    reference_ids: tuple[str, ...],
    seed: int | None,
    send_seed: bool,
    llm_calls: list[dict[str, Any]],
    started_at_utc: str,
    run_started: float,
    rag_retrieval_seconds: float,
    run_context: dict[str, Any] | None,
    failure_stage: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    totals = _llm_totals(llm_calls)
    summary = {
        "schema_version": "3.0",
        "task_type": "clarification",
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "success": response is not None,
        "failure_stage": failure_stage,
        "error": error,
        "attempt_count": 0,
        "response_count": int(response is not None),
        "max_repairs": 0,
        "total_repair_budget": 0,
        "repairs_used": 0,
        "execution_repairs": 0,
        "constraint_repairs": 0,
        "execution_pass_within_repair_budget": None,
        "constraint_pass_within_repair_budget": None,
        "end_to_end_pass_within_repair_budget": None,
        "pass_by_repair_budget": [],
        "clarification_evaluation_status": "pending" if response is not None else "unavailable",
        "clarification_success": None,
        "response_path": str(response_path) if response_path else None,
        "response_characters": len(response) if response is not None else 0,
        "generation_setting": (
            "baseline" if rag_mode == "off" else f"reference_assisted_{rag_mode}"
        ),
        "rag_mode": rag_mode,
        "reference_ids": list(reference_ids),
        "seed_requested": seed,
        "seed_sent_to_provider": seed is not None and send_seed,
        "run_context": run_context or {},
        "timing": {
            "rag_retrieval_seconds": rag_retrieval_seconds,
            "initial_generation_seconds": totals["initial_generation_seconds"],
            "repair_generation_seconds": 0.0,
            "llm_seconds": totals["llm_seconds"],
            "cad_execution_seconds": 0.0,
            "end_to_end_seconds": round(time.monotonic() - run_started, 6),
        },
        "token_usage": totals["token_usage"],
        "cost": totals["cost"],
        "llm_calls": llm_calls,
        "attempts": [],
    }
    (run_dir / "agent_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _write_agent_summary(
    *,
    run_dir: Path,
    description: str,
    attempts: list[CadExecutionResult],
    rag_mode: str,
    reference_ids: tuple[str, ...],
    success: bool,
    max_repairs: int,
    seed: int | None,
    send_seed: bool,
    llm_calls: list[dict[str, Any]],
    started_at_utc: str,
    run_started: float,
    rag_retrieval_seconds: float,
    run_context: dict[str, Any] | None,
    failure_stage: str | None = None,
    error: str | None = None,
    attempt_evaluations: list[dict[str, Any]] | None = None,
    repair_types: list[str] | tuple[str, ...] = (),
    repair_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    initial_code_provided: bool = False,
) -> dict[str, Any]:
    attempt_evaluations = attempt_evaluations or []
    totals = _llm_totals(llm_calls)
    execution_seconds = round(sum(attempt.duration_seconds for attempt in attempts), 6)
    pass_curve = pass_by_repair_budget(attempt_evaluations, max_repairs)
    final_pass = pass_curve[-1]
    execution_repairs = sum(kind == "execution" for kind in repair_types)
    constraint_repairs = sum(kind == "constraint" for kind in repair_types)
    summary = {
        "schema_version": "3.0",
        "started_at_utc": started_at_utc,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": description,
        "success": success,
        "failure_stage": failure_stage,
        "error": error,
        "attempt_count": len(attempts),
        "max_repairs": max_repairs,
        "total_repair_budget": max_repairs,
        "repairs_used": len(repair_types),
        "execution_repairs": execution_repairs,
        "constraint_repairs": constraint_repairs,
        "repair_types": list(repair_types),
        "repair_events": list(repair_events),
        "initial_code_source": "existing" if initial_code_provided else "generated",
        "first_generation_execution_success": bool(attempts) and attempts[0].ok,
        "success_within_repair_budget": any(attempt.ok for attempt in attempts),
        "execution_pass_within_repair_budget": final_pass["execution_pass"],
        "constraint_pass_within_repair_budget": final_pass["constraint_pass"],
        "end_to_end_pass_within_repair_budget": final_pass["end_to_end_pass"],
        "pass_by_repair_budget": pass_curve,
        "generation_setting": (
            "baseline" if rag_mode == "off" else f"reference_assisted_{rag_mode}"
        ),
        "rag_mode": rag_mode,
        "reference_ids": list(reference_ids),
        "seed_requested": seed,
        "seed_sent_to_provider": seed is not None and send_seed,
        "run_context": run_context or {},
        "timing": {
            "rag_retrieval_seconds": rag_retrieval_seconds,
            "initial_generation_seconds": totals["initial_generation_seconds"],
            "repair_generation_seconds": totals["repair_generation_seconds"],
            "llm_seconds": totals["llm_seconds"],
            "cad_execution_seconds": execution_seconds,
            "end_to_end_seconds": round(time.monotonic() - run_started, 6),
        },
        "token_usage": totals["token_usage"],
        "cost": totals["cost"],
        "llm_calls": llm_calls,
        "attempts": [
            {
                "attempt": index,
                "kind": (
                    "source_replay"
                    if index == 0 and initial_code_provided
                    else ("initial" if index == 0 else "repair")
                ),
                "repair_trigger": repair_types[index - 1] if index > 0 else None,
                "returncode": attempt.returncode,
                "execution_seconds": attempt.duration_seconds,
                "code_path": str(attempt.code_path),
                "step_path": str(attempt.step_path),
                "stl_path": str(attempt.stl_path),
                "metadata_path": str(attempt.metadata_path),
                "evaluation_path": str(run_dir / f"evaluation_attempt_{index}.json"),
                "pass": (
                    layered_pass_metrics(attempt_evaluations[index])
                    if index < len(attempt_evaluations)
                    else None
                ),
            }
            for index, attempt in enumerate(attempts)
        ],
    }
    (run_dir / "agent_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _llm_metrics_collector(
    calls: list[dict[str, Any]],
    *,
    call_kind: str,
    agent_attempt: int,
) -> Callable[[dict[str, Any]], None]:
    def collect(metrics: dict[str, Any]) -> None:
        calls.append(
            {
                "call_kind": call_kind,
                "agent_attempt": agent_attempt,
                **metrics,
            }
        )

    return collect


def _llm_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    usage_calls = [call for call in calls if call["usage"]["available"]]

    def token_total(name: str) -> int | None:
        values = [call["usage"].get(name) for call in usage_calls]
        known = [value for value in values if value is not None]
        return sum(known) if known else None

    costs = [call["estimated_cost"] for call in calls]
    known_costs = [cost for cost in costs if cost is not None]
    currencies = {
        call["pricing"]["currency"] for call in calls if call.get("pricing")
    }
    return {
        "llm_seconds": round(sum(call["latency_seconds"] for call in calls), 6),
        "initial_generation_seconds": round(
            sum(
                call["latency_seconds"]
                for call in calls
                if call["call_kind"] in {"generation", "clarification"}
            ),
            6,
        ),
        "repair_generation_seconds": round(
            sum(
                call["latency_seconds"]
                for call in calls
                if call["call_kind"].startswith("repair")
            ),
            6,
        ),
        "token_usage": {
            "available_for_all_calls": bool(calls) and len(usage_calls) == len(calls),
            "calls_with_usage": len(usage_calls),
            "call_count": len(calls),
            "prompt_tokens": token_total("prompt_tokens"),
            "completion_tokens": token_total("completion_tokens"),
            "total_tokens": token_total("total_tokens"),
            "cached_prompt_tokens": token_total("cached_prompt_tokens"),
            "reasoning_tokens": token_total("reasoning_tokens"),
            "text_completion_tokens": token_total("text_completion_tokens"),
        },
        "cost": {
            "available_for_all_calls": bool(calls) and len(known_costs) == len(calls),
            "estimated_total": round(sum(known_costs), 10) if known_costs else None,
            "currency": currencies.pop() if len(currencies) == 1 else None,
        },
    }
