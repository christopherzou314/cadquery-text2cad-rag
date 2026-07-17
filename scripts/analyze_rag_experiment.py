"""Create three-way CSV and Markdown analysis for the CadQuery RAG experiment."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "cadquery_rag"
RESULTS = EXPERIMENT / "results"
CONDITIONS = ("baseline", "lightweight_rag", "full_rag")

VISUAL_SCORES = {
    "P01": (5, 5, 5), "P02": (5, 5, 5), "P03": (5, 5, 5),
    "P04": (5, 5, 5), "P05": (5, 5, 5), "P06": (5, 5, 5),
    "P07": (5, 5, 5), "P08": (4, 5, 5), "P09": (3, 3, 5),
    "P10": (4, 4, 4),
}

READABILITY_SCORES = {
    prompt_id: ((4, 4, 5) if prompt_id == "P09" else (5, 5, 5))
    for prompt_id in VISUAL_SCORES
}

ERRORS = {
    "P01": "None observed.",
    "P02": "None observed.",
    "P03": "All three produce an open top and correct wall/bottom thickness.",
    "P04": "None observed in final geometry.",
    "P05": "None observed; Full RAG interprets 45 mm as the seat's top-surface height.",
    "P06": "All three contain four correctly placed through holes.",
    "P07": "All three contain the flange, hub, bore, and four bolt holes.",
    "P08": "Baseline extends the vertical plate 2.5 mm outward; both RAG versions place it correctly.",
    "P09": "Baseline and Lightweight RAG mis-center XZ extrusions (105 mm width). Full RAG explicitly uses +Y cylinders and produces 63 mm width.",
    "P10": "All three are recognizable but use simplified rectangular wing and tail surfaces.",
}


def main() -> None:
    prompts = {
        item["id"]: item
        for item in json.loads((EXPERIMENT / "prompt_set.json").read_text(encoding="utf-8"))
    }
    records_path = RESULTS / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (RESULTS / "experiment_metadata.json").read_text(encoding="utf-8")
    )

    rows = []
    for prompt_id, prompt in prompts.items():
        condition_records = [records[f"{prompt_id}:{name}"] for name in CONDITIONS]
        for record in condition_records:
            _refresh_dimension_result(record, prompt["target_dimensions_mm"])
            record["parameterized_code"] = record.get("named_numeric_parameters", 0) >= 1
        visual = VISUAL_SCORES[prompt_id]
        readability = READABILITY_SCORES[prompt_id]
        row = {
            "prompt_id": prompt_id,
            "difficulty": prompt["difficulty"],
            "category": prompt["category"],
            "main_errors_observed": ERRORS[prompt_id],
        }
        for index, (name, record) in enumerate(zip(CONDITIONS, condition_records, strict=True)):
            row.update({
                f"{name}_compiles": record["compile_success"],
                f"{name}_visible": record["visible_geometry"],
                f"{name}_dimensions_reasonable": record["dimension_reasonable"],
                f"{name}_dimensions_mm": record["actual_dimensions_mm"],
                f"{name}_visual_score_1_to_5": visual[index],
                f"{name}_readability_1_to_5": readability[index],
                f"{name}_parameterized": record["parameterized_code"],
                f"{name}_elapsed_seconds": record["elapsed_seconds"],
                f"{name}_reference_ids": record.get("reference_ids", []),
            })
        rows.append(row)

    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(RESULTS / "three_way_results.csv", rows)
    (RESULTS / "experiment_report.md").write_text(
        _build_report(rows, records, metadata), encoding="utf-8"
    )


def _refresh_dimension_result(record: dict, targets: list[float | None]) -> None:
    errors = [
        None if target is None else abs(actual - target) / target
        for actual, target in zip(record["actual_dimensions_mm"], targets, strict=True)
    ]
    record["target_dimensions_mm"] = targets
    record["dimension_relative_errors"] = [
        None if value is None else round(value, 3) for value in errors
    ]
    record["dimension_reasonable"] = max(
        (value for value in errors if value is not None), default=0
    ) <= 0.25


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _metrics(rows: list[dict], records: dict, condition: str) -> dict:
    return {
        "compile": sum(row[f"{condition}_compiles"] for row in rows),
        "visible": sum(row[f"{condition}_visible"] for row in rows),
        "dimensions": sum(row[f"{condition}_dimensions_reasonable"] for row in rows),
        "parameterized": sum(row[f"{condition}_parameterized"] for row in rows),
        "visual": statistics.mean(row[f"{condition}_visual_score_1_to_5"] for row in rows),
        "readability": statistics.mean(row[f"{condition}_readability_1_to_5"] for row in rows),
        "time": statistics.mean(row[f"{condition}_elapsed_seconds"] for row in rows),
        "retries": _retry_count(records, condition),
    }


def _build_report(rows: list[dict], records: dict, metadata: dict) -> str:
    n = len(rows)
    b = _metrics(rows, records, "baseline")
    l = _metrics(rows, records, "lightweight_rag")
    f = _metrics(rows, records, "full_rag")
    table_rows = [
        f"| {row['prompt_id']} | {row['difficulty']} | "
        f"{_yes(row['baseline_compiles'])} | {_yes(row['lightweight_rag_compiles'])} | {_yes(row['full_rag_compiles'])} | "
        f"{row['baseline_visual_score_1_to_5']} | {row['lightweight_rag_visual_score_1_to_5']} | {row['full_rag_visual_score_1_to_5']} | "
        f"{row['main_errors_observed']} |"
        for row in rows
    ]

    def metric_row(label: str, key: str, percentage: bool = False) -> str:
        if percentage:
            values = [f"{m[key]}/{n} ({m[key]/n:.0%})" for m in (b, l, f)]
        elif key in {"visual", "readability"}:
            values = [f"{m[key]:.2f}" for m in (b, l, f)]
        elif key == "time":
            values = [f"{m[key]:.1f} s" for m in (b, l, f)]
        else:
            values = [str(m[key]) for m in (b, l, f)]
        return f"| {label} | " + " | ".join(values) + " |"

    return f"""# Baseline vs Lightweight RAG vs Full RAG for CadQuery

## Experimental setup

- Model: `{metadata['model']}`; 10 fixed prompts; temperature 0.1; retrieval top-k = 3.
- Conditions: no retrieval, the original eight-entry lightweight library, and the full CadQuery library.
- Traceback repair was disabled, so compilation measures the first generated program. Network retries do not alter code.
- Full RAG was added in a later session using the same model and settings; timing is therefore descriptive rather than a controlled latency comparison.
- Dimension checks use a 25% tolerance and only explicitly constrained dimensions.

## Scoring rubric

Visual quality: 1 = wrong/invisible, 2 = major structural errors, 3 = recognizable with significant errors, 4 = good but simplified/minor errors, 5 = complete and geometrically faithful.

## Three-way results

| Prompt | Difficulty | Baseline compiles | Light compiles | Full compiles | Baseline visual | Light visual | Full visual | Main errors |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table_rows)}

## Aggregate results

| Metric | Baseline | Lightweight RAG | Full RAG |
|---|---:|---:|---:|
{metric_row('First-generation compilation success', 'compile', True)}
{metric_row('Visible geometry', 'visible', True)}
{metric_row('Explicit dimensions reasonable', 'dimensions', True)}
{metric_row('Parameterized code', 'parameterized', True)}
{metric_row('Mean visual quality (1-5)', 'visual')}
{metric_row('Mean readability (1-5)', 'readability')}
{metric_row('Mean end-to-end time', 'time')}
{metric_row('API transport retries observed', 'retries')}

## Analysis

All three conditions reached 100% first-generation compilation and visible geometry, so neither knowledge base improved those ceiling-level metrics. Lightweight RAG made one small improvement on P08, increasing mean visual quality from {b['visual']:.2f} to {l['visual']:.2f}.

Full RAG produced the strongest substantive improvement on P09. Baseline and Lightweight RAG both used `Workplane(\"XZ\")` with an incorrect signed translation, displacing the cart's wheels and axles and producing a 105 mm overall width. Full RAG retrieved assembly guidance and generated explicit `Solid.makeCylinder` calls with direction `(0, 1, 0)`, producing a sensible 63 mm width. Full RAG therefore reached 10/10 reasonable-dimension results and a mean visual score of {f['visual']:.2f}.

The full library also introduced retrieval noise. P01 and P04 received API entries such as `Plane.named`, `BoundBox.isInside`, and `Shape.isEqual` that were not useful for their tasks. The model still succeeded, but this shows that expanding a lexical knowledge base without improving ranking can waste the top-k context window. Full RAG's benefit came from cases where retrieval found a genuinely relevant skill, especially assemblies and explicit cylinder directions.

All 30 programs used named parameters and remained readable. Timing and retry counts should not be interpreted as a controlled comparison because Full RAG was run later and API service conditions varied.

## Limitations

- Each condition was generated once per prompt; outputs are stochastic and the sample is small.
- Visual scoring combines fixed VTK previews with code inspection and remains a human judgment.
- Full RAG was added after the first two conditions rather than interleaved with them.
- Five earlier Assembly outputs exposed an export harness bug; their unmodified LLM code was re-executed after Assembly-to-Compound conversion, with the correction recorded in `records.json`.
- A stronger follow-up should run at least three repetitions and compare lexical retrieval with embedding or reranker-based retrieval.
"""


def _yes(value: bool) -> str:
    return "Yes" if value else "No"


def _retry_count(records: dict, condition: str) -> int:
    count = 0
    for record in records.values():
        if record["condition"] != condition:
            continue
        log = Path(record["run_dir"]).parent / "progress.log"
        if log.exists():
            final_run = log.read_text(encoding="utf-8").rsplit(
                "Calling LLM to generate CadQuery code...", 1
            )[-1]
            count += final_run.count("Retrying")
    return count


if __name__ == "__main__":
    main()
