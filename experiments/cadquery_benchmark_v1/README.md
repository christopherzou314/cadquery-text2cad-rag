# CadQuery Benchmark v1

This directory contains the final held-out 60-case evaluation for the
constraint-aware CadQuery text-to-CAD agent.

## Study Design

- 60 cases, B013-B072.
- 10 categories with exactly 6 cases per category.
- Three conditions: baseline, lightweight RAG, and full RAG.
- Three provider seeds: 101, 202, and 303.
- Unified repair budget `B=2`.
- Execution and hard-constraint repairs consume the same budget.
- Network retries of an unchanged request do not consume repair budget.
- Conflicting-description cases request clarification and skip CAD execution.

The complete matrix contains 540 API tasks: 486 CAD-generation tasks and 54
clarification-only tasks. Automatic CAD metrics exclude clarification tasks.

## Automatic Metrics

- `ExecutionPass@B`: code executes and required STEP/STL artifacts are exported
  within repair budget B.
- `ConstraintPass@B`: every registered automatic hard constraint is evaluated
  and passed within budget B.
- `EndToEndPass@B`: execution, exports, valid non-degenerate geometry, and all
  registered hard constraints pass within budget B.

The frozen evaluator supports bounding-box dimensions and bounds, exact STEP
solid count, and cylindrical-hole pattern checks. Manual semantic and visual
criteria remain separate and are never counted as automatically satisfied.

## Final Results

All 540 scheduled tasks completed.

| Metric | B=0 | B=1 | B=2 |
|---|---:|---:|---:|
| ExecutionPass | 92.59% | 97.94% | 98.97% |
| ConstraintPass | 73.25% | 89.09% | 91.56% |
| EndToEndPass | 73.05% | 88.89% | 91.56% |

EndToEndPass@2 was 93.83% for baseline, 90.74% for lightweight RAG, and
90.12% for full RAG. The closed loop recovered 90 of 131 initial end-to-end
failures. It used 185 model revisions: 60 execution-triggered repairs and 125
constraint-triggered repairs.

The final English and Chinese reports are available at:

- [`results/full_b2/final_report.md`](results/full_b2/final_report.md)
- [`results/full_b2/final_report_zh.md`](results/full_b2/final_report_zh.md)

## Files

- `full_cases.json`: frozen final 60-case test set.
- `benchmark_config.json`: conditions, seeds, category balance, and budgets.
- `case_schema.json`: JSON Schema for benchmark cases.
- `../evaluation_contract_v2.json`: frozen evaluator semantics.
- `results/full_b2/final_summary.json`: machine-readable aggregate statistics.
- `results/full_b2/final_*_summary.csv`: condition, category, and seed tables.
- `results/full_b2/final_pairwise_comparison.csv`: paired McNemar comparisons.
- `results/full_b2/final_failure_runs.csv`: remaining B=2 automatic failures.
- `results/full_b2/manual_scoring.csv`: blank human/VLM scoring worksheet.
- `results/full_b2/figures/`: report figures.

Raw generated programs, STEP/STL files, logs, and machine-local paths are not
published. They are ignored by Git and remain in the local experiment folder.

## Validate and Run

Validate the final case set without making API requests:

```bash
python scripts/validate_benchmark.py \
  --cases experiments/cadquery_benchmark_v1/full_cases.json
```

Run or resume the complete experiment. `CADQUERY_PYTHON` should point to the
Python interpreter containing CadQuery when it is not the active interpreter.

```bash
bash scripts/run_full_benchmark.sh
```

The launcher calls the equivalent command:

```bash
python scripts/run_benchmark.py \
  --cases experiments/cadquery_benchmark_v1/full_cases.json \
  --experiment-dir experiments/cadquery_benchmark_v1/results/full_b2 \
  --max-repairs 2 \
  --resume
```

After a complete run, regenerate every final table, figure, report, and the
manual-scoring worksheet with:

```bash
python scripts/generate_final_benchmark_report.py
```

## Interpretation Boundary

The automatic results do not establish visual recognizability or general
semantic correctness. The 54 clarification responses and all other manual
checks remain available for later blinded human review or validated VLM
evaluation. See the final report for confidence intervals, paired comparisons,
resource use, failure analysis, and limitations.
