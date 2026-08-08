# CadQuery Benchmark v1: Final Experimental Report

## Executive Summary

This held-out evaluation contains 60 prompts across 10 categories, three generation conditions, and three seeds. All 540 scheduled API tasks completed: 486 CAD-generation runs and 54 clarification-only runs.

Across CAD runs, EndToEndPass increased from 355/486 (73.05%) at B=0 to 445/486 (91.56%) at B=2, a gain of 18.52 percentage points. The final ExecutionPass@2 was 98.97%, so hard constraints rather than Python/CadQuery execution were the larger remaining source of failure.

Baseline achieved the highest EndToEndPass@2 (93.83%), followed by lightweight RAG (90.74%) and full RAG (90.12%). Under this retrieval and prompting configuration, adding reference material did not improve the primary automatic metric.

![End-to-end pass curve](figures/end_to_end_pass_curve.png)

## Experimental Design

- Model: `qwen3.7-max` with temperature 0.1.
- Test set: B013-B072, 60 held-out cases, six per category.
- Seeds: 101, 202, 303.
- Conditions: baseline, lightweight curated CadQuery references, and full-documentation RAG.
- Unified repair budget: B=2. Execution and hard-constraint repairs consume the same budget; unchanged transport retries do not.
- Automatic hard constraints: bounding-box dimensions/bounds, STEP solid count, and cylindrical hole patterns.
- Conflicting-description cases request clarification and skip CAD execution and repair.

## Metric Definitions

- **ExecutionPass@B:** within budget B, code executes and required STEP/STL artifacts are exported.
- **ConstraintPass@B:** within budget B, every registered automatic hard constraint is evaluated and passed.
- **EndToEndPass@B:** execution, exports, valid non-degenerate geometry, and all registered hard constraints pass.
- **B=0** is the initial generation; B=1 and B=2 permit one and two model revisions, respectively.

## Main Results

| Metric | @0 | @1 | @2 |
|---|---:|---:|---:|
| ExecutionPass | 450/486 (92.59%) | 476/486 (97.94%) | 481/486 (98.97%) |
| ConstraintPass | 356/486 (73.25%) | 433/486 (89.09%) | 445/486 (91.56%) |
| EndToEndPass | 355/486 (73.05%) | 432/486 (88.89%) | 445/486 (91.56%) |

| Condition | CAD n | E2E@0 | E2E@1 | E2E@2 | Repairs | Tokens | Cost | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 162 | 76.54% | 90.74% | 93.83% | 53 | 1,598,172 | USD 5.7544 | 85.75s |
| Lightweight RAG | 162 | 76.54% | 88.27% | 90.74% | 57 | 1,978,328 | USD 6.9308 | 87.47s |
| Full RAG | 162 | 66.05% | 87.65% | 90.12% | 75 | 1,892,655 | USD 6.5085 | 98.04s |

The B=2 EndToEndPass estimate has a run-level Wilson 95% interval of 88.76% to 93.72%. This interval is descriptive because runs from the same prompt are not fully independent.

## Paired Condition Comparisons

| Comparison | B | Baseline | Comparison | Difference | Baseline only | Comparison only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lightweight RAG vs Baseline | 0 | 124/162 | 124/162 | +0.00 pp | 21 | 21 | 1.0000 |
| Lightweight RAG vs Baseline | 1 | 147/162 | 143/162 | -2.47 pp | 11 | 7 | 0.4807 |
| Lightweight RAG vs Baseline | 2 | 152/162 | 147/162 | -3.09 pp | 7 | 2 | 0.1797 |
| Full RAG vs Baseline | 0 | 124/162 | 107/162 | -10.49 pp | 32 | 15 | 0.0186 |
| Full RAG vs Baseline | 1 | 147/162 | 142/162 | -3.09 pp | 12 | 7 | 0.3593 |
| Full RAG vs Baseline | 2 | 152/162 | 146/162 | -3.70 pp | 11 | 5 | 0.2101 |

The exact McNemar tests pair conditions by prompt and seed. They test run-level discordance and do not establish generalization beyond this benchmark.

## Repair Effectiveness

There were 131 initial end-to-end failures. The closed loop recovered 77 by B=1 and 90 by B=2; the second repair added 13 recoveries. Conditional recovery at B=2 was 68.70%.

The system used 185 model revisions: 60 execution-triggered and 125 constraint-triggered. Mean repair count was 0.381 per CAD run and 1.412 among repaired runs.

## Category and Seed Results

![Category results](figures/category_end_to_end_at_2.png)

| Category | Baseline E2E@2 | Lightweight RAG | Full RAG |
|---|---:|---:|---:|
| ambiguous | 18/18 (100.00%) | 17/18 (94.44%) | 16/18 (88.89%) |
| assembly | 18/18 (100.00%) | 18/18 (100.00%) | 18/18 (100.00%) |
| complex_boolean | 15/18 (83.33%) | 12/18 (66.67%) | 14/18 (77.78%) |
| conflicting | clarification only | clarification only | clarification only |
| curved_surface | 17/18 (94.44%) | 17/18 (94.44%) | 15/18 (83.33%) |
| holes_and_slots | 15/18 (83.33%) | 15/18 (83.33%) | 15/18 (83.33%) |
| out_of_distribution | 18/18 (100.00%) | 18/18 (100.00%) | 17/18 (94.44%) |
| primitive | 16/18 (88.89%) | 15/18 (83.33%) | 17/18 (94.44%) |
| revolved | 17/18 (94.44%) | 18/18 (100.00%) | 18/18 (100.00%) |
| thin_wall | 18/18 (100.00%) | 17/18 (94.44%) | 16/18 (88.89%) |

![Seed stability](figures/seed_stability_at_2.png)

| Seed | Baseline E2E@2 | Lightweight RAG | Full RAG |
|---:|---:|---:|---:|
| 101 | 51/54 (94.44%) | 49/54 (90.74%) | 50/54 (92.59%) |
| 202 | 50/54 (92.59%) | 47/54 (87.04%) | 49/54 (90.74%) |
| 303 | 51/54 (94.44%) | 51/54 (94.44%) | 47/54 (87.04%) |

## Remaining Automatic Failures

At B=2, 41 of 486 CAD runs remained unsuccessful. The final-stage breakdown was hard_constraint=36, execution_or_export=5.

| Failed hard-constraint type at B=2 | Failed groups |
|---|---:|
| cylindrical_hole_pattern | 21 |
| solid_count | 11 |
| bbox_dimensions | 11 |
| bbox_bounds | 7 |

The complete run-level list is stored in `final_failure_runs.csv`.

## Resource Use and Reliability

![Resource use](figures/resource_use_by_condition.png)

| Condition | Total tokens | Cost | Mean latency | Median latency | P95 latency | Transport retries |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1,598,172 | USD 5.7544 | 161.07s | 85.75s | 565.88s | 37 |
| Lightweight RAG | 1,978,328 | USD 6.9308 | 186.18s | 87.47s | 572.28s | 30 |
| Full RAG | 1,892,655 | USD 6.5085 | 192.97s | 98.04s | 633.83s | 53 |

| LLM call type | Calls | Tokens | Cost | Total LLM latency |
|---|---:|---:|---:|---:|
| clarification | 54 | 62,497 | USD 0.1672 | 1381.30s |
| generation | 486 | 3,828,582 | USD 13.6964 | 68840.35s |
| repair_constraint | 125 | 1,163,369 | USD 3.9577 | 19058.45s |
| repair_execution | 60 | 414,707 | USD 1.3724 | 7266.45s |

Total recorded usage was 5,469,155 tokens and USD 19.1937. There were 120 transport retries across 81 recorded tasks. Network retries did not consume the model repair budget.

## Clarification and Deferred Evaluation

All 54/54 conflicting-description tasks produced a clarification response. Their correctness remains unscored. The generated `manual_scoring.csv` contains 594 rows reserved for later human or VLM assessment. None of these pending scores are included in the automatic success rates.

## Limitations

- Automatic evaluation does not yet measure recognizability, semantic similarity, slot dimensions, wall thickness, or general visual quality unless encoded by a registered v2 constraint.
- Clarification quality is pending; capturing a response is not equivalent to answering correctly.
- The study uses one model, one provider, three seeds, and one retrieval configuration.
- Full RAG retrieves top-k excerpts; it does not place every documentation page into every request.
- Run-level confidence intervals understate prompt-level dependence; paired tests partially address this but do not replace replication on new case sets.
- Usage totals exclude failed provider requests that returned no usage metadata, including some job-level retries completed later by resume.

## Conclusion

The principal positive result is the constraint-aware repair loop: it raised EndToEndPass from 73.05% to 91.56%. The majority of revisions were triggered by measured geometric constraint failures, supporting the value of execution-plus-geometry feedback over traceback-only repair. In contrast, neither RAG mode outperformed baseline on the primary automatic metric in this study. The next evaluation layer should score semantic and visual quality with blinded human review or a validated VLM rubric while expanding the hard-constraint registry.

## Reproducibility

The report was generated directly from `records.json` with:

```bash
python scripts/generate_final_benchmark_report.py
```
