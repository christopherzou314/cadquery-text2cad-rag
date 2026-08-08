# CadQuery Lightweight-RAG Experiment

This experiment compares direct CadQuery generation with generation assisted by
the original eight-entry lightweight reference library and the full CadQuery
reference library.

- Same model and system prompt for all conditions.
- Ten fixed prompts from primitive to complex multi-part objects.
- No traceback repair: compilation success measures the first generated program.
- API transport retries remain enabled and are not generation repairs.
- Condition order alternates for consecutive prompts.
- Version-2 quantitative evaluation checks first-pass execution, STEP/STL
  export, non-degenerate geometry, solid count, and registered bounding-box and
  cylindrical-hole constraints. Version-1 task success is retained for comparison.
- Visual similarity and visual quality are scored from paired PNG previews after
  generation, using the rubric documented in the final report.

Run or resume with:

```bash
python scripts/run_rag_experiment.py --resume
```

Evaluate existing artifacts without making API calls:

```bash
python scripts/evaluate_existing_experiment.py
```

Record an experiment seed and complete per-run telemetry with:

```bash
python scripts/run_rag_experiment.py --seed 101 --limit 1
```

The seed is only sent to the API when `--send-seed` is also supplied. Do this
only for providers that explicitly document support for the request field.

Metric definitions and current limitations are documented in
[`EVALUATION.md`](EVALUATION.md).
