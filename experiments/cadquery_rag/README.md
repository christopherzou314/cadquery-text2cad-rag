# CadQuery Lightweight-RAG Experiment

This experiment compares direct CadQuery generation with generation assisted by
the original eight-entry lightweight reference library and the full CadQuery
reference library.

- Same model and system prompt for all conditions.
- Ten fixed prompts from primitive to complex multi-part objects.
- No traceback repair: compilation success measures the first generated program.
- API transport retries remain enabled and are not generation repairs.
- Condition order alternates for consecutive prompts.
- Dimensions are checked against STL bounding boxes with a 25% tolerance.
- Visual similarity and visual quality are scored from paired PNG previews after
  generation, using the rubric documented in the final report.

Run or resume with:

```bash
python scripts/run_rag_experiment.py --resume
```
