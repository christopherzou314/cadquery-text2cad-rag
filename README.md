# Text2CAD with CadQuery and RAG

A research-oriented prototype that converts natural-language descriptions into
executable CadQuery programs, validates the generated geometry locally, and
exports CAD models as STEP and STL files.

```text
Natural-language prompt
        -> OpenAI-compatible LLM
        -> CadQuery Python program
        -> local execution and repair loop
        -> STEP / STL / rendered preview
```

The project includes a desktop GUI, an execution-feedback agent, SSE streaming,
two CadQuery retrieval-augmented generation (RAG) modes, and a reproducible
three-way experiment comparing generation without retrieval, lightweight RAG,
and full-documentation RAG.

## Features

- OpenAI-compatible `/chat/completions` API integration
- SSE streaming with retry handling for interrupted or timed-out responses
- Automatic traceback feedback and code repair after CadQuery execution errors
- STEP and STL export with support for `Workplane`, `Shape`, and `Assembly`
- Off-screen VTK rendering for PNG previews
- Tkinter GUI with optional CQ-editor launch and automatic rendering
- Baseline, lightweight RAG, and full RAG generation modes
- Searchable CadQuery Workplane, Sketch, Assembly, Examples, and API references
- Reproducible experiment scripts, metrics, reports, and visual comparisons

## Requirements

- Python 3.10 or later
- CadQuery
- Pillow
- VTK
- Tkinter (normally included with Python)
- CQ-editor (optional)

Use a Python environment in which CadQuery can be imported:

```bash
python -c "import cadquery; print(cadquery.__version__)"
```

Install the main Python dependencies if they are not already available:

```bash
python -m pip install cadquery Pillow vtk
```

## Configuration

Create a local environment file from the template:

```bash
cp .env.example .env
```

Configure an OpenAI-compatible provider in `.env`:

```dotenv
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.example.com/v1
OPENAI_MODEL=your-model-name

# Optional: a specific interpreter containing CadQuery.
CADQUERY_PYTHON=/path/to/your/cadquery/python
```

The program appends `/chat/completions` to `OPENAI_BASE_URL`. Do not include that
suffix in the environment variable.

`.env` is ignored by Git. Never commit API keys or access tokens.

## Quick Start

Run the complete export path without making an API request:

```bash
python -m src.text2cad.main \
  "a rectangular plate with a circular hole in the center" \
  --mock
```

Run a real model request:

```bash
python -m src.text2cad.main \
  "a 90 mm by 50 mm mounting plate with four 6 mm corner holes"
```

Generated artifacts are written to a timestamped directory under `outputs/`:

```text
generated_model.py   generated CadQuery program
model.step           STEP model
model.stl            STL mesh
run.json             execution metadata and captured errors
agent_run.json       agent attempts and generation setting
retrieval.json       retrieved references when RAG is enabled
```

## Agent Repair Loop

By default, the agent can make up to two repair attempts. When generated code
fails, the previous program and its traceback are sent back to the same model,
which must return a complete corrected program.

```bash
python -m src.text2cad.main \
  "an L-shaped bracket with four bolt holes" \
  --max-repairs 3
```

API transport retries are handled separately from code-repair attempts. A
partially received stream is discarded before the request is retried, preventing
incomplete code from being executed.

## RAG Modes

The command-line interface and GUI provide three generation settings:

| Mode | CLI option | Reference source |
|---|---|---|
| Baseline | `--rag-mode off` | No retrieved context |
| Lightweight RAG | `--rag-mode lightweight` | Eight curated CadQuery modelling patterns |
| Full RAG | `--rag-mode full` | Curated patterns, official guides, examples, and API index |

Example:

```bash
python -m src.text2cad.main \
  "an open-top box with 3 mm walls" \
  --rag-mode full \
  --rag-top-k 3
```

The knowledge base is stored in [`knowledge/`](knowledge/). Rebuild the API and
official-guide indexes after upgrading CadQuery:

```bash
python scripts/build_cadquery_reference.py
```

## Desktop GUI

Start the GUI with:

```bash
python -m src.text2cad.gui
```

The interface supports prompt entry, repair-count selection, knowledge-mode
selection, generation logs, source-code inspection, and rendered STL previews.
When CQ-editor is available, the generated model can also be opened and rendered
automatically in its interactive 3D viewport.

## RAG Experiment

The included experiment evaluates 10 prompts at increasing difficulty under all
three generation settings. Traceback repair is disabled during evaluation so
that compilation measures the first generated program.

| Metric | Baseline | Lightweight RAG | Full RAG |
|---|---:|---:|---:|
| First-generation compilation success | 100% | 100% | 100% |
| Visible geometry | 100% | 100% | 100% |
| Explicit dimensions reasonable | 90% | 90% | 100% |
| Mean visual quality (1-5) | 4.60 | 4.70 | 4.90 |
| Mean readability (1-5) | 4.90 | 4.90 | 5.00 |

![Three-way comparison for prompts P06-P10](experiments/cadquery_rag/results/contact_sheets/three_way_2.png)

Full reports and result tables are available under
[`experiments/cadquery_rag/results/`](experiments/cadquery_rag/results/).

Run the experiment and rebuild the report with:

```bash
python scripts/run_rag_experiment.py --resume
python scripts/analyze_rag_experiment.py
```

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover streaming response assembly, connection retries, partial-stream
recovery, RAG retrieval behavior, and CadQuery Assembly export.

## Project Structure

```text
src/text2cad/
  main.py       command-line interface
  gui.py        desktop interface and CQ-editor integration
  agent.py      generation, execution, and traceback-repair loop
  llm.py        streaming OpenAI-compatible API client
  prompts.py    generation prompts and response cleanup
  rag.py        reference loading, retrieval, and logging
  runner.py     isolated-process execution and STEP/STL export
  renderer.py   off-screen VTK preview rendering

knowledge/      curated and generated CadQuery references
experiments/    fixed prompts, reports, metrics, and comparison figures
scripts/        knowledge-index and experiment utilities
tests/          unit and export regression tests
```

## Security

This project executes Python code produced by an LLM. The generated program is
run in a separate process with a timeout, but it is not a complete security
sandbox. Use trusted model providers, inspect generated code when appropriate,
and run the project in an isolated development environment when testing unknown
prompts or models.
