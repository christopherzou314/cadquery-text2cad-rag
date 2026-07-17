# Baseline vs Lightweight RAG vs Full RAG for CadQuery

## Experimental setup

- Model: `glm-5.2`; 10 fixed prompts; temperature 0.1; retrieval top-k = 3.
- Conditions: no retrieval, the original eight-entry lightweight library, and the full CadQuery library.
- Traceback repair was disabled, so compilation measures the first generated program. Network retries do not alter code.
- Full RAG was added in a later session using the same model and settings; timing is therefore descriptive rather than a controlled latency comparison.
- Dimension checks use a 25% tolerance and only explicitly constrained dimensions.

## Scoring rubric

Visual quality: 1 = wrong/invisible, 2 = major structural errors, 3 = recognizable with significant errors, 4 = good but simplified/minor errors, 5 = complete and geometrically faithful.

## Three-way results

| Prompt | Difficulty | Baseline compiles | Light compiles | Full compiles | Baseline visual | Light visual | Full visual | Main errors |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P01 | easy | Yes | Yes | Yes | 5 | 5 | 5 | None observed. |
| P02 | easy | Yes | Yes | Yes | 5 | 5 | 5 | None observed. |
| P03 | medium | Yes | Yes | Yes | 5 | 5 | 5 | All three produce an open top and correct wall/bottom thickness. |
| P04 | medium | Yes | Yes | Yes | 5 | 5 | 5 | None observed in final geometry. |
| P05 | medium | Yes | Yes | Yes | 5 | 5 | 5 | None observed; Full RAG interprets 45 mm as the seat's top-surface height. |
| P06 | medium | Yes | Yes | Yes | 5 | 5 | 5 | All three contain four correctly placed through holes. |
| P07 | hard | Yes | Yes | Yes | 5 | 5 | 5 | All three contain the flange, hub, bore, and four bolt holes. |
| P08 | hard | Yes | Yes | Yes | 4 | 5 | 5 | Baseline extends the vertical plate 2.5 mm outward; both RAG versions place it correctly. |
| P09 | hard | Yes | Yes | Yes | 3 | 3 | 5 | Baseline and Lightweight RAG mis-center XZ extrusions (105 mm width). Full RAG explicitly uses +Y cylinders and produces 63 mm width. |
| P10 | complex | Yes | Yes | Yes | 4 | 4 | 4 | All three are recognizable but use simplified rectangular wing and tail surfaces. |

## Aggregate results

| Metric | Baseline | Lightweight RAG | Full RAG |
|---|---:|---:|---:|
| First-generation compilation success | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Visible geometry | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Explicit dimensions reasonable | 9/10 (90%) | 9/10 (90%) | 10/10 (100%) |
| Parameterized code | 10/10 (100%) | 10/10 (100%) | 10/10 (100%) |
| Mean visual quality (1-5) | 4.60 | 4.70 | 4.90 |
| Mean readability (1-5) | 4.90 | 4.90 | 5.00 |
| Mean end-to-end time | 71.1 s | 106.8 s | 111.7 s |
| API transport retries observed | 1 | 3 | 3 |

## Analysis

All three conditions reached 100% first-generation compilation and visible geometry, so neither knowledge base improved those ceiling-level metrics. Lightweight RAG made one small improvement on P08, increasing mean visual quality from 4.60 to 4.70.

Full RAG produced the strongest substantive improvement on P09. Baseline and Lightweight RAG both used `Workplane("XZ")` with an incorrect signed translation, displacing the cart's wheels and axles and producing a 105 mm overall width. Full RAG retrieved assembly guidance and generated explicit `Solid.makeCylinder` calls with direction `(0, 1, 0)`, producing a sensible 63 mm width. Full RAG therefore reached 10/10 reasonable-dimension results and a mean visual score of 4.90.

The full library also introduced retrieval noise. P01 and P04 received API entries such as `Plane.named`, `BoundBox.isInside`, and `Shape.isEqual` that were not useful for their tasks. The model still succeeded, but this shows that expanding a lexical knowledge base without improving ranking can waste the top-k context window. Full RAG's benefit came from cases where retrieval found a genuinely relevant skill, especially assemblies and explicit cylinder directions.

All 30 programs used named parameters and remained readable. Timing and retry counts should not be interpreted as a controlled comparison because Full RAG was run later and API service conditions varied.

## Limitations

- Each condition was generated once per prompt; outputs are stochastic and the sample is small.
- Visual scoring combines fixed VTK previews with code inspection and remains a human judgment.
- Full RAG was added after the first two conditions rather than interleaved with them.
- Five earlier Assembly outputs exposed an export harness bug; their unmodified LLM code was re-executed after Assembly-to-Compound conversion, with the correction recorded in `records.json`.
- A stronger follow-up should run at least three repetitions and compare lexical retrieval with embedding or reranker-based retrieval.
