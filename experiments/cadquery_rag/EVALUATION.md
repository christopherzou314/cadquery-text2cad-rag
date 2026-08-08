# Quantitative Evaluation Schema v2

## Frozen Contract

Schema `2.0` was frozen on 2026-08-04. Its machine-readable contract is stored
at [`../evaluation_contract_v2.json`](../evaluation_contract_v2.json). Bug fixes
that do not change metric meaning may retain this version. Adding a constraint
type or changing pass/fail semantics requires a new evaluation schema version.

Version 2 evaluates generated CadQuery programs at five levels:

1. **Execution**: whether the first program executes and whether execution
   succeeds within the configured repair budget.
2. **Export**: whether non-empty STEP and STL files are produced.
3. **Mesh geometry**: whether the STL contains a non-empty mesh with finite,
   non-degenerate bounding-box dimensions.
4. **STEP topology**: whether the exact BRep is valid, how many solids it
   contains, and which internal cylindrical faces represent holes.
5. **Registered constraints**: whether machine-readable dimensions, positions,
   solid counts, and cylindrical-hole requirements pass their tolerances.

## Constraint Formats

Bounding-box dimensions and positions use the original version-1 formats:

```json
{
  "id": "overall_dimensions",
  "type": "bbox_dimensions",
  "hard": true,
  "expected_mm": {"x": 90, "y": 50, "z": 6},
  "absolute_tolerance_mm": 0.5,
  "relative_tolerance": 0.01
}
```

```json
{
  "id": "single_solid",
  "type": "solid_count",
  "hard": true,
  "expected": 1
}
```

```json
{
  "id": "bolt_circle_holes",
  "type": "cylindrical_hole_pattern",
  "hard": true,
  "expected_count": 4,
  "diameter_mm": 5,
  "diameter_tolerance_mm": 0.3,
  "axis": "z",
  "through": true,
  "pitch_circle_diameter_mm": 46,
  "position_tolerance_mm": 0.5
}
```

Supported types are:

- `bbox_dimensions`: selected overall X/Y/Z dimensions.
- `bbox_bounds`: selected minimum or maximum X/Y/Z coordinates.
- `solid_count`: number of solids in the imported STEP BRep.
- `cylindrical_hole_pattern`: hole count, diameter, principal axis, through or
  blind status, and optional pitch-circle diameter.

For each bounding-box scalar, the allowed error is:

```text
max(absolute_tolerance_mm, abs(expected_mm) * relative_tolerance)
```

Hole checks use exact cylindrical BRep faces rather than screenshots or STL
facets. Internal faces are separated from exterior cylinders by surface-normal
orientation. Through status is tested with a thin axial probe, and pitch-circle
diameter is computed from projected hole centers, making it rotation-invariant
within the specified principal-axis plane.

## Main Metrics

- `first_generation_success`: attempt 0 executes successfully.
- `success_within_repair_budget`: at least one execution attempt succeeds.
- `artifact_success`: final execution succeeds, both exports exist, and the STL
  passes the non-degenerate geometry check.
- `constraint_group_pass_rate`: passed evaluated groups divided by evaluated
  groups.
- `constraint_evaluation_coverage`: evaluated registered groups divided by all
  registered groups.
- `task_success_v1`: artifact success plus all registered hard bounding-box
  constraints. This preserves direct comparison with earlier results.
- `task_success_v2`: artifact success plus every registered v2 hard constraint,
  including STEP solid and hole checks.

An unavailable artifact does not silently pass its constraints. The affected
groups are marked unevaluated, evaluation coverage falls, and task success is
false because not every registered hard constraint was evaluated and passed.

## Current Coverage

The fixed ten-prompt set registers 27 hard constraint groups: 13 bounding-box
groups, 9 solid-count groups, and 5 cylindrical-hole groups. It checks all
explicit hole counts and diameters in P06-P08, through-hole status and direction,
and the P07 bolt-circle diameter.

Version 2 still does not automatically judge every natural-language property.
Examples include P03 wall thickness and open-top semantics, P04/P05 individual
part dimensions, P06 edge inset, P08 symmetry, P09 assembly names and placements,
P10 airplane recognizability, and source-code parameterization. These remain
future constraint groups rather than being treated as implicit passes.

## Outputs

Each new run writes `evaluation.json` beside its generated code and CAD files.
For existing results, run:

```bash
python scripts/evaluate_existing_experiment.py
```

This performs no API calls. It creates:

- `results/evaluation_results.csv`: one quantitative row per run.
- `results/evaluation_summary.json`: aggregate metrics by generation condition.
- `raw/.../evaluation.json`: complete per-run checks and measured features.
