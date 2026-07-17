"""Prompting and code extraction helpers."""

from __future__ import annotations

import re


SYSTEM_PROMPT = """You generate small, executable CadQuery Python programs.

Return only Python code. Do not include Markdown fences or explanation.

Rules:
- Use `import cadquery as cq`.
- Build one final 3D object and assign it to a variable named `result`.
- `result` must be a `cq.Workplane`, `cq.Shape`, or `cq.Assembly`.
- Prefer simple, robust primitives and dimensions in millimeters.
- Do not call `show_object`.
- Do not read files, write files, use network access, or import non-standard packages.
- If dimensions are missing, choose reasonable values and keep proportions faithful.
"""


def build_user_prompt(description: str, reference_context: str | None = None) -> str:
    reference_section = ""
    if reference_context:
        reference_section = f"""

Retrieved CadQuery references:
{reference_context}

Use these references as concise technical guidance. Adapt them to the request; do not merely copy them.
"""

    return f"""Create a CadQuery model for this natural-language CAD request:

{description}
{reference_section}

Remember: output only executable Python code and assign the final object to `result`.
"""


def build_repair_prompt(
    description: str,
    previous_code: str,
    traceback: str,
    reference_context: str | None = None,
) -> str:
    reference_section = ""
    if reference_context:
        reference_section = f"""

Relevant CadQuery references:
{reference_context}
"""

    return f"""The following CadQuery code failed when executed.

Original CAD request:
{description}

Failed code:
```python
{previous_code}
```

Traceback:
```text
{traceback}
```
{reference_section}

Please return a corrected complete CadQuery Python program.
Return only executable Python code. The final object must be assigned to `result`.
"""


_FENCED_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(text: str) -> str:
    """Extract Python from a model response, tolerating accidental Markdown fences."""
    match = _FENCED_BLOCK_RE.search(text)
    code = match.group(1) if match else text
    return code.strip()


def mock_cadquery_code(description: str) -> str:
    """Deterministic fallback for testing the CadQuery execution/export path."""
    lowered = description.lower()
    if "hole" in lowered or "孔" in description:
        return """import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(80, 50, 8)
    .faces(">Z")
    .workplane()
    .hole(12)
)
"""

    if "cylinder" in lowered or "圆柱" in description:
        return """import cadquery as cq

result = cq.Workplane("XY").cylinder(40, 15)
"""

    return """import cadquery as cq

result = cq.Workplane("XY").box(60, 40, 20)
"""
