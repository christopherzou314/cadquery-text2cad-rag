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


CLARIFICATION_SYSTEM_PROMPT = """You assess natural-language CAD requests for internal consistency.

The request may contain mutually incompatible mandatory requirements. If it does,
identify the exact conflict and ask one concise clarification question. Do not
choose one interpretation, generate CAD code, or propose corrected dimensions.
Return natural-language text only.
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


def build_clarification_prompt(
    description: str, reference_context: str | None = None
) -> str:
    reference_section = ""
    if reference_context:
        reference_section = f"""

Retrieved CadQuery references:
{reference_context}

Use the references only to understand CAD terminology. Do not generate code.
"""

    return f"""Review this CAD request for mutually incompatible mandatory requirements:

{description}
{reference_section}

Identify the conflict precisely and ask for the minimum clarification needed.
Return natural-language text only.
"""


def build_repair_prompt(
    description: str,
    previous_code: str,
    feedback: str,
    reference_context: str | None = None,
    repair_type: str = "execution",
) -> str:
    reference_section = ""
    if reference_context:
        reference_section = f"""

Relevant CadQuery references:
{reference_context}
"""

    if repair_type == "execution":
        failure_heading = "The following CadQuery code failed execution or artifact validation."
        feedback_heading = "Execution traceback or artifact feedback"
        repair_instruction = (
            "Correct the execution/export failure while preserving all requirements "
            "from the original request."
        )
    elif repair_type == "constraint":
        failure_heading = (
            "The following CadQuery code executed and exported, but failed one or "
            "more automatic hard constraints."
        )
        feedback_heading = "Structured hard-constraint feedback"
        repair_instruction = (
            "Correct the measured geometric errors without removing requirements or "
            "breaking constraints that already pass."
        )
    else:
        raise ValueError(f"Unknown repair type: {repair_type}")

    return f"""{failure_heading}

Original CAD request:
{description}

Failed code:
```python
{previous_code}
```

{feedback_heading}:
```text
{feedback}
```
{reference_section}

{repair_instruction}
Please return a corrected complete CadQuery Python program, not a patch.
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
