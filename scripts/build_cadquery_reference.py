"""Build a comprehensive local RAG index from the installed CadQuery API."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import cadquery as cq


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge"
API_OUTPUT = KNOWLEDGE_DIR / "cadquery_api_reference.json"
GUIDES_OUTPUT = KNOWLEDGE_DIR / "cadquery_official_guides.json"
MANIFEST_OUTPUT = KNOWLEDGE_DIR / "cadquery_reference_manifest.json"

CLASS_REFERENCE = "https://cadquery.readthedocs.io/en/latest/classreference.html"
WORKPLANE_GUIDE = "https://cadquery.readthedocs.io/en/latest/workplane.html"
SKETCH_GUIDE = "https://cadquery.readthedocs.io/en/latest/sketch.html"
ASSEMBLY_GUIDE = "https://cadquery.readthedocs.io/en/latest/assy.html"
EXAMPLES_GUIDE = "https://cadquery.readthedocs.io/en/latest/examples.html"
FREE_FUNCTION_GUIDE = "https://cadquery.readthedocs.io/en/latest/free-func.html"

CLASS_NAMES = [
    "Assembly",
    "BoundBox",
    "Color",
    "Compound",
    "Constraint",
    "Edge",
    "Face",
    "Location",
    "Material",
    "Matrix",
    "Plane",
    "Shape",
    "Shell",
    "Sketch",
    "Solid",
    "Vertex",
    "Vector",
    "Wire",
    "Workplane",
    "Selector",
    "StringSyntaxSelector",
    "TypeSelector",
    "DirectionSelector",
    "ParallelDirSelector",
    "PerpendicularDirSelector",
    "DirectionMinMaxSelector",
    "NearestToPointSelector",
]

STOPWORDS = {
    "about", "after", "also", "been", "being", "between", "class", "from",
    "into", "method", "object", "other", "return", "returns", "self", "shape",
    "that", "their", "then", "these", "this", "using", "value", "when", "where",
    "which", "will", "with",
}
GENERIC_METHOD_WORDS = {"add", "from", "get", "make", "remove", "set", "to"}


def main() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    api_entries = build_api_entries()
    guide_entries = build_guide_entries()

    API_OUTPUT.write_text(
        json.dumps(api_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    GUIDES_OUTPUT.write_text(
        json.dumps(guide_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "cadquery_version": getattr(cq, "__version__", "unknown"),
        "api_entry_count": len(api_entries),
        "guide_entry_count": len(guide_entries),
        "generated_from": "Installed CadQuery signatures and docstrings",
        "official_sources": [
            WORKPLANE_GUIDE,
            SKETCH_GUIDE,
            ASSEMBLY_GUIDE,
            CLASS_REFERENCE,
            EXAMPLES_GUIDE,
            FREE_FUNCTION_GUIDE,
        ],
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generated {len(api_entries)} API entries and {len(guide_entries)} guide entries.")


def build_api_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for class_name in CLASS_NAMES:
        cls = getattr(cq, class_name)
        entries.append(class_entry(class_name, cls))

        for name, raw_member in sorted(cls.__dict__.items()):
            if name.startswith("_"):
                continue
            member = unwrap_member(raw_member)
            if member is None:
                continue
            entries.append(member_entry(class_name, name, member))

    module_targets: list[tuple[str, ModuleType, str]] = [
        ("exporters", cq.exporters, CLASS_REFERENCE),
        ("importers", cq.importers, CLASS_REFERENCE),
        ("free_function", cq.occ_impl.shapes, FREE_FUNCTION_GUIDE),
    ]
    for label, module, source_url in module_targets:
        for name, member in inspect.getmembers(module):
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            if member.__module__ != module.__name__:
                continue
            entries.append(function_entry(label, name, member, source_url))

    return entries


def class_entry(class_name: str, cls: type) -> dict[str, Any]:
    signature = safe_signature(cls)
    doc = clean_doc(inspect.getdoc(cls))
    return make_entry(
        entry_id=f"api.{class_name}",
        title=f"API class cq.{class_name}{signature}",
        keywords=keywords_for(class_name, class_name, doc),
        content=f"Installed API signature: cq.{class_name}{signature}\n\n{doc}",
        source_url=f"{CLASS_REFERENCE}#cadquery.{class_name}",
    )


def member_entry(class_name: str, name: str, member: Any) -> dict[str, Any]:
    signature = safe_signature(member)
    doc = clean_doc(inspect.getdoc(member))
    return make_entry(
        entry_id=f"api.{class_name}.{name}",
        title=f"API cq.{class_name}.{name}{signature}",
        keywords=keywords_for(class_name, name, doc),
        content=f"Installed API signature: cq.{class_name}.{name}{signature}\n\n{doc}",
        source_url=f"{CLASS_REFERENCE}#cadquery.{class_name}.{name}",
    )


def function_entry(label: str, name: str, member: Any, source_url: str) -> dict[str, Any]:
    signature = safe_signature(member)
    doc = clean_doc(inspect.getdoc(member))
    qualified_name = f"cq.{label}.{name}" if label != "free_function" else name
    return make_entry(
        entry_id=f"api.{label}.{name}",
        title=f"API {qualified_name}{signature}",
        keywords=keywords_for(label, name, doc),
        content=f"Installed API signature: {qualified_name}{signature}\n\n{doc}",
        source_url=source_url,
    )


def make_entry(
    *,
    entry_id: str,
    title: str,
    keywords: list[str],
    content: str,
    source_url: str,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "title": title,
        "category": "api_reference",
        "keywords": keywords,
        "search_terms": search_terms_for(content),
        "content": content,
        "example": "",
        "source_url": source_url,
    }


def unwrap_member(raw_member: Any) -> Any | None:
    if isinstance(raw_member, (staticmethod, classmethod)):
        return raw_member.__func__
    if isinstance(raw_member, property):
        return raw_member.fget
    return raw_member if callable(raw_member) else None


def safe_signature(member: Any) -> str:
    try:
        return str(inspect.signature(member))
    except (TypeError, ValueError):
        return "(...)"


def clean_doc(doc: str | None) -> str:
    if not doc:
        return "No installed docstring is available; consult the linked official API reference."
    return "\n".join(line.rstrip() for line in doc.strip().splitlines())


def keywords_for(owner: str, name: str, doc: str) -> list[str]:
    words = split_identifier(owner)
    words.append(name.lower())
    words.extend(
        word
        for word in split_identifier(name)
        if word.lower() not in GENERIC_METHOD_WORDS and word.lower() not in STOPWORDS
    )
    return list(dict.fromkeys(word.lower() for word in words))


def search_terms_for(content: str) -> list[str]:
    first_paragraph = content.split("\n\n", 1)[-1].split("\n\n", 1)[0]
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", first_paragraph.lower())
    return list(dict.fromkeys(word for word in words if word not in STOPWORDS))[:30]


def split_identifier(value: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [token for token in re.split(r"[^A-Za-z0-9]+", spaced) if token]


def build_guide_entries() -> list[dict[str, Any]]:
    topics = [
        ("workplane.coordinates", "Workplane local coordinates", ["workplane", "plane", "coordinates", "origin"], "Workplanes define a local 2D coordinate system in 3D space. Features are created relative to the active plane; face-based workplanes reduce manual global-coordinate calculations.", WORKPLANE_GUIDE),
        ("workplane.2d", "Workplane 2D construction", ["line", "arc", "circle", "rect", "polyline", "profile", "2d"], "Use lines, arcs, circles, rectangles, polygons, splines and points to create closed wires or construction geometry before a 3D operation.", WORKPLANE_GUIDE),
        ("workplane.3d", "Workplane 3D construction", ["box", "cylinder", "sphere", "extrude", "revolve", "sweep", "loft", "3d"], "Create direct primitives or turn pending 2D wires into solids with extrude, revolve, sweep, loft and related operations.", WORKPLANE_GUIDE),
        ("workplane.selectors", "Workplane selectors", ["selector", "faces", "edges", "vertices", "wires", "solids"], "Selectors filter topology such as faces, edges, vertices, wires and solids. Direction selectors like >Z and type selectors can drive subsequent features.", WORKPLANE_GUIDE),
        ("workplane.construction", "Construction geometry", ["construction", "forconstruction", "vertices", "pattern", "holes"], "Construction geometry locates features without becoming part of the final profile. A construction rectangle followed by vertices is a common hole-pattern technique.", WORKPLANE_GUIDE),
        ("workplane.stack", "Workplane stack and iteration", ["stack", "each", "all", "first", "last", "end", "iteration"], "Each chained operation returns a Workplane with stack objects and a parent. Many methods apply automatically to every location or object on the stack.", WORKPLANE_GUIDE),
        ("workplane.chaining", "Workplane chaining and tags", ["chain", "parent", "tag", "workplanefromtagged"], "Fluent chains preserve parent links and shared modelling context. Tags let later operations return to a named state or select from it.", WORKPLANE_GUIDE),
        ("workplane.context", "Context solid and pending wires", ["context", "solid", "pending", "topending", "combine"], "The first solid becomes the context solid. Later additive or subtractive operations search the parent chain for it; combine=False keeps new solids separate. Pending wires feed 3D operations.", WORKPLANE_GUIDE),
        ("sketch.face_api", "Sketch face-based API and modes", ["sketch", "face", "mode", "add", "subtract", "intersect", "replace", "construction"], "Sketch face operations combine geometry in-place using additive, subtractive, intersect, replace or construction modes. Sketch selection must be reset explicitly because Sketch has no history chain.", SKETCH_GUIDE),
        ("sketch.edge_api", "Sketch edge-based API", ["sketch", "segment", "arc", "edge", "assemble", "close"], "Build individual segments and arcs, close or connect them, then call assemble() to convert edge geometry into faces before face operations.", SKETCH_GUIDE),
        ("sketch.hull", "Sketch convex hull", ["sketch", "hull", "convex", "circle", "segment"], "The experimental hull operation creates a convex hull for supported straight segments and circles.", SKETCH_GUIDE),
        ("sketch.constraints", "Sketch constraints and solver", ["sketch", "constraint", "solve", "coincident", "angle", "length", "distance", "radius", "orientation"], "Experimental sketch constraints support fixed points, coincidence, angle, length, distance, radius, orientation and arc angle for segments and arcs; solve before assemble.", SKETCH_GUIDE),
        ("sketch.workplane", "Sketch and Workplane integration", ["sketch", "finalize", "placesketch", "extrude", "revolve", "sweep", "loft"], "Create sketches in-place with Workplane.sketch().finalize(), or reuse an existing Sketch with placeSketch(). Sketches can drive extrude, twistExtrude, revolve, sweep, cuts and loft.", SKETCH_GUIDE),
        ("sketch.multi", "Sketch reuse, offsets, import and export", ["sketch", "multiple", "offset", "export", "import", "dxf"], "Sketches can be placed at multiple stack locations, combined, offset, lofted between, and imported or exported for reuse.", SKETCH_GUIDE),
        ("assembly.parameters", "Assembly parameters and reusable components", ["assembly", "parameter", "component", "function", "reusable"], "Define shared dimensions first and create reusable component functions. Add named component instances to a cq.Assembly so parts remain identifiable.", ASSEMBLY_GUIDE),
        ("assembly.locations", "Assembly object locations", ["assembly", "location", "translate", "rotate", "position"], "Parts may be placed explicitly with Location values. Explicit placement is often the simplest robust approach when exact constraint solving is unnecessary.", ASSEMBLY_GUIDE),
        ("assembly.constraints", "Assembly constraints", ["assembly", "constraint", "point", "axis", "plane", "fixed", "solve"], "Assembly constraints include Point, Axis, Plane, PointInPlane, PointOnLine, FixedPoint, FixedRotation and FixedAxis. Name or tag mating topology, constrain it, then solve.", ASSEMBLY_GUIDE),
        ("assembly.colors", "Assembly colors and hierarchy", ["assembly", "color", "nested", "hierarchy", "subassembly"], "Assembly items and nested subassemblies can have names, colors, materials, metadata and relative locations.", ASSEMBLY_GUIDE),
        ("assembly.export", "Assembly export", ["assembly", "export", "step", "xml", "gltf", "compound"], "Assemblies can be converted to compounds or exported in supported assembly-aware formats. Preserve names and hierarchy when the chosen format supports them.", ASSEMBLY_GUIDE),
    ]

    example_topics = [
        ("simple_plate", ["box", "plate", "primitive"], "Create a basic rectangular plate with Workplane.box."),
        ("plate_with_hole", ["plate", "hole", "faces", "workplane"], "Select the top face, create a workplane and drill a centered through-hole."),
        ("extruded_prism", ["circle", "rect", "extrude", "profile"], "Combine closed 2D profiles and extrude them into a prismatic solid."),
        ("lines_and_arcs", ["lineto", "threepointarc", "close", "profile"], "Build a closed profile using lines and arcs, then extrude it."),
        ("moving_workpoint", ["center", "move", "working", "point"], "Move the local working point to place multiple profile features."),
        ("point_lists", ["pushpoints", "pattern", "points"], "Push multiple locations onto the stack so later operations repeat at each point."),
        ("polygons", ["polygon", "cutthruall"], "Create polygonal profiles at stack points and use them for cuts or solids."),
        ("polylines", ["polyline", "mirror", "beam"], "Build point-driven profiles with polyline and mirror symmetric halves."),
        ("spline_edge", ["spline", "curve", "profile"], "Use spline points for smooth complex profile edges."),
        ("mirroring", ["mirror", "symmetric", "union"], "Mirror 2D profiles, 3D objects, or geometry about selected faces and optionally union results."),
        ("face_workplanes", ["faces", "workplane", "vertex", "offset", "rotated", "copyworkplane"], "Create, offset, copy, rotate or locate workplanes from selected faces and vertices."),
        ("construction_geometry", ["construction", "forconstruction", "holes"], "Use construction profiles and their vertices to locate repeated features."),
        ("shelling", ["shell", "thin", "wall", "hollow"], "Shell solids inward or outward and select faces to remove for openings."),
        ("lofts", ["loft", "section", "transition"], "Loft through multiple closed section wires to create changing cross-sections."),
        ("sweeps", ["sweep", "path", "profile"], "Sweep a profile along a path, choosing transitions appropriate to the geometry."),
        ("blind_and_projected_cuts", ["cutblind", "cutthruall", "project", "next"], "Cut to a fixed depth, through all, or to a target face while ensuring projection is valid."),
        ("special_holes", ["cborehole", "cskhole", "counterbore", "countersink"], "Use built-in counterbore and countersink operations at one or many stack locations."),
        ("offset_profiles", ["offset2d", "topending", "wire"], "Move selected edges to pending geometry before applying 2D offsets."),
        ("edge_finishing", ["fillet", "chamfer", "edges", "round"], "Select suitable edges before applying fillets or chamfers; radii must fit local geometry."),
        ("text", ["text", "engrave", "emboss"], "Create text profiles and combine or cut them with solids."),
        ("occ_bottle", ["bottle", "profile", "mirror", "extrude", "shell", "thread"], "The classic bottle combines a mirrored profile, extrusion, neck feature and shelling."),
        ("parametric_enclosure", ["enclosure", "parameter", "shell", "boss", "screw"], "A parameterized enclosure combines shells, rounded edges, screw posts, cuts and repeated features."),
        ("lego_brick", ["lego", "brick", "parameter", "shell", "array"], "A parametric brick derives body dimensions, shell thickness, studs and underside posts from bump counts."),
        ("braille", ["braille", "text", "sphere", "pattern"], "Map text to point patterns and construct raised dot geometry parametrically."),
        ("connector_panel", ["panel", "connector", "hole", "arc", "pattern"], "Construct a panel containing repeated complex connector cutout profiles."),
        ("cycloidal_gear", ["gear", "cycloidal", "parametriccurve", "twistextrude"], "Generate a cycloidal profile from a parametric curve and twist-extrude it into a helical gear."),
    ]
    entries = [guide_entry(*topic, category="official_guide") for topic in topics]
    entries.extend(
        guide_entry(
            f"examples.{entry_id}",
            f"Official example topic: {entry_id.replace('_', ' ').title()}",
            keywords,
            summary,
            EXAMPLES_GUIDE,
            category="official_example",
        )
        for entry_id, keywords, summary in example_topics
    )
    return entries


def guide_entry(
    entry_id: str,
    title: str,
    keywords: list[str],
    content: str,
    source_url: str,
    *,
    category: str,
) -> dict[str, Any]:
    return {
        "id": entry_id,
        "title": title,
        "category": category,
        "keywords": keywords,
        "content": content,
        "example": "",
        "source_url": source_url,
    }


if __name__ == "__main__":
    main()
