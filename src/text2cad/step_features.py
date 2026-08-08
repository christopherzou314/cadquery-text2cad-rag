"""Extract exact solid and cylindrical-hole features from STEP geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any


AXIS_VECTORS = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


def read_step_features(step_path: Path) -> dict[str, Any]:
    """Import one STEP file and return JSON-serializable topology features."""
    try:
        from cadquery import importers

        shape = importers.importStep(str(step_path)).val()
        return extract_shape_features(shape)
    except Exception as exc:
        return empty_step_features(f"{type(exc).__name__}: {exc}")


def extract_shape_features(shape) -> dict[str, Any]:
    """Extract features from a CadQuery Shape or Compound."""
    try:
        solids = list(shape.Solids())
        holes = []
        for solid_index, solid in enumerate(solids):
            holes.extend(_cylindrical_holes(solid, solid_index))
        holes = _deduplicate_holes(holes)
        return {
            "step_valid": bool(solids) and shape.isValid(),
            "solid_count": len(solids),
            "cylindrical_hole_count": len(holes),
            "cylindrical_holes": holes,
            "error": None,
        }
    except Exception as exc:
        return empty_step_features(f"{type(exc).__name__}: {exc}")


def empty_step_features(error: str | None = None) -> dict[str, Any]:
    return {
        "step_valid": False,
        "solid_count": None,
        "cylindrical_hole_count": 0,
        "cylindrical_holes": [],
        "error": error,
    }


def _cylindrical_holes(solid, solid_index: int) -> list[dict[str, Any]]:
    import cadquery as cq
    from OCP.BRepAdaptor import BRepAdaptor_Surface

    holes = []
    for face_index, face in enumerate(solid.Faces()):
        if face.geomType() != "CYLINDER":
            continue
        adaptor = BRepAdaptor_Surface(face.wrapped, True)
        cylinder = adaptor.Cylinder()
        axis = cylinder.Axis()
        axis_location = cq.Vector(
            axis.Location().X(), axis.Location().Y(), axis.Location().Z()
        )
        direction = cq.Vector(
            axis.Direction().X(), axis.Direction().Y(), axis.Direction().Z()
        ).normalized()
        u = (adaptor.FirstUParameter() + adaptor.LastUParameter()) / 2
        v = (adaptor.FirstVParameter() + adaptor.LastVParameter()) / 2
        sample = adaptor.Value(u, v)
        sample_point = cq.Vector(sample.X(), sample.Y(), sample.Z())
        radial = sample_point - axis_location
        radial = radial - direction.multiply(radial.dot(direction))
        if radial.Length < 1e-9:
            continue
        normal = face.normalAt(sample_point)
        normal_dot_radial = normal.dot(radial.normalized())
        if normal_dot_radial >= -0.5:
            continue

        projections = [
            (vertex.Center() - axis_location).dot(direction)
            for vertex in face.Vertices()
        ]
        if projections:
            axial_min, axial_max = min(projections), max(projections)
        else:
            axial_min = float(adaptor.FirstVParameter())
            axial_max = float(adaptor.LastVParameter())
        axial_length = abs(axial_max - axial_min)
        center = axis_location + direction.multiply((axial_min + axial_max) / 2)
        canonical_direction = _canonical_direction(direction)
        axis_label = _axis_label(canonical_direction)
        through, probe_volume, through_error = _is_through_hole(
            solid,
            radius=float(cylinder.Radius()),
            axis_location=axis_location,
            direction=direction,
            axial_min=axial_min,
            axial_max=axial_max,
        )
        holes.append(
            {
                "solid_index": solid_index,
                "face_index": face_index,
                "diameter_mm": round(float(cylinder.Radius()) * 2, 6),
                "radius_mm": round(float(cylinder.Radius()), 6),
                "axis": axis_label,
                "axis_direction": _vector_dict(canonical_direction),
                "center_mm": _vector_dict(center),
                "axial_length_mm": round(axial_length, 6),
                "through": through,
                "through_probe_intersection_volume_mm3": probe_volume,
                "through_check_error": through_error,
                "internal_normal_dot": round(normal_dot_radial, 6),
            }
        )
    return holes


def _is_through_hole(
    solid,
    *,
    radius: float,
    axis_location,
    direction,
    axial_min: float,
    axial_max: float,
) -> tuple[bool | None, float | None, str | None]:
    try:
        import cadquery as cq

        axial_length = abs(axial_max - axial_min)
        if axial_length < 1e-6:
            return None, None, "Cylindrical face has no measurable axial length."
        margin = max(0.2, min(1.0, axial_length * 0.05))
        probe_radius = max(0.02, min(0.25, radius * 0.2))
        origin = axis_location + direction.multiply(axial_min - margin)
        probe = cq.Solid.makeCylinder(
            probe_radius,
            axial_length + 2 * margin,
            origin,
            direction,
        )
        intersection_volume = float(solid.intersect(probe).Volume())
        tolerance = max(1e-8, float(probe.Volume()) * 1e-7)
        return (
            intersection_volume <= tolerance,
            round(intersection_volume, 9),
            None,
        )
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _canonical_direction(direction):
    components = direction.toTuple()
    dominant = max(range(3), key=lambda index: abs(components[index]))
    return direction.multiply(-1) if components[dominant] < 0 else direction


def _axis_label(direction, tolerance: float = 0.995) -> str | None:
    components = direction.toTuple()
    for axis, expected in AXIS_VECTORS.items():
        dot = sum(a * b for a, b in zip(components, expected, strict=True))
        if abs(dot) >= tolerance:
            return axis
    return None


def _vector_dict(vector) -> dict[str, float]:
    return {
        axis: round(float(value), 6)
        for axis, value in zip(("x", "y", "z"), vector.toTuple(), strict=True)
    }


def _deduplicate_holes(holes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for hole in holes:
        key = (
            hole["solid_index"],
            round(hole["diameter_mm"], 4),
            hole["axis"],
            tuple(round(value, 4) for value in hole["center_mm"].values()),
            round(hole["axial_length_mm"], 4),
        )
        if key not in seen:
            seen.add(key)
            unique.append(hole)
    return unique
