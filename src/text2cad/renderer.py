"""Render exported STL files to PNG previews with VTK."""

from __future__ import annotations

from pathlib import Path


def render_stl_to_png(stl_path: Path, png_path: Path, width: int = 900, height: int = 650) -> Path:
    """Render an STL file to a PNG preview."""
    import vtk

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.8, 0.84, 0.9)
    actor.GetProperty().SetSpecular(0.35)
    actor.GetProperty().SetSpecularPower(25)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(1, -1, 2)
    light.SetFocalPoint(0, 0, 0)
    renderer.AddLight(light)

    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(width, height)

    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Azimuth(35)
    camera.Elevation(25)
    renderer.ResetCameraClippingRange()

    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(png_path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    return png_path
