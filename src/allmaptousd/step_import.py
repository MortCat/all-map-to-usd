"""STEP (.step / .stp) files as USD stages, read through OpenCASCADE.

STEP support is a pip extra (``allmaptousd[step]``) because OpenCASCADE is a large download;
``start_app.cmd`` installs it, but the app still runs if that install fails. The STEP assembly
tree becomes a prim tree (assemblies and component placements as Xforms, parts as triangle
meshes with their STEP colours), so the import dialog and the USD import path handle STEP exactly like a USD file.
OpenCASCADE reads STEP lengths in millimetres.
"""

from __future__ import annotations

import importlib.util
import math
import re

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

STEP_EXTENSIONS = (".step", ".stp")
STEP_INSTALL_HINT = (
    "STEP support requires a compatible OpenCASCADE 7 installation. Close the app and run start_app.cmd --force-setup, "
    'or run pip install "allmaptousd[step]", then start the app again.'
)


def step_support_available() -> bool:
    return importlib.util.find_spec("OCP") is not None


def step_to_stage(path: str) -> Usd.Stage:
    """Read a STEP file into an in-memory USD stage (Z up, millimetres)."""
    if not step_support_available():
        raise ValueError(STEP_INSTALL_HINT)
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDF import TDF_LabelSequence
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except (ImportError, OSError) as exc:
        raise ValueError(STEP_INSTALL_HINT) from exc

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(path) != IFSelect_RetDone or not reader.Transfer(doc):
        raise ValueError("This STEP file could not be read.")
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    roots = TDF_LabelSequence()
    shapes.GetFreeShapes(roots)
    labels = [roots.Value(i) for i in range(1, roots.Length() + 1)]
    if not labels:
        raise ValueError("This STEP file has no shapes.")

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.001)
    writer = _StageWriter(stage, shapes, colors, _deflection(shapes, labels))
    used: set[str] = set()
    for label in labels:
        writer.add(label, None, Sdf.Path.absoluteRootPath, used)
    if not writer.part_count:
        raise ValueError("This STEP file has no solid or surface geometry.")
    return stage


def _deflection(shapes, labels) -> float:
    """Meshing tolerance: 0.05 % of the model's diagonal, at least 0.01 mm."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    for label in labels:
        BRepBndLib.Add_s(shapes.GetShape_s(label), box)
    if box.IsVoid():
        return 0.1
    x0, y0, z0, x1, y1, z1 = box.Get()
    return max(math.dist((x0, y0, z0), (x1, y1, z1)) * 5e-4, 0.01)


def _label_name(label) -> str:
    from OCP.TDataStd import TDataStd_Name

    attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attr):
        return attr.Get().ToExtString().strip()
    return ""


def _prim_name(name: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not clean or clean.startswith("_") and not clean.strip("_"):
        clean = fallback
    if clean[0].isdigit():
        clean = f"{fallback}_{clean}"
    return clean


def unique_name(name: str, used: set[str]) -> str:
    """``name``, or ``name_2``, ``name_3``, ... when it is taken; the result is added to ``used``."""
    candidate, i = name, 2
    while candidate in used:
        candidate, i = f"{name}_{i}", i + 1
    used.add(candidate)
    return candidate


def _trsf_matrix(trsf) -> Gf.Matrix4d:
    """gp_Trsf (column vectors) as a USD row-vector matrix."""
    rows = [[trsf.Value(r, c) for r in (1, 2, 3)] + [0.0] for c in (1, 2, 3)]  # row i: image of axis i
    rows.append([trsf.Value(1, 4), trsf.Value(2, 4), trsf.Value(3, 4), 1.0])
    return Gf.Matrix4d(*(v for row in rows for v in row))


class _StageWriter:
    def __init__(self, stage: Usd.Stage, shapes, colors, deflection: float) -> None:
        self.stage, self.shapes, self.colors, self.deflection = stage, shapes, colors, deflection
        self.meshes: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}  # referred label -> mesh
        self.part_count = 0

    def add(self, label, placement, parent: Sdf.Path, used: set[str]) -> None:
        """Write ``label`` (a component reference, an assembly or a part) under ``parent``."""
        from OCP.TDF import TDF_Label, TDF_LabelSequence

        referred = label
        if self.shapes.IsReference_s(label):
            referred = TDF_Label()
            self.shapes.GetReferredShape_s(label, referred)
        own_name = _label_name(label)
        if not own_name or own_name.startswith("=>") or re.fullmatch(r"NAUO\d*", own_name):
            own_name = _label_name(referred)  # component names are often generic; use the part's
        is_assembly = self.shapes.IsAssembly_s(referred)
        name = unique_name(_prim_name(own_name, "Assembly" if is_assembly else "Part"), used)
        path = parent.AppendChild(name)

        if is_assembly:
            prim = UsdGeom.Xform.Define(self.stage, path)
            components = TDF_LabelSequence()
            self.shapes.GetComponents_s(referred, components, False)
            child_names: set[str] = set()
            for i in range(1, components.Length() + 1):
                component = components.Value(i)
                self.add(component, self.shapes.GetLocation_s(component).Transformation(), path, child_names)
        else:
            mesh = self._mesh(referred)
            if mesh is None:
                used.discard(name)
                return
            points, triangles = mesh
            prim = UsdGeom.Mesh.Define(self.stage, path)
            prim.CreatePointsAttr(points)
            prim.CreateFaceVertexCountsAttr(np.full(len(triangles), 3, np.int32))
            prim.CreateFaceVertexIndicesAttr(triangles.ravel())
            prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            prim.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
            prim.CreateDisplayColorAttr([Gf.Vec3f(*self._color(label, referred))])
            self.part_count += 1
        if placement is not None and placement.Form() != 0:  # gp_Identity is form 0
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(_trsf_matrix(placement))

    def _color(self, *labels) -> tuple[float, float, float]:
        from OCP.Quantity import Quantity_Color
        from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool

        color = Quantity_Color()
        for label in labels:
            for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
                if XCAFDoc_ColorTool.GetColor_s(label, kind, color):
                    return color.Red(), color.Green(), color.Blue()
        return 0.62, 0.66, 0.72

    def _mesh(self, label) -> tuple[np.ndarray, np.ndarray] | None:
        """Triangulate a part once (shared parts reuse it), in the part's own coordinates."""
        key = label.EntryDumpToString() if hasattr(label, "EntryDumpToString") else str(label.Tag())
        if key not in self.meshes:
            self.meshes[key] = self._triangulate(self.shapes.GetShape_s(label))
        return self.meshes[key]

    def _triangulate(self, shape) -> tuple[np.ndarray, np.ndarray] | None:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS

        BRepMesh_IncrementalMesh(shape, self.deflection, False, 0.5, True)
        all_points: list[np.ndarray] = []
        all_triangles: list[np.ndarray] = []
        offset = 0
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            explorer.Next()
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is None or tri.NbTriangles() == 0:
                continue
            trsf = location.Transformation()
            points = np.array(
                [tri.Node(i).Transformed(trsf).Coord() for i in range(1, tri.NbNodes() + 1)], np.float64,
            )
            triangles = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], np.int64) - 1
            if face.Orientation() == TopAbs_REVERSED:
                triangles = triangles[:, [0, 2, 1]]
            all_points.append(points)
            all_triangles.append(triangles + offset)
            offset += len(points)
        if not all_points:
            return None
        return np.concatenate(all_points).astype(np.float32), np.concatenate(all_triangles).astype(np.int32)
