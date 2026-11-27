"""Imported 3D models (USD / STEP / STL), placement configuration, and web API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from allmaptousd import step_import
from allmaptousd.model_assets import import_model, open_model_source
from allmaptousd.models import EquipmentPlacement
from allmaptousd.parsers import MapInputBundle
from allmaptousd.pipeline import MapSession
from allmaptousd.web import server as web_server

SAMPLE = Path(__file__).resolve().parents[1] / "sample_maps" / "sample_factory.yaml"


def _robot_usda(tmp_path: Path) -> bytes:
    """Y-up, centimetre file: two instanced 50 cm cubes on a 3 m x 1 m plate, 10 m from the origin.
    The cubes' material lives outside /Robot."""
    stage = Usd.Stage.CreateNew(str(tmp_path / "robot.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.Xform.Define(stage, "/Proto/Box")
    stage.GetPrimAtPath("/Proto").SetSpecifier(Sdf.SpecifierClass)
    cube = UsdGeom.Cube.Define(stage, "/Proto/Box/Geo")
    cube.CreateSizeAttr(50.0)
    material = UsdShade.Material.Define(stage, "/Looks/Red")
    shader = UsdShade.Shader.Define(stage, "/Looks/Red/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    UsdGeom.Xform.Define(stage, "/Robot").AddTranslateOp().Set((1000, 0, 0))
    for name, x in (("A", 0), ("B", 200)):
        part = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        part.AddTranslateOp().Set((x, 25, 0))
        part.GetPrim().GetReferences().AddInternalReference("/Proto")
        part.GetPrim().SetInstanceable(True)
    plate = UsdGeom.Mesh.Define(stage, "/Robot/Plate")
    plate.CreatePointsAttr([(-50, 0, -50), (250, 0, -50), (250, 0, 50), (-50, 0, 50)])
    plate.CreateFaceVertexCountsAttr([4])
    plate.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()
    return (tmp_path / "robot.usda").read_bytes()


def _box_stl(size_mm=(2000, 1000, 500), ascii=False) -> bytes:
    sx, sy, sz = size_mm
    v = np.array([[x, y, z] for z in (0, sz) for y in (0, sy) for x in (0, sx)], np.float32)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = v[[t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]]
    if ascii:
        facets = "".join(
            "facet normal 0 0 0\n outer loop\n" + "".join(f"  vertex {x} {y} {z}\n" for x, y, z in tri)
            + " endloop\nendfacet\n" for tri in tris
        )
        return f"solid box\n{facets}endsolid box\n".encode("ascii")
    record = np.zeros(len(tris), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    record["v"] = tris
    return b"\0" * 80 + len(tris).to_bytes(4, "little") + record.tobytes()


def test_usd_model_is_normalized_and_self_contained(tmp_path: Path) -> None:
    source = open_model_source("robot.usda", _robot_usda(tmp_path))
    info = source.describe()
    assert info["units"] == "cm" and info["up_axis"] == "y"
    assert [n["path"] for n in info["tree"]] == ["/Robot"]  # the class prototype and the material are not parts
    assert {c["name"] for c in info["tree"][0]["children"]} == {"A", "B", "Plate"}

    whole = import_model(source, [])
    assert whole.size == pytest.approx((3.0, 1.0, 0.5))  # metres, Z up
    assert len(whole.footprint) == 1
    lo, hi = whole.footprint[0].min(axis=0), whole.footprint[0].max(axis=0)
    assert lo == pytest.approx((-1.5, -0.5), abs=0.03) and hi == pytest.approx((1.5, 0.5), abs=0.03)

    stage = Usd.Stage.Open(whole.layer)
    cubes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Cube)]
    assert len(cubes) == 2 and not any(p.IsInstance() or p.IsInstanceProxy() for p in stage.Traverse())
    for cube in cubes:  # the material was copied in and the binding follows it
        bound = UsdShade.MaterialBindingAPI(cube).GetDirectBinding().GetMaterialPath()
        assert bound == Sdf.Path("/Model/Looks/Red") and stage.GetPrimAtPath(bound).IsA(UsdShade.Material)
        assert cube.HasAPI(UsdPhysics.CollisionAPI)
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPrimAtPath("/Model"))
    assert box.ComputeAlignedRange().GetMin()[2] == pytest.approx(0.0, abs=1e-6)

    part = import_model(source, ["/Robot/B"])
    assert part.name == "B" and part.size == pytest.approx((0.5, 0.5, 0.5))


def test_stl_model_units(tmp_path: Path) -> None:
    for ascii_stl in (False, True):
        model = import_model(open_model_source("box.stl", _box_stl(ascii=ascii_stl)), [])
        assert model.size == pytest.approx((2.0, 1.0, 0.5))
    in_metres = import_model(open_model_source("box.stl", _box_stl((2, 1, 0.5))), [], unit="m")
    assert in_metres.size == pytest.approx((2.0, 1.0, 0.5))
    assert "units" in import_model(open_model_source("box.stl", _box_stl((2, 1, 0.5))), []).summary()["warning"]
    with pytest.raises(ValueError):
        open_model_source("broken.stl", b"solid nothing here\nendsolid\n")


def test_placements_reset_on_map_change() -> None:
    session = MapSession()
    session.load(MapInputBundle.from_path(SAMPLE))
    box = import_model(open_model_source("box.stl", _box_stl()), [])
    session.models[box.id] = box
    placement = EquipmentPlacement(model_id=box.id, name="Box", x=3.0, y=4.0, yaw_deg=90.0)
    session.rebuild(session.config.model_copy(update={"equipment": [placement]}))
    assert session.config.equipment == [placement]
    session.load(MapInputBundle.from_path(SAMPLE))
    assert session.config.equipment == [] and session.models == {box.id: box}


def test_model_api(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(web_server, "EXPORT_DIR", tmp_path / "exports")
    monkeypatch.setattr(web_server, "state", web_server.AppState())
    client = TestClient(web_server.app, base_url="http://127.0.0.1:8765")
    assert client.post("/api/load-sample/ros_factory").status_code == 200

    bad = client.post("/api/models/inspect", files={"files": ("notes.txt", b"hello")})
    assert bad.status_code == 400
    info = client.post("/api/models/inspect", files={"files": ("robot.usda", _robot_usda(tmp_path))}).json()
    assert info["format"] == "usd" and info["tree"][0]["name"] == "Robot"
    model = client.post("/api/models/import", json={"token": info["token"], "paths": ["/Robot/A"]}).json()
    assert model["size"] == pytest.approx([0.5, 0.5, 0.5]) and model["footprint"]
    preview = client.get(f"/api/models/{model['id']}/preview").json()
    assert len(preview["positions"]) and len(preview["indices"])
    assert client.post("/api/models/import", json={"token": "gone", "paths": []}).status_code == 404

    assert [m["id"] for m in client.get("/api/models").json()] == [model["id"]]
    config = web_server.state.session.config.model_dump(mode="json")
    config["equipment"] = [{"model_id": model["id"], "name": "Cube", "x": 1.0, "y": 1.0, "yaw_deg": 0.0}]
    payload = client.post("/api/rebuild", json=config).json()
    assert [m["id"] for m in payload["models"]] == [model["id"]]
    assert payload["config"]["equipment"] == config["equipment"]

    assert client.delete(f"/api/models/{model['id']}").status_code == 200
    assert client.get("/api/models").json() == []


def _cell_step(tmp_path: Path) -> bytes:
    """A STEP assembly in mm: "Cell" = a 2000 x 1000 x 100 base and two red 800 mm posts (one shared part)."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool

    def name(label, text):
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(text))
        return label

    def at(x, y, z):
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(x, y, z))
        return TopLoc_Location(trsf)

    doc = TDocStd_Document(TCollection_ExtendedString("XmlOcaf"))
    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    cell = name(shapes.NewShape(), "Cell")
    base = name(shapes.AddShape(BRepPrimAPI_MakeBox(2000.0, 1000.0, 100.0).Shape(), False), "Base")
    post = name(shapes.AddShape(BRepPrimAPI_MakeCylinder(100.0, 800.0).Shape(), False), "Post")
    XCAFDoc_DocumentTool.ColorTool_s(doc.Main()).SetColor(
        post, Quantity_Color(0.9, 0.2, 0.1, Quantity_TOC_RGB), XCAFDoc_ColorSurf,
    )
    shapes.AddComponent(cell, base, at(0, 0, 0))
    name(shapes.AddComponent(cell, post, at(300, 500, 100)), "Post_Left")
    shapes.AddComponent(cell, post, at(1700, 500, 100))
    shapes.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(str(tmp_path / "cell.step"))
    return (tmp_path / "cell.step").read_bytes()


def test_step_model_assembly_tree_and_parts(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    source = open_model_source("cell.step", _cell_step(tmp_path))
    info = source.describe()
    assert info["format"] == "step" and info["units"] == "mm" and info["up_axis"] == "z"
    assert [n["path"] for n in info["tree"]] == ["/Cell"]
    assert [c["name"] for c in info["tree"][0]["children"]] == ["Base", "Post_Left", "Post"]

    whole = import_model(source, [])
    assert whole.size == pytest.approx((2.0, 1.0, 0.9), abs=1e-3)
    stage = Usd.Stage.Open(whole.layer)
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    assert len(meshes) == 3 and all(m.HasAPI(UsdPhysics.CollisionAPI) for m in meshes)
    red = UsdGeom.Mesh(stage.GetPrimAtPath("/Model/Cell/Post_Left")).GetDisplayColorAttr().Get()[0]
    assert tuple(red) == pytest.approx((0.9, 0.2, 0.1), abs=1e-3)

    post = import_model(source, ["/Cell/Post"])
    assert post.name == "Post" and post.size == pytest.approx((0.2, 0.2, 0.8), abs=2e-3)


def test_step_without_opencascade_explains_the_install(monkeypatch) -> None:
    monkeypatch.setattr(step_import, "step_support_available", lambda: False)
    with pytest.raises(ValueError, match=r"allmaptousd\[step\]"):
        open_model_source("part.stp", b"ISO-10303-21;")
