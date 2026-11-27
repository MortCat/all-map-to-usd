"""3D models keep their textures and materials: folder / zip / USDZ uploads, the browser preview,
USD export, the project save, and Push Models Only."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdUtils
from test_isaac_models import isaac  # noqa: F401  (fixture)

from allmaptousd.exporters.ros_map import export_ros_map_zip
from allmaptousd.model_assets import ASSET_DIR, import_model, open_model_files, open_model_source
from allmaptousd.models import BuildConfig, EquipmentPlacement, IsaacPullConfig, IsaacSimConfig
from allmaptousd.parsers import MapInputBundle
from allmaptousd.pipeline import MapSession
from allmaptousd.web import server

MDL = """mdl 1.6;
import ::tex::*;
export material Painted(uniform texture_2d grain = texture_2d("../textures/grain.png")) = material();
"""


def _png(bgr: tuple[int, int, int]) -> bytes:
    image = np.zeros((8, 8, 3), np.uint8)
    image[:] = bgr
    return cv2.imencode(".png", image)[1].tobytes()


def _crate(folder: Path) -> Path:
    """Crate/crate.usda: a textured top (UsdPreviewSurface), a blue lid (OmniPBR colour) and a body
    using a custom MDL that names its own texture. Files sit in textures/ and materials/."""
    (folder / "textures").mkdir(parents=True)
    (folder / "materials").mkdir()
    (folder / "textures" / "wood.png").write_bytes(_png((0, 0, 255)))  # red
    (folder / "textures" / "grain.png").write_bytes(_png((0, 255, 0)))
    (folder / "materials" / "Painted.mdl").write_text(MDL)
    (folder / "notes.txt").write_text("not used by the model")

    stage = Usd.Stage.CreateNew(str(folder / "crate.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Crate").GetPrim())
    top = UsdGeom.Mesh.Define(stage, "/Crate/Top")
    top.CreatePointsAttr([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    top.CreateFaceVertexCountsAttr([4])
    top.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(top).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    lid = UsdGeom.Cube.Define(stage, "/Crate/Lid")
    UsdGeom.XformCommonAPI(lid).SetTranslate(Gf.Vec3d(0.5, 0.5, 0.5))
    UsdGeom.XformCommonAPI(lid).SetScale(Gf.Vec3f(0.5, 0.5, 0.5))
    body = UsdGeom.Cube.Define(stage, "/Crate/Body")
    UsdGeom.XformCommonAPI(body).SetTranslate(Gf.Vec3d(2.5, 0.5, 0.5))
    UsdGeom.XformCommonAPI(body).SetScale(Gf.Vec3f(0.5, 0.5, 0.5))

    wood = UsdShade.Material.Define(stage, "/Looks/Wood")
    surface = UsdShade.Shader.Define(stage, "/Looks/Wood/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, "/Looks/Wood/St")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, "/Looks/Wood/Tex")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("./textures/wood.png")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    wood.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(wood)

    def mdl_material(path: str, source: str, sub_id: str, **inputs) -> UsdShade.Material:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.SetSourceAsset(Sdf.AssetPath(source), "mdl")
        shader.SetSourceAssetSubIdentifier(sub_id, "mdl")
        for name, (kind, value) in inputs.items():
            shader.CreateInput(name, kind).Set(value)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        return material

    blue = mdl_material("/Looks/Blue", "OmniPBR.mdl", "OmniPBR",
                        diffuse_color_constant=(Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0, 0, 1)))
    painted = mdl_material("/Looks/Painted", "./materials/Painted.mdl", "Painted")
    UsdShade.MaterialBindingAPI.Apply(lid.GetPrim()).Bind(blue)
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(painted)
    stage.Save()
    return folder / "crate.usda"


def _folder_upload(root: Path) -> dict[str, bytes]:
    return {f"{root.name}/{p.relative_to(root).as_posix()}": p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _asset_paths(layer: Sdf.Layer) -> list[str]:
    stage = Usd.Stage.Open(layer)
    found = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                found.append(value.path)
    return sorted(found)


def test_folder_upload_bundles_textures_and_materials(tmp_path: Path) -> None:
    _crate(tmp_path / "Crate")
    source = open_model_files(_folder_upload(tmp_path / "Crate"))
    assert source.filename == "Crate/crate.usda" and source.missing == []
    asset = import_model(source, [])
    source.close()

    assert sorted(asset.files) == [
        "Crate/materials/Painted.mdl", "Crate/textures/grain.png", "Crate/textures/wood.png",  # not notes.txt
    ]
    prefix = f"./{ASSET_DIR}/{asset.id}/Crate/"
    assert _asset_paths(asset.layer) == [f"{prefix}materials/Painted.mdl", f"{prefix}textures/wood.png", "OmniPBR.mdl"]

    mesh = asset.preview_mesh
    assert [g[2] for g in mesh.groups] == [-1, 0] and len(mesh.textures) == 1  # the top is textured
    colors = {tuple(c) for c in mesh.colors.tolist()}
    assert (0, 0, 255) in colors  # the OmniPBR lid, blue
    assert (255, 255, 255) in colors  # textured top: white, times the texture
    texture = cv2.imdecode(np.frombuffer(mesh.textures[0], np.uint8), cv2.IMREAD_COLOR)
    assert texture[0, 0].tolist() == [0, 0, 255]  # the red wood image (BGR)
    preview = asset.preview()
    assert preview["groups"] == [[0, 24, -1], [24, 2, 0]] and preview["textures"] == 1


def test_zip_and_usdz_uploads_and_missing_files(tmp_path: Path) -> None:
    usda = _crate(tmp_path / "Crate")
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as zf:
        for name, data in _folder_upload(tmp_path / "Crate").items():
            zf.writestr(name, data)
    from_zip = import_model(open_model_source("crate.zip", packed.getvalue()), [])
    assert len(from_zip.files) == 3 and from_zip.missing == []

    UsdUtils.CreateNewUsdzPackage(str(usda), str(tmp_path / "crate.usdz"))
    from_usdz = import_model(open_model_source("crate.usdz", (tmp_path / "crate.usdz").read_bytes()), [])
    assert "crate/textures/wood.png" in from_usdz.files and len(from_usdz.preview_mesh.textures) == 1

    lone = open_model_source("crate.usda", usda.read_bytes())
    assert lone.missing == ["materials/Painted.mdl", "textures/wood.png"]  # OmniPBR.mdl comes with Isaac Sim
    asset = import_model(lone, [])
    assert asset.files == {} and asset.missing == ["materials/Painted.mdl", "textures/wood.png"]
    assert asset.summary()["missing_files"] == asset.missing
    # where the file would go, so it can still be added next to an export
    assert f"./{ASSET_DIR}/{asset.id}/textures/wood.png" in _asset_paths(asset.layer)
    assert {tuple(c) for c in asset.preview_mesh.colors.tolist()} >= {(0, 0, 255)}

    with pytest.raises(ValueError, match="not a valid"):
        open_model_files({"../escape.usda": usda.read_bytes()})


def test_main_file_of_a_folder(tmp_path: Path) -> None:
    root = tmp_path / "Line"
    (root / "parts").mkdir(parents=True)
    part = Usd.Stage.CreateNew(str(root / "parts" / "arm.usda"))
    part.SetDefaultPrim(UsdGeom.Cube.Define(part, "/Arm").GetPrim())
    part.Save()
    main = Usd.Stage.CreateNew(str(root / "zcell.usda"))  # sorts after the part by name
    UsdGeom.Xform.Define(main, "/Cell").GetPrim().GetReferences().AddReference("./parts/arm.usda")
    main.Save()
    source = open_model_files(_folder_upload(root))
    assert source.candidates == ["Line/zcell.usda", "Line/parts/arm.usda"]  # the file nothing references first
    assert source.describe()["main"] == "Line/zcell.usda"
    source.select("Line/parts/arm.usda")
    assert source.describe()["tree"][0]["name"] == "Arm"


def _placed(asset, **update) -> BuildConfig:
    return BuildConfig(equipment=[EquipmentPlacement(model_id=asset.id, name="Crate", x=1, y=1)], **update)


def test_export_and_project_save_carry_the_files(tmp_path: Path) -> None:
    _crate(tmp_path / "Crate")
    asset = import_model(open_model_files(_folder_upload(tmp_path / "Crate")), [])
    session = MapSession()
    session.load(MapInputBundle.from_path(
        Path(__file__).resolve().parents[1] / "src" / "allmaptousd" / "samples" / "sample_factory.yaml"))
    session.models = {asset.id: asset}
    session.rebuild(_placed(asset))

    usd, files = session.export_package(".usda")
    assert set(files) == {f"{ASSET_DIR}/{asset.id}/{rel}" for rel in asset.files}
    out = session.export_file(tmp_path / "out" / "map.usda")
    stage = Usd.Stage.Open(str(out))
    shader = stage.GetPrimAtPath("/World/Map/Equipment/Crate/Model/Looks/Wood/Tex")
    assert shader.GetAttribute("inputs:file").Get().resolvedPath  # the texture is found next to the USD
    assert usd.decode().count(f"./{ASSET_DIR}/{asset.id}/") == 2

    saved = export_ros_map_zip(session.built, session.config, "plant", session.models)
    reopened = MapSession()
    reopened.load(MapInputBundle(primary_filename="plant.zip", primary_bytes=saved))
    restored = reopened.models[asset.id]
    assert restored.files == asset.files and len(restored.preview_mesh.textures) == 1


def test_export_endpoint_zips_usd_with_files(tmp_path: Path, monkeypatch) -> None:
    _crate(tmp_path / "Crate")
    asset = import_model(open_model_files(_folder_upload(tmp_path / "Crate")), [])
    monkeypatch.setattr(server, "state", server.AppState())
    monkeypatch.setattr(server, "EXPORT_DIR", tmp_path / "exports")
    server.create_empty_map(server.EmptyMapRequest(width=10, height=10, resolution=0.05, config=BuildConfig()))
    server.state.session.models = {asset.id: asset}
    response = server.export_usd(_placed(asset), ext="usda")
    assert response.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.body)) as zf:
        names = set(zf.namelist())
    assert "Empty_Map.usda" in names and f"{ASSET_DIR}/{asset.id}/Crate/textures/wood.png" in names
    assert (tmp_path / "exports" / ASSET_DIR / asset.id / "Crate" / "textures" / "wood.png").is_file()


def test_push_models_only_sends_the_files(isaac, tmp_path: Path) -> None:  # noqa: F811
    from test_isaac_models import _scene

    _crate(tmp_path / "Crate")
    asset = import_model(open_model_files(_folder_upload(tmp_path / "Crate")), [])
    stage = isaac["stage"] = _scene()
    server.state.session.models = {asset.id: asset}
    save_dir = tmp_path / "isaac_maps"
    request = server.IsaacPushRequest(build_config=_placed(asset), isaac_config=IsaacSimConfig(
        remote_save_dir=str(save_dir), pull=IsaacPullConfig()))
    result = server.push_models_to_isaac(request)
    assert result["ok"] and result["count"] == 1, result

    shader = stage.GetPrimAtPath("/World/AGV_Equipment/Crate/Model/Looks/Wood/Tex")
    texture = shader.GetAttribute("inputs:file").Get()
    expected = save_dir.resolve() / ASSET_DIR / asset.id / "Crate" / "textures" / "wood.png"
    assert texture.path == expected.as_posix() and expected.is_file()  # absolute: the scene lives elsewhere
    mdl = stage.GetPrimAtPath("/World/AGV_Equipment/Crate/Model/Looks/Blue/Shader")
    assert mdl.GetAttribute("info:mdl:sourceAsset").Get().path == "OmniPBR.mdl"  # Isaac Sim's own

    # File transfer mode copies them the same way.
    request.isaac_config.transfer_mode = "file"
    (expected).unlink()
    assert server.push_models_to_isaac(request)["ok"] and expected.is_file()


def test_export_to_a_chosen_folder_and_name(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    _crate(tmp_path / "Crate")
    asset = import_model(open_model_files(_folder_upload(tmp_path / "Crate")), [])
    monkeypatch.setattr(server, "state", server.AppState())
    monkeypatch.setattr(server, "EXPORT_DIR", tmp_path / "exports")
    server.create_empty_map(server.EmptyMapRequest(width=10, height=10, resolution=0.05, config=BuildConfig()))
    server.state.session.models = {asset.id: asset}
    client = TestClient(server.app, base_url="http://127.0.0.1:8765")
    config = _placed(asset).model_dump(mode="json")
    folder = tmp_path / "chosen" / "new folder"

    def export(name: str, overwrite: bool, ext: str = "usda", url: str = "/api/export/usd"):
        params = {"name": name, "folder": str(folder), "overwrite": overwrite, "download": False}
        return client.post(url, params={"ext": ext, **params}, json=config)

    resp = export("Line 3 Layout.usda", False)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"saved_path": str(folder / "Line 3 Layout.usda"), "files": len(asset.files)}
    assert (folder / ASSET_DIR / asset.id / "Crate" / "textures" / "wood.png").is_file()
    assert not (tmp_path / "exports").exists()

    assert export("Line 3 Layout", False).status_code == 409  # asks before replacing
    assert export("Line 3 Layout", True).status_code == 200
    saved = export("", False, url="/api/export/ros-map")
    assert saved.status_code == 200 and (folder / "Empty_Map.zip").is_file()

    relative = client.post("/api/export/usd", params={"folder": "maps", "download": False}, json=config)
    assert relative.status_code == 400 and "full path" in relative.json()["detail"]


def test_save_as_dialog_returns_the_chosen_path(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "EXPORT_DIR", tmp_path / "exports")
    calls: list[list[str]] = []
    chosen = {"path": str(tmp_path / "地圖" / "Line 3.usd")}

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=chosen["path"] + "\n")

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    client = TestClient(server.app, base_url="http://127.0.0.1:8765")
    request = {"ext": "usd", "name": "Factory A", "folder": str(tmp_path / "gone")}
    resp = client.post("/api/export/save-as", json=request)
    assert resp.json() == {"folder": str(tmp_path / "地圖"), "name": "Line 3.usd"}
    assert calls[0][-4:] == [str(tmp_path), "Factory A.usd", ".usd", "USD (binary)"]  # nearest existing folder

    chosen["path"] = ""  # cancelled
    assert client.post("/api/export/save-as", json={"ext": "zip"}).json() == {"folder": "", "name": ""}
