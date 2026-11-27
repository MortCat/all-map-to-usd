"""Push Models Only and Pull with models: run the real python_server cells against a local stage."""

from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from allmaptousd.exporters import isaac_connector
from allmaptousd.model_assets import import_model, open_model_source
from allmaptousd.models import BuildConfig, EquipmentPlacement, IsaacPullConfig, IsaacSimConfig
from allmaptousd.web import server


def _box_stl(sx: float, sy: float, sz: float) -> bytes:
    """Binary STL of a box, in millimetres."""
    v = np.array([[x, y, z] for z in (0, sz) for y in (0, sy) for x in (0, sx)], np.float32)
    quads = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    tris = v[[t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]]
    record = np.zeros(len(tris), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    record["v"] = tris
    return b"\0" * 80 + len(tris).to_bytes(4, "little") + record.tobytes()


def _scene(axis: str = "Z", meters: float = 1.0) -> Usd.Stage:
    """A 20 m floor slab whose top is 0.1 m up, and a rack at (6, 0)."""
    stage = Usd.Stage.CreateInMemory("scene.usda")
    UsdGeom.SetStageUpAxis(stage, axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    UsdGeom.Xform.Define(stage, "/World")
    s = 1.0 / meters

    def box(path, center, size):
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        if axis == "Y":  # Z-up (x, y, z) -> Y-up (x, z, -y)
            center, size = (center[0], center[2], -center[1]), (size[0], size[2], size[1])
        api = UsdGeom.XformCommonAPI(cube)
        api.SetTranslate(Gf.Vec3d(*(c * s for c in center)))
        api.SetScale(Gf.Vec3f(*(v * s for v in size)))

    box("/World/Floor", (0, 0, 0.05), (20, 20, 0.1))
    box("/World/Rack", (6, 0, 1.0), (1, 1, 1.8))
    return stage


@pytest.fixture
def isaac(monkeypatch, tmp_path: Path):
    """Route python_server cells into a local stage; returns a setter for the stage."""
    holder = {"stage": None}
    omni = types.ModuleType("omni")
    omni.usd = types.SimpleNamespace(get_context=lambda: types.SimpleNamespace(get_stage=lambda: holder["stage"]))
    omni.timeline = types.SimpleNamespace(
        get_timeline_interface=lambda: types.SimpleNamespace(get_current_time=lambda: 0.0)
    )
    for name, module in (("omni", omni), ("omni.usd", omni.usd), ("omni.timeline", omni.timeline)):
        monkeypatch.setitem(sys.modules, name, module)

    def python_server(_host, _port, source, timeout):
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(source, "isaac_cell", "exec"), {"__name__": "__isaac_cell__"})
        except Exception as exc:  # the python_server reports errors like this
            return {"status": "error", "ename": type(exc).__name__, "evalue": str(exc)}
        return {"status": "ok", "output": output.getvalue()}

    monkeypatch.setattr(isaac_connector, "send_python_cell", python_server)
    monkeypatch.setattr(server, "state", server.AppState())
    monkeypatch.setattr(server, "ISAAC_CONFIG_FILE", tmp_path / "isaac.json")
    server.create_empty_map(server.EmptyMapRequest(width=10, height=10, resolution=0.05, config=BuildConfig()))
    return holder


def _models():
    cart = import_model(open_model_source("cart.stl", _box_stl(2000, 1000, 500)), [])
    post = import_model(open_model_source("post.stl", _box_stl(400, 400, 1200)), [])
    server.state.session.models = {cart.id: cart, post.id: post}
    return cart, post


def _request(config: BuildConfig, **isaac) -> server.IsaacPushRequest:
    return server.IsaacPushRequest(build_config=config, isaac_config=IsaacSimConfig(**isaac))


def _world_box(stage: Usd.Stage, path: str):
    """World bounds of a prim in Z-up metres."""
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(stage.GetPrimAtPath(path))
    rng = box.ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        lo, hi = np.array([lo[0], -hi[2], lo[1]]), np.array([hi[0], -lo[2], hi[1]])
    return lo * meters, hi * meters


@pytest.mark.parametrize(("axis", "meters"), [("Z", 1.0), ("Y", 0.01)])
def test_push_models_only_then_pull_them_back(isaac, axis: str, meters: float) -> None:
    stage = isaac["stage"] = _scene(axis, meters)
    cart, post = _models()
    config = BuildConfig(origin_x=1.0, equipment=[  # map frame is 1 m off the stage frame
        EquipmentPlacement(model_id=cart.id, name="Cart A", x=1.0, y=3.0, yaw_deg=90.0),
        EquipmentPlacement(model_id=post.id, name="Post", x=-3.0, y=-4.0),
    ])
    result = server.push_models_to_isaac(_request(config))
    assert result["ok"] and result["count"] == 2, result
    assert result["floors"] == pytest.approx([0.1, 0.1], abs=1e-6)  # stands on the slab, not at 0
    assert "floor found under every model" in result["message"]

    lo, hi = _world_box(stage, "/World/AGV_Equipment/Cart_A")
    assert lo == pytest.approx((-0.5, 2.0, 0.1), abs=1e-4)  # turned 90°: 1 m x 2 m, centred at (0, 3)
    assert hi == pytest.approx((0.5, 4.0, 0.6), abs=1e-4)
    assert stage.GetRootLayer().ExportToString().count("references") == 0  # a copy, not a reference
    assert stage.GetPrimAtPath("/World/Rack")  # the scene itself is untouched

    pulled = server.pull_from_isaac(_request(BuildConfig(), pull=IsaacPullConfig(floor_elevation_m=0.1)))
    restored = {e["name"]: e for e in pulled["config"]["equipment"]}
    assert set(restored) == {"Cart A", "Post"}
    assert (restored["Cart A"]["x"], restored["Cart A"]["y"]) == pytest.approx((0.0, 3.0), abs=1e-3)
    assert restored["Cart A"]["yaw_deg"] == pytest.approx(90.0, abs=1e-3)
    assert (restored["Post"]["x"], restored["Post"]["y"]) == pytest.approx((-4.0, -4.0), abs=1e-3)
    points = np.asarray(pulled["raw_points"])
    near = lambda x, y: bool(np.any(np.linalg.norm(points - [x, y], axis=1) < 0.1))  # noqa: E731
    assert near(6, 0) and not near(0, 3) and not near(-4, -4)  # the rack is an obstacle, the models are not
    assert pulled["metadata"]["extra"]["isaac_restored_models"] == 2


def test_push_again_replaces_models_and_unknown_models_pull_as_obstacles(isaac) -> None:
    stage = isaac["stage"] = _scene()
    cart, post = _models()
    two = BuildConfig(equipment=[
        EquipmentPlacement(model_id=cart.id, name="Cart", x=0, y=3),
        EquipmentPlacement(model_id=post.id, name="Post", x=-3, y=-4),
    ])
    assert server.push_models_to_isaac(_request(two))["count"] == 2
    one = two.model_copy(update={"equipment": two.equipment[:1]})
    assert server.push_models_to_isaac(_request(one))["count"] == 1
    assert [p.GetName() for p in stage.GetPrimAtPath("/World/AGV_Equipment").GetChildren()] == ["Cart"]

    del server.state.session.models[cart.id]  # e.g. the app was restarted without the model
    pulled = server.pull_from_isaac(_request(BuildConfig(), pull=IsaacPullConfig(floor_elevation_m=0.1)))
    assert pulled["config"]["equipment"] == []
    assert pulled["metadata"]["extra"]["isaac_unknown_models"] == 1
    points = np.asarray(pulled["raw_points"])
    assert np.any(np.linalg.norm(points - [0, 3], axis=1) < 0.1)  # sliced like any other geometry

    empty = server.push_models_to_isaac(_request(BuildConfig()))
    assert empty["ok"] and empty["count"] == 0 and not stage.GetPrimAtPath("/World/AGV_Equipment")


def test_fixed_floor_and_foreign_target(isaac) -> None:
    stage = isaac["stage"] = _scene()
    cart, _ = _models()
    config = BuildConfig(equipment=[EquipmentPlacement(model_id=cart.id, name="Cart", x=0, y=3)])
    fixed = server.push_models_to_isaac(_request(
        config, equipment_floor="floor_elevation", pull=IsaacPullConfig(floor_elevation_m=0.3),
    ))
    assert fixed["ok"] and "floor elevation 0.3 m" in fixed["message"]
    assert _world_box(stage, "/World/AGV_Equipment/Cart")[0][2] == pytest.approx(0.3)

    UsdGeom.Xform.Define(stage, "/World/Line/Conveyor")
    refused = server.push_models_to_isaac(_request(config, equipment_prim_path="/World/Line"))
    assert not refused["ok"] and "did not write" in refused["message"]
    assert stage.GetPrimAtPath("/World/Line/Conveyor")


def test_missing_floor_falls_back_to_floor_elevation(isaac) -> None:
    stage = isaac["stage"] = _scene()
    stage.RemovePrim("/World/Floor")
    cart, _ = _models()
    config = BuildConfig(equipment=[EquipmentPlacement(model_id=cart.id, name="Cart", x=0, y=3)])
    result = server.push_models_to_isaac(_request(config))
    assert result["floors"] == [None] and "floor found under 0 of 1" in result["message"]
    assert _world_box(stage, "/World/AGV_Equipment/Cart")[0][2] == pytest.approx(0.0, abs=1e-6)
