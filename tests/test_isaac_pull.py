"""Regression coverage for stage projection and importing the result into the live map session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from allmaptousd.exporters.usd_slice import collect_stage_slice
from allmaptousd.geometry.isaac_map import map_from_stage_slice
from allmaptousd.models import BuildConfig, IsaacSimConfig
from allmaptousd.web import server


def make_stage(axis="Z", meters=1.0):
    stage = Usd.Stage.CreateInMemory("warehouse.usda")
    UsdGeom.SetStageUpAxis(stage, axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    return stage


def box(stage, path, position, size, *, collider=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1)
    xf = UsdGeom.XformCommonAPI(shape)
    xf.SetTranslate(Gf.Vec3d(*position))
    xf.SetScale(Gf.Vec3f(*size))
    if collider:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def near(points, x, y):
    return np.any(np.linalg.norm(points - [x, y], axis=1) < 0.075)


def test_slice_respects_stage_axis_units_scope_and_collision() -> None:
    stage = make_stage("Y", 0.01)
    UsdGeom.Xform.Define(stage, "/Warehouse").AddTranslateOp().Set(Gf.Vec3d(100, 0, -200))
    hidden_wall = box(stage, "/Warehouse/Wall", (100, 70, -100), (100, 140, 100), collider=True)
    hidden_wall.MakeInvisible()
    box(stage, "/Warehouse/Floor", (100, -5, -100), (1000, 10, 1000), collider=True)
    box(stage, "/Warehouse/Overhead", (-100, 100, 0), (50, 20, 50), collider=True)
    box(stage, "/Warehouse/Robot", (300, 40, 0), (80, 80, 80), collider=True)
    box(stage, "/Warehouse/VisualOnly", (0, 40, 0), (25, 80, 25))
    box(stage, "/Outside", (0, 40, 0), (25, 80, 25), collider=True)

    args = (stage, Usd.TimeCode.Default(), "/Warehouse", "/Warehouse/Robot")
    data = collect_stage_slice(*args, 0.02, 0.6, True, 0.05)
    raw = map_from_stage_slice(data, 0.05, 0.02, 0.6)
    assert data["prim_count"] == 1 and data["up_axis"] == "Y"
    assert near(raw.points, 2, 3)  # Solid interior of the clipped wall
    assert not near(raw.points, 0, 2)  # Overhead
    assert not near(raw.points, 4, 2)  # Excluded robot
    assert collect_stage_slice(*args, 0.02, 1.1, True, 0.05)["prim_count"] == 2
    assert collect_stage_slice(*args, 0.02, 0.6, False, 0.05)["prim_count"] == 2


def test_mesh_slice_preserves_room_interior_and_clips_sloped_face() -> None:
    stage = make_stage()
    room = UsdGeom.Mesh.Define(stage, "/Room")
    room.CreatePointsAttr([(x, y, z) for z in (0, 2) for x, y in ((0, 0), (4, 0), (4, 4), (0, 4))])
    room.CreateFaceVertexCountsAttr([4] * 5)
    room.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    ramp = UsdGeom.Mesh.Define(stage, "/Ramp")
    ramp.CreatePointsAttr([(6, 0, 0), (8, 0, 2), (8, 1, 2), (6, 1, 0)])
    ramp.CreateFaceVertexCountsAttr([4])
    ramp.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    data = collect_stage_slice(stage, Usd.TimeCode.Default(), "/", "", 0.02, 0.6, False, 0.05)
    raw = map_from_stage_slice(data, 0.05, 0.02, 0.6)
    assert not near(raw.points, 2, 2)
    assert near(raw.points, 0, 2)
    assert near(raw.points, 6.3, 0.5)
    assert not near(raw.points, 7, 0.5)


def test_pull_replaces_map_only_after_success(tmp_path: Path, monkeypatch) -> None:
    state = server.AppState()
    monkeypatch.setattr(server, "state", state)
    monkeypatch.setattr(server, "ISAAC_CONFIG_FILE", tmp_path / "isaac.json")
    initial = server.create_empty_map(server.EmptyMapRequest(width=5, height=5, resolution=0.05, config=BuildConfig()))
    stage = make_stage()
    request = server.IsaacPushRequest(
        build_config=BuildConfig(origin_x=25, origin_yaw_deg=90),
        isaac_config=IsaacSimConfig(),
    )

    def pull(_config, low, high, resolution, known_models=()):
        assert (low, high) == (0.02, 0.6)
        return collect_stage_slice(
            stage, Usd.TimeCode.Default(), "/", "", low, high, False, resolution=resolution
        )

    monkeypatch.setattr(server, "pull_stage_slice", pull)
    with pytest.raises(HTTPException) as exc:
        server.pull_from_isaac(request, initial["edit_revision"])
    assert exc.value.status_code == 422
    assert state.session.edit_revision == initial["edit_revision"]
    assert state.session.raw.metadata.name == "Empty Map"

    box(stage, "/Wall", (8, 0, 0.5), (3, 1, 1))
    pulled = server.pull_from_isaac(request, initial["edit_revision"])
    assert pulled["metadata"]["source_format"] == "isaac_stage"
    assert pulled["config"]["origin_x"] == pulled["config"]["origin_yaw_deg"] == 0
    assert pulled["config"]["reconstruction_mode"] == "aabb"
    assert not pulled["config"]["fill_solid_obstacles"]
    assert near(np.asarray(pulled["raw_points"]), 8, 0)
    assert state.session.export_bytes(".usda").count(b"PhysicsCollisionAPI") >= 2


def test_rasterized_pull_handles_high_poly_warehouse_mesh() -> None:
    """High-poly meshes (>500k faces) should pull cleanly via BBox Z-culling & on-host rasterization."""
    stage = make_stage()
    # 1. Ceiling mesh with 600,000 triangles strictly above the AGV height band -> culled by BBox
    nx, ny = 600, 500  # 600 * 500 * 2 = 600,000 triangles
    xs = np.linspace(0.0, 10.0, nx + 1)
    ys = np.linspace(0.0, 10.0, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    pts_ceil = np.column_stack((gx.ravel(), gy.ravel(), np.full(gx.size, 5.0)))
    i0 = np.arange(ny)[:, None] * (nx + 1) + np.arange(nx)[None, :]
    t1 = np.stack((i0, i0 + 1, i0 + nx + 2), axis=-1).reshape(-1, 3)
    t2 = np.stack((i0, i0 + nx + 2, i0 + nx + 1), axis=-1).reshape(-1, 3)
    tris = np.vstack((t1, t2))
    ceil_mesh = UsdGeom.Mesh.Define(stage, "/Warehouse/CeilingHighPoly")
    ceil_mesh.CreatePointsAttr(pts_ceil.tolist())
    ceil_mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    ceil_mesh.CreateFaceVertexIndicesAttr(tris.ravel().tolist())

    # 2. A real obstacle intersecting 0.02..0.60 m
    box(stage, "/Warehouse/Rack", (4, 3, 1.0), (2, 1, 2), collider=True)

    data = collect_stage_slice(
        stage, Usd.TimeCode.Default(), "/Warehouse", "", 0.02, 0.6, False, resolution=0.05
    )
    assert "grid_b64" in data
    assert data["prim_count"] == 1
    raw = map_from_stage_slice(data, 0.05, 0.02, 0.6)
    assert near(raw.points, 4.0, 3.0)
    assert not near(raw.points, 8.0, 8.0)


def box_mesh(stage, path, x0, y0, x1, y1, height=1.0):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(x, y, z) for z in (0, height) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    return mesh


def test_rasterized_pull_covers_meshes_with_stale_or_missing_extents() -> None:
    stage = make_stage()
    box_mesh(stage, "/Good", 0, 0, 1, 1).CreateExtentAttr([(0, 0, 0), (1, 1, 1)])
    # Authored extent says 0..1, but the points sit at x = 6..7 (e.g. edited without updating extent)
    box_mesh(stage, "/Stale", 6, 0, 7, 1).CreateExtentAttr([(0, 0, 0), (1, 1, 1)])
    box_mesh(stage, "/NoExtent", 0, 5, 1, 6)

    data = collect_stage_slice(stage, Usd.TimeCode.Default(), "/", "", 0.02, 0.6, False, resolution=0.05)
    raw = map_from_stage_slice(data, 0.05, 0.02, 0.6)
    assert data["prim_count"] == 3
    for x, y in ((0.5, 0.5), (6.5, 0.5), (0.5, 5.5)):
        assert near(raw.points, x, y)


def test_pull_script_round_trip_through_python_server(monkeypatch) -> None:
    """Run the generated Isaac Sim cell (with stub omni modules) and parse its reply like a real pull."""
    import contextlib
    import io
    import sys
    import types

    from allmaptousd.exporters import isaac_connector

    stage = make_stage()
    box(stage, "/World/Rack", (2, 1, 0.5), (1, 1, 1), collider=True)
    omni = types.ModuleType("omni")
    omni.usd = types.SimpleNamespace(get_context=lambda: types.SimpleNamespace(get_stage=lambda: stage))
    omni.timeline = types.SimpleNamespace(
        get_timeline_interface=lambda: types.SimpleNamespace(get_current_time=lambda: 0.0)
    )
    for name, module in (("omni", omni), ("omni.usd", omni.usd), ("omni.timeline", omni.timeline)):
        monkeypatch.setitem(sys.modules, name, module)

    def python_server(_host, _port, source, timeout):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(source, "isaac_cell", "exec"), {"__name__": "__isaac_cell__"})
        return {"status": "ok", "output": output.getvalue()}

    monkeypatch.setattr(isaac_connector, "send_python_cell", python_server)
    scene = isaac_connector.pull_stage_slice(IsaacSimConfig(), 0.02, 0.6, resolution=0.05)
    raw = map_from_stage_slice(scene, 0.05, 0.02, 0.6)
    assert scene["prim_count"] == 1
    assert near(raw.points, 2, 1)
