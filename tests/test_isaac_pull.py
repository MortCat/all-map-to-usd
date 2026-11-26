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
    data = collect_stage_slice(*args, 0.02, 0.6, True)
    raw = map_from_stage_slice(data, 0.05, 0.02, 0.6)
    assert data["prim_count"] == 1 and data["up_axis"] == "Y"
    assert near(raw.points, 2, 3)  # Solid interior of the clipped wall
    assert not near(raw.points, 0, 2)  # Overhead
    assert not near(raw.points, 4, 2)  # Excluded robot
    assert collect_stage_slice(*args, 0.02, 1.1, True)["prim_count"] == 2
    assert collect_stage_slice(*args, 0.02, 0.6, False)["prim_count"] == 2


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
    data = collect_stage_slice(stage, Usd.TimeCode.Default(), "/", "", 0.02, 0.6, False)
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

    def pull(_config, low, high):
        assert (low, high) == (0.02, 0.6)
        return collect_stage_slice(stage, Usd.TimeCode.Default(), "/", "", low, high, False)

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
