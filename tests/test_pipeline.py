"""End-to-end tests for AllMapToUSD parsers (SEER, ROS, DXF/SICK, Raster), USD export & Isaac Sim push."""

from __future__ import annotations

import base64
import io
import json
import re
import socket
import threading
import zipfile
from pathlib import Path

# `src` and `sample_maps` are on sys.path via [tool.pytest.ini_options] pythonpath in pyproject.toml
import generate_samples
import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics

from allmaptousd.exporters import check_isaac_connection, export_to_usd
from allmaptousd.geometry import reconstruct_obstacles
from allmaptousd.models import BuildConfig, IsaacSimConfig
from allmaptousd.parsers import MapInputBundle, create_default_registry
from allmaptousd.web import server as web_server


def test_seer_smap_parser_and_usd(tmp_path: Path) -> None:
    smap_doc = {
        "header": {"mapType": "2D-Map", "mapName": "Inline_SEER", "resolution": 0.05},
        "normalPosList": [
            {"x": x * 0.05, "y": 0.0} for x in range(-40, 41)
        ] + [
            {"x": x * 0.05, "y": 4.0} for x in range(-40, 41)
        ],
        "advancedAreaList": [
            {
                "className": "ForbiddenArea",
                "instanceName": "NoGo_1",
                "posGroup": [
                    {"x": 0.5, "y": 1.0},
                    {"x": 1.5, "y": 1.0},
                    {"x": 1.5, "y": 2.0},
                    {"x": 0.5, "y": 2.0},
                ],
            }
        ],
    }
    bundle = MapInputBundle(
        primary_filename="inline.smap",
        primary_bytes=json.dumps(smap_doc).encode("utf-8"),
    )
    canonical = create_default_registry().parse(bundle, format_hint="seer_smap")
    assert canonical.metadata.source_format == "seer_smap"
    assert len(canonical.forbidden_zones) == 1

    cfg = BuildConfig(reconstruction_mode="contour_polygon")
    reconstruct_obstacles(canonical, cfg)
    assert len(canonical.obstacle_polygons) >= 2

    usd_path = export_to_usd(canonical, cfg, tmp_path / "seer.usd")
    stage = Usd.Stage.Open(str(usd_path))
    assert stage.GetPrimAtPath("/World/Map/ForbiddenZones/NoGo_1").HasAPI(UsdPhysics.CollisionAPI)


def test_ros_noisy_slam_zip_both_modes(tmp_path: Path) -> None:
    yaml_path, pgm_path = generate_samples.generate_ros_sample(tmp_path)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(yaml_path, arcname=yaml_path.name)
        zf.write(pgm_path, arcname=pgm_path.name)

    registry = create_default_registry()
    bundle = MapInputBundle(primary_filename="factory_map.zip", primary_bytes=buf.getvalue())

    # AABB mode -> .usda
    canonical_aabb = registry.parse(bundle, format_hint="ros_occupancy")
    cfg_aabb = BuildConfig(reconstruction_mode="aabb", grid_resolution=0.05, wall_height=3.0)
    reconstruct_obstacles(canonical_aabb, cfg_aabb)
    assert len(canonical_aabb.obstacle_boxes) > 20

    usda_path = export_to_usd(canonical_aabb, cfg_aabb, tmp_path / "factory.usda")
    text = usda_path.read_text(encoding="utf-8")
    assert 'upAxis = "Z"' in text
    assert "metersPerUnit = 1" in text
    assert "PhysicsCollisionAPI" in text

    # Contour Polygon mode preserves round pillars & drastically reduces prim count
    canonical_poly = registry.parse(bundle, format_hint="ros_occupancy")
    cfg_poly = BuildConfig(reconstruction_mode="contour_polygon", grid_resolution=0.05, wall_height=3.0)
    reconstruct_obstacles(canonical_poly, cfg_poly)
    assert len(canonical_poly.obstacle_polygons) < len(canonical_aabb.obstacle_boxes) * 0.1
    assert any(len(p) >= 7 for p in canonical_poly.obstacle_polygons)

    poly_usd = export_to_usd(canonical_poly, cfg_poly, tmp_path / "factory_poly.usd")
    poly_stage = Usd.Stage.Open(str(poly_usd))
    assert len(poly_stage.GetPrimAtPath("/World/Map/Obstacles").GetChildren()) == len(
        canonical_poly.obstacle_polygons
    )


def test_dxf_cad_and_raster_image_parsers(tmp_path: Path) -> None:
    registry = create_default_registry()

    # 1. DXF CAD / SICK SMET vector floorplan
    dxf_path = generate_samples.generate_dxf_sample(tmp_path / "plant.dxf")
    dxf_map = registry.parse(MapInputBundle.from_path(dxf_path), format_hint="dxf_cad")
    assert dxf_map.metadata.source_format == "dxf_cad"
    assert len(dxf_map.segments) > 20
    reconstruct_obstacles(dxf_map, BuildConfig(reconstruction_mode="contour_polygon"))
    assert len(dxf_map.obstacle_polygons) >= 6
    dxf_usd = export_to_usd(dxf_map, BuildConfig(), tmp_path / "plant.usd")
    assert Usd.Stage.Open(str(dxf_usd)).GetPrimAtPath("/World/Map/Obstacles").IsValid()

    # 2. Standalone PNG raster floorplan (e.g., SICK SMET image export) with custom resolution
    _, pgm_path = generate_samples.generate_ros_sample(tmp_path)
    png_buf = io.BytesIO()
    Image.open(pgm_path).save(png_buf, format="PNG")
    img_bundle = MapInputBundle(
        primary_filename="sick_export.png",
        primary_bytes=png_buf.getvalue(),
        options={"image_resolution": 0.05},
    )
    img_map = registry.parse(img_bundle, format_hint="raster_image")
    assert img_map.metadata.source_format == "raster_image"
    reconstruct_obstacles(img_map, BuildConfig(reconstruction_mode="contour_polygon"))
    assert len(img_map.obstacle_polygons) >= 6


def test_fastapi_endpoints(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_server, "EXPORT_DIR", tmp_path / "exports")

    from fastapi.testclient import TestClient

    client = TestClient(web_server.app)
    index = client.get("/")
    assert index.status_code == 200 and "AllMapToUSD" in index.text
    # every stylesheet / script the page references is served
    assets = re.findall(r'(?:href|src)="(/static/[^"]+)"', index.text)
    assert "/static/app.css" in assets and "/static/js/main.js" in assets
    for url in assets:
        assert client.get(url).status_code == 200, url

    payload = web_server.load_sample("ros_factory")
    assert payload["metadata"]["source_format"] == "ros_occupancy"
    assert len(payload["obstacle_polygons"]) >= 5

    usd_resp = web_server.export_usd(BuildConfig(), ext="usd")
    assert usd_resp.status_code == 200
    assert len(usd_resp.body) > 1000
    assert Path(usd_resp.headers["X-Saved-Path"]).exists()


def test_isaac_sim_python_server_push(tmp_path: Path, monkeypatch) -> None:
    """Verify exact python_server TCP protocol & USD payload delivery with a mock server."""
    monkeypatch.setattr(web_server, "ISAAC_CONFIG_FILE", tmp_path / "test_isaac_config.json")
    prev_cfg = web_server.state.isaac_config.model_copy(deep=True)
    received_scripts: list[str] = []

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]

    def _serve() -> None:
        for _ in range(2):
            conn, _ = srv.accept()
            with conn:
                chunks = []
                while True:
                    data = conn.recv(65536)
                    if not data:
                        break
                    chunks.append(data)
                src = b"".join(chunks).decode("utf-8")
                received_scripts.append(src)
                if "base64.b64decode" in src:
                    resp = {
                        "status": "ok",
                        "output": json.dumps(
                            {
                                "ok": True,
                                "remote_usd_path": "/tmp/allmaptousd/sample_factory.usd",
                                "mounted_prim": "/World/AGV_Map",
                                "push_mode": "reference",
                                "stage_url": "anon:live_stage.usd",
                            }
                        ),
                        "elapsed_seconds": 0.12,
                    }
                else:
                    resp = {
                        "status": "ok",
                        "output": json.dumps({"stage_url": "anon:live_stage.usd", "prim_count": 42}),
                    }
                conn.sendall(json.dumps(resp).encode("utf-8"))
        srv.close()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()

    cfg = IsaacSimConfig(host="127.0.0.1", port=port, target_prim_path="/World/AGV_Map")
    probe = check_isaac_connection(cfg)
    assert probe["reachable"] is True
    assert probe["prim_count"] == 42

    web_server.load_sample("ros_factory")
    push_resp = web_server.push_to_isaac(
        web_server.IsaacPushRequest(build_config=BuildConfig(), isaac_config=cfg)
    )
    assert push_resp["ok"] is True
    assert push_resp["mounted_prim"] == "/World/AGV_Map"

    t.join(timeout=3.0)
    assert len(received_scripts) == 2

    m = re.search(r"base64\.b64decode\('([^']+)'\)", received_scripts[1])
    assert m is not None
    decoded_usd = tmp_path / "decoded_pushed.usd"
    decoded_usd.write_bytes(base64.b64decode(m.group(1)))
    stage = Usd.Stage.Open(str(decoded_usd))
    assert stage.GetPrimAtPath("/World/Map/Obstacles").IsValid()
    web_server.state.isaac_config = prev_cfg


def test_dxf_bulge_and_block_references() -> None:
    import ezdxf

    doc = ezdxf.new(units=6)
    msp = doc.modelspace()
    # Semicircular bulge (bulge=1) from (0,0) to (2,0): the arc apex is 1 m away from the chord
    msp.add_lwpolyline([(0, 0, 0, 0, 1), (2, 0, 0, 0, 0)], format="xyseb")
    # Column defined once as a block and placed via INSERT
    blk = doc.blocks.new("COLUMN")
    blk.add_circle((0, 0), 0.3)
    msp.add_blockref("COLUMN", (5, 5))

    buf = io.StringIO()
    doc.write(buf)
    canonical = create_default_registry().parse(
        MapInputBundle(primary_filename="plant.dxf", primary_bytes=buf.getvalue().encode("utf-8"))
    )
    pts = canonical.segments.reshape(-1, 2)
    arc_pts = pts[pts[:, 0] <= 2.0]
    assert abs(np.hypot(arc_pts[:, 0] - 1.0, arc_pts[:, 1]) - 1.0).max() < 0.02  # arc, not a chord
    col_pts = pts[pts[:, 0] > 4.0]  # exploded block reference
    assert len(col_pts) > 8
    assert abs(np.hypot(col_pts[:, 0] - 5.0, col_pts[:, 1] - 5.0) - 0.3).max() < 0.02


def test_export_non_ascii_and_unsafe_map_names(tmp_path: Path, monkeypatch) -> None:
    from urllib.parse import unquote

    monkeypatch.setattr(web_server, "EXPORT_DIR", tmp_path / "exports")
    registry = create_default_registry()
    for name, expected in (("一號工廠", "一號工廠.usd"), ("../../escaped", "_.._escaped.usd")):
        doc = {"header": {"mapName": name}, "normalPosList": [{"x": i * 0.05, "y": 0.0} for i in range(40)]}
        bundle = MapInputBundle(primary_filename="a.smap", primary_bytes=json.dumps(doc).encode("utf-8"))
        web_server.state.session.use(registry.parse(bundle))
        resp = web_server.export_usd(BuildConfig(), ext="usd")
        saved = Path(unquote(resp.headers["X-Saved-Path"]))
        assert saved.parent == tmp_path / "exports"
        assert saved.name == expected and saved.exists()
        assert expected in unquote(resp.headers["Content-Disposition"])


# ---------------------------------------------------------------------------
# New features: navigation, ROS modes, CAD layers, output frame, CLI, Isaac file transfer
# ---------------------------------------------------------------------------

def _usd_points_world(stage, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts = UsdGeom.PointBased(prim).GetPointsAttr().Get()
    return np.array([list(xf.Transform(p)) for p in pts])


def test_seer_stations_paths_to_usd(tmp_path: Path) -> None:
    doc = {
        "header": {"mapName": "Nav"},
        "normalPosList": [{"x": i * 0.05, "y": -1.0} for i in range(60)],
        "advancedPointList": [
            {"className": "LocationMark", "instanceName": "LM1", "pos": {"x": 0.0, "y": 0.0}, "dir": 0.0},
            {"className": "ChargePoint", "instanceName": "CP1", "pos": {"x": 2.0, "y": 1.0}, "dir": 1.5708},
        ],
        "advancedCurveList": [
            {"className": "BezierPath", "instanceName": "LM1-CP1",
             "startPos": {"instanceName": "LM1", "pos": {"x": 0.0, "y": 0.0}},
             "endPos": {"instanceName": "CP1", "pos": {"x": 2.0, "y": 1.0}},
             "controlPos1": {"x": 1.0, "y": 0.0}, "controlPos2": {"x": 2.0, "y": 0.0}},
            {"className": "StraightPath", "instanceName": "CP1-LM1",
             "startPos": {"pos": {"x": 2.0, "y": 1.0}}, "endPos": {"pos": {"x": 0.0, "y": 0.0}}},
        ],
    }
    canonical = create_default_registry().parse(
        MapInputBundle(primary_filename="nav.smap", primary_bytes=json.dumps(doc).encode("utf-8"))
    )
    assert [s.name for s in canonical.stations] == ["LM1", "CP1"]
    bez, straight = canonical.paths
    assert len(bez.points) == 25 and len(straight.points) == 2
    assert np.allclose(bez.points[0], (0, 0)) and np.allclose(bez.points[-1], (2, 1))
    assert bez.points[:, 1].min() >= 0.0 and bez.points[12, 1] < 0.5  # bends toward control points

    cfg = BuildConfig()
    reconstruct_obstacles(canonical, cfg)
    stage = Usd.Stage.Open(str(export_to_usd(canonical, cfg, tmp_path / "nav.usda")))
    cp = stage.GetPrimAtPath("/World/Map/Navigation/Stations/CP1")
    assert cp.IsValid() and cp.GetCustomDataByKey("station_kind") == "ChargePoint"
    curve = UsdGeom.BasisCurves(stage.GetPrimAtPath("/World/Map/Navigation/Paths/LM1_CP1"))
    assert len(curve.GetPointsAttr().Get()) == 25
    assert not curve.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
    obstacle = stage.GetPrimAtPath("/World/Map/Obstacles/Obstacle_0000")
    assert UsdPhysics.MeshCollisionAPI(obstacle).GetApproximationAttr().Get() == "none"

    no_nav = export_to_usd(canonical, BuildConfig(include_navigation=False), tmp_path / "nonav.usda")
    assert not Usd.Stage.Open(str(no_nav)).GetPrimAtPath("/World/Map/Navigation").IsValid()


def _ros_bundle(tmp_path: Path, img: np.ndarray, yaml_extra: str = "") -> MapInputBundle:
    Image.fromarray(img).save(tmp_path / "m.pgm")
    (tmp_path / "m.yaml").write_text(
        "image: m.pgm\nresolution: 0.1\norigin: [0, 0, 0]\nnegate: 0\n"
        "occupied_thresh: 0.65\nfree_thresh: 0.196\n" + yaml_extra
    )
    return MapInputBundle.from_path(tmp_path / "m.yaml")


def test_ros_unknown_sealing_and_raw_mode(tmp_path: Path) -> None:
    registry = create_default_registry()
    img = np.full((20, 20), 205, np.uint8)  # unknown everywhere
    img[5:15, 5:15] = 254                    # free room, no walls scanned
    img[9:11, 9:11] = 0                      # one occupied pillar

    canonical = registry.parse(_ros_bundle(tmp_path, img))
    assert len(canonical.points) == 4
    assert canonical.metadata.extra["unknown_frontier_cells"] == 12 * 12 - 10 * 10

    # Sealing is a build option: the same parsed map rebuilds with or without the frontier ring
    def obstacle_area(cfg: BuildConfig) -> float:
        rebuilt = canonical.copy()
        reconstruct_obstacles(rebuilt, cfg)
        return sum(
            0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
            for p in rebuilt.obstacle_polygons
        )

    base = dict(fill_solid_obstacles=False, morph_close_cells=0, grid_resolution=0.1)
    plain_area = obstacle_area(BuildConfig(**base))
    sealed_area = obstacle_area(BuildConfig(**base, seal_unknown=True))
    assert abs(plain_area - 0.04) < 1e-6                 # just the 2x2-cell pillar
    # + the 1-cell ring (12^2 - 10^2 cells); contour tracing bevels its 4 inner corners slightly
    assert abs(sealed_area - plain_area - 0.44) < 0.03

    # Free cells on the image edge are sealed too (the map ends there)
    edge = np.full((10, 10), 254, np.uint8)
    edge_map = registry.parse(_ros_bundle(tmp_path, edge))
    assert edge_map.metadata.extra["unknown_frontier_cells"] == 36

    raw = np.full((10, 10), 255, np.uint8)   # raw mode: 255 = unknown
    raw[2:8, 2:8] = 0                        # 0 % = free
    raw[4, 4] = 100                          # 100 % = occupied
    raw[5, 5] = 50                           # 50 % < occupied_thresh -> not occupied
    raw_map = registry.parse(_ros_bundle(tmp_path, raw, "mode: raw\n"))
    assert len(raw_map.points) == 1


def test_dxf_layer_heights_and_exclusion(tmp_path: Path) -> None:
    import ezdxf

    doc = ezdxf.new(units=6)
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (10, 0), (10, 8), (0, 8)], close=True, dxfattribs={"layer": "WALL"})
    msp.add_lwpolyline([(2, 2), (4, 2), (4, 3), (2, 3)], close=True, dxfattribs={"layer": "RACK"})
    msp.add_line((6, 6), (8, 6), dxfattribs={"layer": "DIM"})
    msp.add_line((100, 100), (120, 100), dxfattribs={"layer": "FRAME"})  # far-away title block
    path = tmp_path / "layers.dxf"
    doc.saveas(path)

    canonical = create_default_registry().parse(MapInputBundle.from_path(path))
    assert set(canonical.layer_names) == {"WALL", "RACK", "DIM", "FRAME"}
    cfg = BuildConfig(layer_heights={"RACK": 1.0, "FRAME": 0}, excluded_layers=["DIM"], wall_height=3.0)
    assert cfg.dropped_layers() == {"DIM", "FRAME"}
    # Dropped layers do not stretch the map extent (fit view, ground plane, AGV start pose)
    assert canonical.compute_bounds().max_x == pytest.approx(120.0)
    kept = canonical.compute_bounds(cfg.dropped_layers())
    assert (kept.min_x, kept.min_y, kept.max_x, kept.max_y) == pytest.approx((0, 0, 10, 8))
    reconstruct_obstacles(canonical, cfg)
    assert sorted(set(canonical.obstacle_polygon_heights)) == [1.0, 3.0]
    # the DIM line at y=6 must not produce an obstacle
    assert not any(
        p[:, 1].min() > 5.5 and p[:, 1].max() < 6.5 for p in canonical.obstacle_polygons
    )

    stage = Usd.Stage.Open(str(export_to_usd(canonical, cfg, tmp_path / "layers.usda")))
    tops = sorted(
        round(float(_usd_points_world(stage, str(p.GetPath()))[:, 2].max()), 3)
        for p in stage.GetPrimAtPath("/World/Map/Obstacles").GetChildren()
    )
    assert tops[0] == 1.0 and tops[-1] == 3.0
    ground = UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/World/Environment/Ground"))
    scale = ground.GetXformVectors(Usd.TimeCode.Default())[2]
    assert scale[0] == pytest.approx(10 + 2 * cfg.ground_margin, abs=0.2)  # not ~124 m wide


def test_usd_output_frame(tmp_path: Path) -> None:
    doc = {"header": {"mapName": "Frame"},
           "advancedPointList": [{"instanceName": "A", "pos": {"x": 1.0, "y": 3.0}, "dir": 0.0}],
           "normalPosList": [{"x": 5.0, "y": 5.0}, {"x": 5.05, "y": 5.0}, {"x": 5.1, "y": 5.0}]}
    canonical = create_default_registry().parse(
        MapInputBundle(primary_filename="f.smap", primary_bytes=json.dumps(doc).encode("utf-8"))
    )
    cfg = BuildConfig(origin_x=1.0, origin_y=2.0, origin_yaw_deg=90.0)
    reconstruct_obstacles(canonical, cfg)
    stage = Usd.Stage.Open(str(export_to_usd(canonical, cfg, tmp_path / "frame.usda")))
    prim = stage.GetPrimAtPath("/World/Map/Navigation/Stations/A")
    pos = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    # p_usd = R(-90deg) * ((1, 3) - (1, 2)) = R(-90deg) * (0, 1) = (1, 0)
    assert abs(pos[0] - 1.0) < 1e-6 and abs(pos[1]) < 1e-6


def test_cli_convert(tmp_path: Path, capsys) -> None:
    from allmaptousd import cli

    yaml_path, _ = generate_samples.generate_ros_sample(tmp_path)
    out = tmp_path / "cli" / "factory.usda"
    rc = cli.main(["convert", str(yaml_path), "-o", str(out), "--mode", "aabb", "--wall-height", "2.0",
                   "--origin", "0", "0", "0", "--no-vehicle"])
    assert rc == 0 and out.exists()
    stage = Usd.Stage.Open(str(out))
    assert len(stage.GetPrimAtPath("/World/Map/Obstacles").GetChildren()) > 20
    assert not stage.GetPrimAtPath("/World/TestVehicle").IsValid()
    assert "factory.usda" in capsys.readouterr().out
    assert cli.main(["convert", str(tmp_path / "missing.yaml")]) == 1

    # The sample's west loading bay opens onto unknown space: --seal-unknown closes it in the USD
    sealed = tmp_path / "cli" / "sealed.usda"
    assert cli.main(["convert", str(yaml_path), "-o", str(sealed), "--seal-unknown", "--no-vehicle"]) == 0

    def covers(usd_path: Path, x: float, y: float) -> bool:
        st = Usd.Stage.Open(str(usd_path))
        for prim in st.GetPrimAtPath("/World/Map/Obstacles").GetChildren():
            pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get() or [(0, 0, 0)])
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                return True
        return False

    assert not covers(out, -11.45, 0.0) and covers(sealed, -11.45, 0.0)


def test_isaac_file_transfer_local_and_scp(tmp_path: Path, monkeypatch) -> None:
    from allmaptousd.exporters import isaac_connector as ic

    # Local host: large payloads are written straight into remote_save_dir, not embedded.
    received: list[str] = []

    def _fake_cell(host, port, src, timeout):
        received.append(src)
        out = {"ok": True, "remote_usd_path": "x", "mounted_prim": "/World/AGV_Map"}
        return {"status": "ok", "output": json.dumps(out)}

    monkeypatch.setattr(ic, "send_python_cell", _fake_cell)
    cfg = IsaacSimConfig(host="127.0.0.1", remote_save_dir=str(tmp_path / "maps"), transfer_mode="auto",
                         inline_limit_mb=0.1)
    resp = ic.push_usd_to_isaac(b"#usda 1.0\n" + b" " * 200_000, "Big Map", cfg)
    assert resp["ok"] and resp["delivery"].startswith("local copy")
    assert (tmp_path / "maps" / "Big_Map.usd").exists()
    assert "b64decode(" not in received[-1] and "is_file()" in received[-1]

    small = ic.push_usd_to_isaac(b"#usda 1.0\n", "Small", cfg)
    assert small["delivery"] == "inline" and "b64decode(" in received[-1]

    # Remote host: ssh mkdir + scp into the home-relative save dir.
    calls: list[list[str]] = []

    class _Done:
        returncode, stderr = 0, ""

    def _fake_run(cmd, **kw):
        calls.append(cmd)
        return _Done()

    monkeypatch.setattr(ic.subprocess, "run", _fake_run)
    remote = IsaacSimConfig(host="10.0.0.5", ssh_user="isaac", remote_save_dir="~/allmaptousd_maps")
    desc = ic.deliver_usd_file(b"data", "m.usd", remote)
    assert calls[0][0] == "ssh" and calls[0][-1] == "mkdir -p -- allmaptousd_maps"
    assert calls[1][0] == "scp" and calls[1][-1] == "isaac@10.0.0.5:allmaptousd_maps/m.usd"
    assert desc == "scp -> isaac@10.0.0.5:allmaptousd_maps/m.usd"


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _covers(polys: list[np.ndarray], x: float, y: float) -> bool:
    import cv2

    return any(
        cv2.pointPolygonTest(p.astype(np.float32).reshape(-1, 1, 2), (x, y), False) >= 0 for p in polys
    )


def test_closed_ring_room_is_split_hole_free() -> None:
    from allmaptousd.models import CanonicalMap

    # A closed 6 m x 6 m room with 0.2 m walls and no doorway: its mask has a real hole.
    res = 0.05
    xs, ys = np.meshgrid(np.arange(0, 6, res) + res / 2, np.arange(0, 6, res) + res / 2)
    wall = (xs < 0.2) | (xs > 5.8) | (ys < 0.2) | (ys > 5.8)
    room = CanonicalMap(points=np.column_stack((xs[wall], ys[wall])))

    hollow = room.copy()
    reconstruct_obstacles(hollow, BuildConfig(fill_solid_obstacles=False, morph_close_cells=0))
    # USD extrusion is hole-less, so the ring must come back as several simple pieces that
    # cover the walls exactly and leave the floor free.
    assert len(hollow.obstacle_polygons) >= 2
    assert sum(map(_polygon_area, hollow.obstacle_polygons)) == pytest.approx(36 - 5.6**2, abs=0.05)
    assert not _covers(hollow.obstacle_polygons, 3.0, 3.0)
    assert _covers(hollow.obstacle_polygons, 0.1, 3.0) and _covers(hollow.obstacle_polygons, 5.9, 3.0)

    # Rooms smaller than the solid-fill limit are filled (documented Fill Solid Loops behaviour)
    filled = room.copy()
    reconstruct_obstacles(filled, BuildConfig(morph_close_cells=0, solid_fill_max_area_m2=40.0))
    assert _covers(filled.obstacle_polygons, 3.0, 3.0)
    kept = room.copy()
    reconstruct_obstacles(kept, BuildConfig(morph_close_cells=0, solid_fill_max_area_m2=20.0))
    assert not _covers(kept.obstacle_polygons, 3.0, 3.0)


def test_dxf_units_header_and_override(tmp_path: Path) -> None:
    import ezdxf

    def extent(units: int, override: str = "auto") -> tuple[float, float]:
        doc = ezdxf.new(units=units)
        doc.modelspace().add_lwpolyline([(0, 0), (10000, 0), (10000, 8000), (0, 8000)], close=True)
        path = tmp_path / f"units_{units}.dxf"
        doc.saveas(path)
        cmap = create_default_registry().parse(MapInputBundle.from_path(path, options={"dxf_units": override}))
        b = cmap.compute_bounds()
        return round(b.width, 6), round(b.height, 6)

    assert extent(4) == (10.0, 8.0)            # $INSUNITS 4 = millimeters
    assert extent(5) == (100.0, 80.0)          # $INSUNITS 5 = centimeters
    assert extent(1) == (254.0, 203.2)         # $INSUNITS 1 = inches
    assert extent(6, "mm") == (10.0, 8.0)      # manual override beats the header
    assert extent(4, "m") == (10000.0, 8000.0)


def test_ros_negate_and_origin_yaw(tmp_path: Path) -> None:
    img = np.zeros((10, 10), np.uint8)
    img[0, 2] = 255  # top row, third column
    Image.fromarray(img).save(tmp_path / "n.pgm")

    def parse(negate: int, yaw: float):
        (tmp_path / "n.yaml").write_text(
            f"image: n.pgm\nresolution: 0.1\norigin: [1.0, 2.0, {yaw}]\nnegate: {negate}\n"
            "occupied_thresh: 0.65\nfree_thresh: 0.196\n"
        )
        return create_default_registry().parse(MapInputBundle.from_path(tmp_path / "n.yaml"))

    # negate: 0 -> black is occupied (99 cells); negate: 1 -> white is occupied (1 cell)
    assert len(parse(0, 0.0).points) == 99
    flipped = parse(1, 0.0)
    assert flipped.points.shape == (1, 2) and flipped.points[0] == pytest.approx([1.25, 2.95])
    # Cell centre (0.25, 0.95) in the map frame, rotated by origin yaw 90deg about the origin
    rotated = parse(1, float(np.pi / 2))
    assert rotated.points[0] == pytest.approx([1.0 - 0.95, 2.0 + 0.25])


def test_triangulation_covers_self_touching_contours() -> None:
    """A cell contour that touches itself along a zero-width bridge must not leave holes in the mesh."""
    from allmaptousd.geometry.reconstruct import _polygon_signed_area, triangulate_polygon

    pinched = np.array([
        (31, 10), (32, 11), (33, 10), (35, 10), (35, 11), (34, 12), (33, 12), (32, 13), (33, 14), (34, 14),
        (35, 15), (35, 16), (34, 16), (33, 17), (35, 19), (35, 20), (30, 20), (30, 16), (31, 15), (32, 15),
        (33, 14), (32, 13), (31, 13), (31, 12), (30, 11), (30, 10),
    ], dtype=float)
    pts, tris = triangulate_polygon(pinched)
    covered = sum(abs(_polygon_signed_area([pts[a], pts[b], pts[c]])) for a, b, c in tris)
    assert covered == pytest.approx(abs(_polygon_signed_area(pts)))


@pytest.mark.parametrize(
    ("area", "forbidden"),
    [
        ({"className": "ForbiddenArea"}, True),
        ({"className": "NoGoArea"}, True),
        ({"instanceName": "virtual_wall_3"}, True),
        ({"className": "BanArea"}, True),
        ({"instanceName": "禁行区"}, True),
        ({"className": "UnrestrictedArea"}, False),
        ({"instanceName": "Banner_Display"}, False),
        ({"className": "SpeedArea"}, False),
        ({"className": "AdvancedArea", "property": [{"key": "isForbidden", "boolValue": False}]}, False),
        ({"className": "AdvancedArea", "property": [{"key": "isForbidden", "boolValue": True}]}, True),
    ],
)
def test_seer_forbidden_area_keywords(area: dict, forbidden: bool) -> None:
    from allmaptousd.parsers.seer_smap import _is_forbidden_area

    assert _is_forbidden_area(area) is forbidden
