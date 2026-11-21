"""Pure-Python OpenUSD (usd-core) exporter for Isaac Sim obstacle layouts.

Stage layout::

    /World                 PhysicsScene, Looks (materials), Lighting
    /World/Environment     Ground (output frame applied)
    /World/Map             Obstacles, ForbiddenZones, Navigation (output frame applied)
    /World/TestVehicle     AGV_Proxy (optional)
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from allmaptousd.geometry.reconstruct import triangulate_polygon
from allmaptousd.models import BuildConfig, CanonicalMap

# name -> (diffuse color, opacity, roughness): high-contrast industrial floor vs bright walls
_MATERIALS: dict[str, tuple[tuple[float, float, float], float, float]] = {
    "Obstacle": ((0.78, 0.80, 0.82), 1.0, 0.65),
    "Forbidden": ((0.92, 0.22, 0.24), 0.52, 0.4),
    "Ground": ((0.09, 0.11, 0.14), 1.0, 0.88),
    "VehicleProxy": ((0.08, 0.78, 0.96), 0.82, 0.35),
    "VehicleFront": ((1.0, 0.56, 0.08), 0.95, 0.3),
    "Navigation": ((0.13, 0.83, 0.93), 1.0, 0.5),
}


def _sanitize_prim_name(name: str, fallback: str = "Item") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name.strip()) or fallback
    return f"_{cleaned}" if cleaned[0].isdigit() else cleaned


def _unique_name(name: str, fallback: str, used: set[str]) -> str:
    base = _sanitize_prim_name(name, fallback)
    unique, suffix = base, 1
    while unique in used:
        suffix += 1
        unique = f"{base}_{suffix}"
    used.add(unique)
    return unique


def _bind(gprim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI(gprim.GetPrim()).Bind(material)


def _define_materials(stage: Usd.Stage) -> dict[str, UsdShade.Material]:
    UsdGeom.Scope.Define(stage, "/World/Looks")
    materials: dict[str, UsdShade.Material] = {}
    for name, (color, opacity, roughness) in _MATERIALS.items():
        path = f"/World/Looks/{name}Mat"
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[name] = material
    return materials


def _add_box(
    stage: Usd.Stage,
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: UsdShade.Material,
    collision: bool = False,
) -> UsdGeom.Cube:
    """Unit cube scaled to ``size`` (x, y, z) and centred at ``center``."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(cube)
    api.SetTranslate(Gf.Vec3d(*(float(c) for c in center)))
    api.SetScale(Gf.Vec3f(*(max(1e-4, float(s)) for s in size)))
    _bind(cube, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _add_extruded_polygon(
    stage: Usd.Stage,
    path: str,
    vertices: np.ndarray,
    height: float,
    material: UsdShade.Material,
    collision: bool,
) -> None:
    pts2d, tris = triangulate_polygon(vertices)
    if len(pts2d) < 3 or not tris:
        return
    n = len(pts2d)
    # Bottom ring (0..n-1) at z=0, top ring (n..2n-1) at z=height
    points = [Gf.Vec3f(x, y, 0.0) for x, y in pts2d] + [Gf.Vec3f(x, y, float(height)) for x, y in pts2d]
    counts = [3] * (2 * len(tris)) + [4] * n
    indices: list[int] = []
    for a, b, c in tris:
        indices += (n + a, n + b, n + c)  # top, CCW seen from +Z
    for a, b, c in tris:
        indices += (a, c, b)  # bottom, outward normal -Z
    for i in range(n):
        j = (i + 1) % n
        indices += (i, j, n + j, n + i)  # sides

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    _bind(mesh, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # Static colliders: the exact triangle mesh ("none") is accurate for concave walls and
        # cooks faster than convexDecomposition, which is only required for dynamic bodies.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)


def _framed_xform(stage: Usd.Stage, path: str, config: BuildConfig) -> Usd.Prim:
    """Xform that re-expresses map coordinates in the output frame: p_usd = R(-yaw) * (p_map - origin)."""
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    yaw = math.radians(float(config.origin_yaw_deg))
    if abs(yaw) < 1e-12 and config.origin_x == 0.0 and config.origin_y == 0.0:
        return prim
    c, s = math.cos(-yaw), math.sin(-yaw)
    ox, oy = float(config.origin_x), float(config.origin_y)
    api = UsdGeom.XformCommonAPI(prim)
    api.SetRotate(Gf.Vec3f(0.0, 0.0, -float(config.origin_yaw_deg)))
    api.SetTranslate(Gf.Vec3d(-(c * ox - s * oy), -(s * ox + c * oy), 0.0))
    return prim


def _add_scene_basics(stage: Usd.Stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Soft overhead fill light to avoid harsh long black shadows
    UsdGeom.Xform.Define(stage, "/World/Lighting")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/AmbientDome")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/OverheadKey")
    sun.CreateIntensityAttr(1200.0)
    sun.CreateAngleAttr(8.0)
    UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(18.0, 12.0, 0.0))


def _add_ground(stage: Usd.Stage, canonical: CanonicalMap, config: BuildConfig, material) -> None:
    _framed_xform(stage, "/World/Environment", config)
    bounds = canonical.compute_bounds(config.dropped_layers())
    margin, thickness = float(config.ground_margin), 0.05
    cx, cy = bounds.center
    size = (max(10.0, bounds.width + 2.0 * margin), max(10.0, bounds.height + 2.0 * margin), thickness)
    _add_box(stage, "/World/Environment/Ground", (cx, cy, -0.5 * thickness), size, material, collision=True)


def _add_obstacles(stage: Usd.Stage, canonical: CanonicalMap, config: BuildConfig, mats) -> None:
    """Reconstructed walls (polygon meshes or AABB cubes) and forbidden-zone prisms."""
    UsdGeom.Xform.Define(stage, "/World/Map/Obstacles")
    wall_h = float(config.wall_height)
    poly_heights = canonical.obstacle_polygon_heights or [wall_h] * len(canonical.obstacle_polygons)
    for idx, (poly, height) in enumerate(zip(canonical.obstacle_polygons, poly_heights)):
        _add_extruded_polygon(
            stage, f"/World/Map/Obstacles/Obstacle_{idx:04d}", poly, height, mats["Obstacle"], collision=True
        )
    box_heights = canonical.obstacle_box_heights or [wall_h] * len(canonical.obstacle_boxes)
    for idx, ((x0, y0, x1, y1), height) in enumerate(zip(canonical.obstacle_boxes.tolist(), box_heights)):
        _add_box(
            stage, f"/World/Map/Obstacles/Wall_{idx:04d}",
            (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * height), (x1 - x0, y1 - y0, height),
            mats["Obstacle"], collision=True,
        )

    UsdGeom.Xform.Define(stage, "/World/Map/ForbiddenZones")
    used: set[str] = set()
    for idx, zone in enumerate(canonical.forbidden_zones):
        name = _unique_name(zone.name, f"Forbidden_{idx + 1}", used)
        _add_extruded_polygon(
            stage, f"/World/Map/ForbiddenZones/{name}", zone.vertices, float(config.forbidden_height),
            mats["Forbidden"], collision=config.include_forbidden_as_collision,
        )


def _add_navigation(stage: Usd.Stage, canonical: CanonicalMap, material: UsdShade.Material) -> None:
    """Stations as posed Xforms (+ small marker) and routes as BasisCurves; no colliders."""
    if not canonical.stations and not canonical.paths:
        return
    UsdGeom.Xform.Define(stage, "/World/Map/Navigation")
    used: set[str] = set()

    if canonical.stations:
        UsdGeom.Xform.Define(stage, "/World/Map/Navigation/Stations")
    for st in canonical.stations:
        path = f"/World/Map/Navigation/Stations/{_unique_name(st.name, 'Station', used)}"
        xf = UsdGeom.Xform.Define(stage, path)
        api = UsdGeom.XformCommonAPI(xf)
        api.SetTranslate(Gf.Vec3d(st.x, st.y, 0.0))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(st.yaw)))
        xf.GetPrim().SetCustomDataByKey("station_kind", st.kind)
        xf.GetPrim().SetCustomDataByKey("source_name", st.name)

        disc = UsdGeom.Cylinder.Define(stage, f"{path}/Marker")
        disc.CreateRadiusAttr(0.12)
        disc.CreateHeightAttr(0.02)
        disc.CreateAxisAttr(UsdGeom.Tokens.z)
        UsdGeom.XformCommonAPI(disc).SetTranslate(Gf.Vec3d(0.0, 0.0, 0.01))
        _bind(disc, material)
        _add_box(stage, f"{path}/Heading", (0.2, 0.0, 0.01), (0.2, 0.04, 0.02), material)

    if canonical.paths:
        UsdGeom.Xform.Define(stage, "/World/Map/Navigation/Paths")
    for route in canonical.paths:
        path = f"/World/Map/Navigation/Paths/{_unique_name(route.name, 'Path', used)}"
        curve = UsdGeom.BasisCurves.Define(stage, path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreatePointsAttr([Gf.Vec3f(float(x), float(y), 0.005) for x, y in route.points])
        curve.CreateCurveVertexCountsAttr([len(route.points)])
        curve.CreateWidthsAttr([0.04])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.GetPrim().SetCustomDataByKey("path_kind", route.kind)
        curve.GetPrim().SetCustomDataByKey("source_name", route.name)
        _bind(curve, material)


def _add_vehicle_proxy(stage: Usd.Stage, config: BuildConfig, mats) -> None:
    """Parametric AGV footprint box with an orange +X heading strip (visual only, no collider)."""
    vp = config.vehicle_proxy
    _framed_xform(stage, "/World/TestVehicle", config)
    root = "/World/TestVehicle/AGV_Proxy"
    api = UsdGeom.XformCommonAPI(UsdGeom.Xform.Define(stage, root))
    api.SetTranslate(Gf.Vec3d(float(vp.pos_x), float(vp.pos_y), 0.0))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, float(vp.yaw_deg)))

    length, width, height = float(vp.length), float(vp.width), float(vp.height)
    _add_box(stage, f"{root}/Body", (0.0, 0.0, 0.5 * height), (length, width, height), mats["VehicleProxy"])
    front = min(0.08, length * 0.12)
    _add_box(
        stage, f"{root}/FrontMarker", (0.5 * length - 0.5 * front + 0.005, 0.0, 0.5 * height),
        (front, width * 0.85, height * 0.7), mats["VehicleFront"],
    )


def export_to_usd(canonical: CanonicalMap, config: BuildConfig, output_path: str | Path) -> Path:
    """Build a clean Isaac Sim-ready USD stage from a reconstructed CanonicalMap."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    stage = Usd.Stage.CreateNew(str(out))
    _add_scene_basics(stage)
    mats = _define_materials(stage)
    if config.include_ground:
        _add_ground(stage, canonical, config, mats["Ground"])

    map_root = _framed_xform(stage, "/World/Map", config)
    map_root.SetCustomDataByKey("map_name", canonical.metadata.name)
    map_root.SetCustomDataByKey("source_format", canonical.metadata.source_format)
    map_root.SetCustomDataByKey("grid_resolution_m", float(config.grid_resolution))
    map_root.SetCustomDataByKey(
        "output_origin_xy_yawdeg", Gf.Vec3d(config.origin_x, config.origin_y, config.origin_yaw_deg)
    )
    _add_obstacles(stage, canonical, config, mats)
    if config.include_navigation:
        _add_navigation(stage, canonical, mats["Navigation"])
    if config.vehicle_proxy.enabled:
        _add_vehicle_proxy(stage, config, mats)

    stage.GetRootLayer().Save()
    return out


def export_to_usd_bytes(canonical: CanonicalMap, config: BuildConfig, extension: str = ".usd") -> bytes:
    """Export a USD stage to bytes (.usd / .usdc binary Crate or .usda ASCII)."""
    ext = extension if extension.startswith(".") else f".{extension}"
    if ext not in (".usd", ".usda", ".usdc"):
        ext = ".usd"
    with tempfile.TemporaryDirectory() as tmpdir:
        return export_to_usd(canonical, config, Path(tmpdir) / f"map_layout{ext}").read_bytes()
