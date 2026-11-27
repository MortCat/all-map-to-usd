"""Placed 3D models in a live Isaac Sim stage: write them into the scene, find the floor under them,
and recognise them again on Pull.

Like :mod:`usd_slice`, this module also runs inside Isaac Sim's python_server: the push and pull cells
paste ``usd_slice.py`` and then this file, so the helpers imported below are already defined there.
Every placed model is an Xform tagged with ``customData["allmaptousd"]["model_id"]``; its origin is the
model's centre at floor level, and its local X axis is the model's heading.
"""

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

try:
    from allmaptousd.exporters.usd_slice import (
        _is_map_helper,
        _mesh_triangles,
        _primitive_triangles,
        _transform_points,
        _world_bbox_xy_z,
    )
except ImportError:  # inside Isaac Sim, where usd_slice.py is pasted above this module
    pass

TAG = "allmaptousd"
_SHAPES = (UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Sphere, UsdGeom.Capsule)


def equipment_tag(prim):
    """The model tag of a placed model prim, or None."""
    data = prim.GetCustomDataByKey(TAG)
    if isinstance(data, dict) and isinstance(data.get("model_id"), str):
        return data
    return None


def _stage_frame(stage):
    return float(UsdGeom.GetStageMetersPerUnit(stage)), UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y


def find_equipment(stage, time, root_path, known_ids):
    """Placed models under ``root_path``: poses (Z-up metres) of those whose model the app has loaded.

    Returns (records, number of tagged models the app does not have)."""
    meters, y_up = _stage_frame(stage)
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return [], 0
    transforms = UsdGeom.XformCache(time)
    known = set(known_ids)
    records, unknown = [], 0
    prim_range = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    for prim in prim_range:
        tag = equipment_tag(prim)
        if tag is None:
            continue
        prim_range.PruneChildren()
        if tag["model_id"] not in known:
            unknown += 1  # sliced like any other geometry
            continue
        origin, ahead = _transform_points(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), transforms.GetLocalToWorldTransform(prim), meters, y_up,
        )
        heading = ahead - origin
        records.append({
            "model_id": tag["model_id"],
            "name": str(tag.get("name") or prim.GetName())[:128],
            "x": float(origin[0]),
            "y": float(origin[1]),
            "z": float(origin[2]),
            "yaw_deg": math.degrees(math.atan2(heading[1], heading[0])),
            "path": str(prim.GetPath()),
        })
    return records, unknown


def _surface_candidates(stage, time, points, low, high, skip_paths):
    """Visible or collidable geometry whose world box spans [low, high] over any of ``points``."""
    meters, y_up = _stage_frame(stage)
    skip = {Sdf.Path(p) for p in skip_paths if p}
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    bbox_cache = UsdGeom.BBoxCache(time, purposes, useExtentsHint=True)
    lo_xy, hi_xy = points.min(axis=0), points.max(axis=0)
    prim_range = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in prim_range:
        if (
            prim.GetPath() in skip or equipment_tag(prim) is not None
            or prim.HasAPI(UsdPhysics.ArticulationRootAPI) or _is_map_helper(prim)
        ):
            prim_range.PruneChildren()
            continue
        is_mesh = prim.IsA(UsdGeom.Mesh)
        if not is_mesh and not any(prim.IsA(shape) for shape in _SHAPES):
            continue
        imageable = UsdGeom.Imageable(prim)
        collision = UsdPhysics.CollisionAPI(prim)
        collides = bool(collision and collision.GetCollisionEnabledAttr().Get(time))
        visible = (
            imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
        )
        if not (visible or collides):
            continue
        box = _world_bbox_xy_z(bbox_cache, prim, meters, y_up)
        if box is not None:
            x0, y0, x1, y1, z0, z1 = box
            if z1 < low or z0 > high or x1 < lo_xy[0] or x0 > hi_xy[0] or y1 < lo_xy[1] or y0 > hi_xy[1]:
                continue
        yield prim, is_mesh


def _highest_hits(world, tris, points, low, high, best):
    """Raise ``best[k]`` to the highest surface z in [low, high] straight below ``points[k]``."""
    a, b, c = world[tris[:, 0]], world[tris[:, 1]], world[tris[:, 2]]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    flat = np.abs(den) > 1e-12  # vertical faces cannot be stood on
    if not np.any(flat):
        return
    a, b, c, v0, v1, den = a[flat], b[flat], c[flat], v0[flat], v1[flat], den[flat]
    tri_lo = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    tri_hi = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    for k, (px, py) in enumerate(points):
        near = (tri_lo[:, 0] <= px) & (px <= tri_hi[:, 0]) & (tri_lo[:, 1] <= py) & (py <= tri_hi[:, 1])
        if not np.any(near):
            continue
        ak, v0k, v1k, dk = a[near], v0[near], v1[near], den[near]
        wx, wy = px - ak[:, 0], py - ak[:, 1]
        u = (wx * v1k[:, 1] - v1k[:, 0] * wy) / dk
        v = (v0k[:, 0] * wy - wx * v0k[:, 1]) / dk
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        if not np.any(inside):
            continue
        z = ak[:, 2] + u * (b[near][:, 2] - ak[:, 2]) + v * (c[near][:, 2] - ak[:, 2])
        z = z[inside & (z >= low) & (z <= high)]
        if len(z):
            best[k] = max(best[k], float(z.max()))


def floor_heights(stage, time, samples, low, high, skip_paths=()):
    """For each model (a list of (x, y) points in Z-up metres): the floor under it, or None.

    A point's floor is the highest surface in [low, high] straight below it; a model's floor is the
    median over its points, so a small object under one corner does not lift the whole model."""
    groups = [np.asarray(points, dtype=np.float64).reshape(-1, 2) for points in samples]
    if not groups or not sum(len(g) for g in groups):
        return [None] * len(groups)
    meters, y_up = _stage_frame(stage)
    every = np.vstack(groups)
    best = np.full(len(every), -math.inf)
    transforms = UsdGeom.XformCache(time)
    for prim, is_mesh in _surface_candidates(stage, time, every, low, high, skip_paths):
        pts, tris, _ = (_mesh_triangles if is_mesh else _primitive_triangles)(prim, time)
        if len(tris):
            world = _transform_points(pts, transforms.GetLocalToWorldTransform(prim), meters, y_up)
            _highest_hits(world, tris, every, low, high, best)
    heights, start = [], 0
    for group in groups:
        hits = best[start:start + len(group)]
        start += len(group)
        hits = hits[np.isfinite(hits)]
        heights.append(float(np.median(hits)) if len(hits) else None)
    return heights


def _define_ancestors(stage, path):
    for prefix in path.GetParentPath().GetPrefixes():
        prim = stage.GetPrimAtPath(prefix)
        if not prim or not prim.IsDefined():
            UsdGeom.Xform.Define(stage, prefix)


def _to_stage_units(prim, meters, y_up):
    """Prepend a transform that turns the copied Z-up metre content into the stage's units and axis."""
    if abs(meters - 1.0) < 1e-12 and not y_up:
        return
    s = 1.0 / meters
    rows = [(s, 0, 0, 0), (0, 0, -s, 0), (0, s, 0, 0)] if y_up else [(s, 0, 0, 0), (0, s, 0, 0), (0, 0, s, 0)]
    xform = UsdGeom.Xformable(prim)
    op = xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "toStageUnits")
    op.Set(Gf.Matrix4d(*(v for row in rows for v in row), 0, 0, 0, 1))
    xform.SetXformOpOrder([op] + [o for o in xform.GetOrderedXformOps() if o.GetName() != op.GetName()])


def _set_floor(prim, z):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            op.Set(Gf.Vec3d(value[0], value[1], z))
            return


def place_equipment(stage, time, source, source_root, target_path, floor, auto_floor, search, skip_paths, samples,
                    asset_prefix="", asset_folder=""):
    """Replace ``target_path`` in the stage's edit layer with a copy of ``source_root`` from ``source``.

    ``floor`` is the fallback floor height (Z-up metres); with ``auto_floor`` each model stands on the
    surface found under ``samples[i]`` within ``search`` = (low, high). Asset paths starting with
    ``asset_prefix`` (relative, e.g. ``./allmaptousd_assets/``) point at files that were saved in
    ``asset_folder``: they become absolute, as the copy lives in the scene's own layer, wherever that is."""
    target = Sdf.Path(target_path)
    if not target.IsAbsolutePath() or not target.IsPrimPath():
        raise ValueError(f"Invalid models prim path: {target_path}")
    existing = stage.GetPrimAtPath(target)
    if existing:
        tag = existing.GetCustomDataByKey(TAG)
        if existing.GetChildren() and not (isinstance(tag, dict) and tag.get("equipment_root")):
            raise ValueError(
                f"{target_path} already holds prims that AllMapToUSD did not write. Choose another Models Prim."
            )
        stage.RemovePrim(target)

    children = list(source.GetPrimAtPath(source_root).nameChildren) if source.GetPrimAtPath(source_root) else []
    if not children:
        return {"mounted_prim": target_path, "count": 0, "floors": []}
    heights = [None] * len(children)
    if auto_floor:
        heights = floor_heights(stage, time, samples, search[0], search[1], [target_path, *skip_paths])

    meters, y_up = _stage_frame(stage)
    _define_ancestors(stage, target)
    if asset_prefix and asset_folder:
        UsdUtils.ModifyAssetPaths(source, lambda p: f"{asset_folder}/{p[2:]}" if p.startswith(asset_prefix) else p)
    Sdf.CopySpec(source, source_root, stage.GetEditTarget().GetLayer(), target)
    root = stage.GetPrimAtPath(target)
    _to_stage_units(root, meters, y_up)
    floors = []
    for spec, height in zip(children, heights):
        z = floor if height is None else height
        _set_floor(root.GetChild(spec.name), z)
        floors.append(None if height is None else round(z, 4))
    return {"mounted_prim": target_path, "count": len(children), "floors": floors}
