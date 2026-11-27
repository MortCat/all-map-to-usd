"""Read-only stage projection & vectorized 2D slice rasterization.

This module also runs standalone inside NVIDIA Isaac Sim's python_server.
Uses world-space BBox Z-culling + vectorized NumPy triangle filtering so
multi-million-face warehouse stages pull cleanly and quickly without hitting
arbitrary face/vertex caps.
"""

from __future__ import annotations

import base64
import math
import zlib
from collections import defaultdict

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

_EPS = 1e-8
MAX_GRID_CELLS = 4_000_000  # also read by geometry.isaac_map when decoding
_SHIFT = 8


def _stage_path(value: str, allow_root: bool) -> Sdf.Path:
    valid, _ = Sdf.Path.IsValidPathString(value)
    if valid:
        path = Sdf.Path(value)
        if path.IsAbsolutePath() and (path.IsPrimPath() or (allow_root and path == Sdf.Path.absoluteRootPath)):
            return path
    raise ValueError(f"Invalid absolute prim path: {value}")


def _mesh_triangles(prim: Usd.Prim, time: Usd.TimeCode) -> tuple[np.ndarray, np.ndarray, bool | None]:
    """Return (points_Nx3, triangles_Mx3, closed_hint) using fast vectorized NumPy fan-triangulation."""
    mesh = UsdGeom.Mesh(prim)
    raw_pts = mesh.GetPointsAttr().Get(time)
    raw_counts = mesh.GetFaceVertexCountsAttr().Get(time)
    raw_indices = mesh.GetFaceVertexIndicesAttr().Get(time)
    if not raw_pts or not raw_counts or not raw_indices:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64), False

    pts = np.asarray(raw_pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.int64), False

    counts = np.asarray(raw_counts, dtype=np.int64)
    indices = np.asarray(raw_indices, dtype=np.int64)
    holes = mesh.GetHoleIndicesAttr().Get(time)
    hole_mask = np.zeros(len(counts), dtype=bool)
    if holes:
        h_arr = np.asarray(holes, dtype=np.int64)
        valid_h = h_arr[(h_arr >= 0) & (h_arr < len(counts))]
        hole_mask[valid_h] = True

    n_pts = len(pts)
    valid_face = (counts >= 3) & (~hole_mask)
    if not np.any(valid_face):
        return pts, np.empty((0, 3), dtype=np.int64), False

    # Fast paths for pure-triangle or pure-quad meshes (99% of Omniverse/CAD assets)
    if not np.any(hole_mask) and np.all(counts == 3) and len(indices) == len(counts) * 3:
        tris = indices.reshape(-1, 3)
    elif not np.any(hole_mask) and np.all(counts == 4) and len(indices) == len(counts) * 4:
        quads = indices.reshape(-1, 4)
        tris = np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
    else:
        # General mixed n-gon fan triangulation (vectorized for tris/quads, loop only for >4-gons)
        offsets = np.empty(len(counts), dtype=np.int64)
        offsets[0] = 0
        if len(counts) > 1:
            np.cumsum(counts[:-1], out=offsets[1:])
        parts: list[np.ndarray] = []
        tri_mask = valid_face & (counts == 3)
        if np.any(tri_mask):
            o = offsets[tri_mask]
            parts.append(np.column_stack((indices[o], indices[o + 1], indices[o + 2])))
        quad_mask = valid_face & (counts == 4)
        if np.any(quad_mask):
            o = offsets[quad_mask]
            parts.append(np.column_stack((indices[o], indices[o + 1], indices[o + 2])))
            parts.append(np.column_stack((indices[o], indices[o + 2], indices[o + 3])))
        poly_idx = np.nonzero(valid_face & (counts > 4))[0]
        if len(poly_idx):
            extra = []
            for fi in poly_idx.tolist():
                off = int(offsets[fi])
                cnt = int(counts[fi])
                if off + cnt <= len(indices):
                    v0 = int(indices[off])
                    for k in range(1, cnt - 1):
                        extra.append((v0, int(indices[off + k]), int(indices[off + k + 1])))
            if extra:
                parts.append(np.asarray(extra, dtype=np.int64))
        tris = np.vstack(parts) if parts else np.empty((0, 3), dtype=np.int64)

    if len(tris):
        in_range = (tris.min(axis=1) >= 0) & (tris.max(axis=1) < n_pts)
        tris = tris[in_range]
    return pts, tris, None


def _primitive_triangles(prim: Usd.Prim, time: Usd.TimeCode) -> tuple[np.ndarray, np.ndarray, bool]:
    if prim.IsA(UsdGeom.Cube):
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get(time)) / 2.0
        pts = np.asarray(
            [(x * half, y * half, z * half) for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)],
            dtype=np.float64,
        )
        quads = np.asarray(
            [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6)],
            dtype=np.int64,
        )
        tris = np.vstack((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
        return pts, tris, True

    shape_type = prim.GetTypeName()
    schema = getattr(UsdGeom, shape_type)(prim)
    radius = float(schema.GetRadiusAttr().Get(time))
    axis = "Z" if shape_type == "Sphere" else str(schema.GetAxisAttr().Get(time))
    half = 0.0 if shape_type == "Sphere" else float(schema.GetHeightAttr().Get(time)) / 2.0
    if shape_type == "Cylinder":
        rings = [(-half, radius), (half, radius)]
    elif shape_type == "Cone":
        rings = [(-half, radius), (half, 0.0)]
    else:
        latitudes = [math.pi * n / 16 for n in range(-8, 9)]
        rings = [
            (radius * math.sin(angle) + (-half if angle < 0 else half), radius * math.cos(angle))
            for angle in latitudes
        ]
        if shape_type == "Capsule":
            rings.insert(8, (-half, radius))

    sides = 32
    pts_list = []
    for z, ring_radius in rings:
        for i in range(sides):
            angle = 2.0 * math.pi * i / sides
            x, y = ring_radius * math.cos(angle), ring_radius * math.sin(angle)
            pts_list.append({"X": (z, x, y), "Y": (y, z, x), "Z": (x, y, z)}[axis])
    pts = np.asarray(pts_list, dtype=np.float64)

    tris_list = []
    # Top and bottom caps
    for i in range(1, sides - 1):
        tris_list.append((0, i, i + 1))
        base = len(pts_list) - sides
        tris_list.append((base, base + i, base + i + 1))
    # Side rings
    for ring in range(len(rings) - 1):
        for i in range(sides):
            a, b = ring * sides + i, ring * sides + (i + 1) % sides
            tris_list.append((a, b, b + sides))
            tris_list.append((a, b + sides, a + sides))
    return pts, np.asarray(tris_list, dtype=np.int64), True


def _transform_points(pts: np.ndarray, transform: Gf.Matrix4d, meters: float, y_up: bool) -> np.ndarray:
    """Vectorized local-to-world transform (row-vector convention matching Gf.Matrix4d.Transform)."""
    mat = np.asarray(transform, dtype=np.float64)
    world = (pts @ mat[:3, :3] + mat[3, :3]) * meters
    if y_up:
        world = np.column_stack((world[:, 0], -world[:, 2], world[:, 1]))
    return world


def _crossing(a, b, height: float):
    t = (height - a[2]) / (b[2] - a[2])
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, height


def _clip_face(vertices, height: float, above: bool):
    if not vertices:
        return []
    result = []
    a = vertices[-1]
    for b in vertices:
        a_inside = a[2] >= height if above else a[2] <= height
        b_inside = b[2] >= height if above else b[2] <= height
        if a_inside != b_inside:
            result.append(_crossing(a, b, height))
        if b_inside:
            result.append(b)
        a = b
    return result


def _closed_mesh_fast(tris: np.ndarray) -> bool:
    """Check if a triangle mesh is closed (every undirected edge shared by 2 triangles)."""
    if len(tris) == 0 or len(tris) > 250_000:
        return False
    e0 = np.sort(tris[:, [0, 1]], axis=1)
    e1 = np.sort(tris[:, [1, 2]], axis=1)
    e2 = np.sort(tris[:, [2, 0]], axis=1)
    edges = np.vstack((e0, e1, e2))
    # Pack pairs into 64-bit keys for fast unique-count check
    max_v = int(edges.max()) + 1
    keys = edges[:, 0] * max_v + edges[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def _section_loops_tris(world: np.ndarray, tris: np.ndarray, height: float):
    """Extract horizontal cross-section loops at z=height using only crossing triangles."""
    z = world[:, 2]
    tz = z[tris]
    cross_mask = (tz.min(axis=1) < height) & (tz.max(axis=1) > height)
    if not np.any(cross_mask):
        return []

    neighbors = defaultdict(set)
    for tri in tris[cross_mask].tolist():
        vertices = [world[i].tolist() for i in tri]
        crossings = []
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            if (a[2] < height) != (b[2] < height):
                x, y, _ = _crossing(a, b, height)
                crossings.append((round(x, 8), round(y, 8)))
        if len(crossings) == 2 and crossings[0] != crossings[1]:
            neighbors[crossings[0]].add(crossings[1])
            neighbors[crossings[1]].add(crossings[0])

    loops = []
    remaining = set(neighbors)
    for start in sorted(neighbors):
        if start not in remaining:
            continue
        loop = [start]
        previous, current = None, start
        while len(neighbors[current]) == 2:
            end = min(neighbors[current] - {previous})
            if end == start:
                if len(loop) >= 3:
                    loops.append(loop)
                break
            if end not in remaining:
                break
            remaining.remove(end)
            loop.append(end)
            previous, current = current, end
        remaining.remove(start)
    return loops


def _rasterize_poly_numpy(grid: np.ndarray, poly_px: np.ndarray) -> None:
    """Pure-NumPy scanline polygon fill fallback when OpenCV is unavailable."""
    h, w = grid.shape
    if len(poly_px) < 3:
        return
    xs = poly_px[:, 0]
    ys = poly_px[:, 1]
    min_y = max(0, int(math.floor(float(ys.min()))))
    max_y = min(h - 1, int(math.ceil(float(ys.max()))))
    if min_y > max_y:
        return
    n = len(poly_px)
    for y in range(min_y, max_y + 1):
        yc = y + 0.5
        inter = []
        j = n - 1
        for i in range(n):
            yi, yj = float(ys[i]), float(ys[j])
            if (yi > yc) != (yj > yc):
                xi, xj = float(xs[i]), float(xs[j])
                inter.append(xi + (yc - yi) * (xj - xi) / (yj - yi))
            j = i
        if len(inter) < 2:
            continue
        inter.sort()
        for k in range(0, len(inter) - 1, 2):
            x0 = max(0, int(math.floor(inter[k])))
            x1 = min(w - 1, int(math.ceil(inter[k + 1])))
            if x0 <= x1:
                grid[y, x0 : x1 + 1] = 1


def _fill_polygons_into_grid(
    grid: np.ndarray,
    polys_world: list[np.ndarray],
    origin: np.ndarray,
    res: float,
    together: bool = False,
) -> None:
    if not polys_world:
        return
    if cv2 is not None:
        scaled = [
            np.rint(((p - origin) / res - 0.5) * (1 << _SHIFT)).astype(np.int32)
            for p in polys_world
            if len(p) >= 3
        ]
        if not scaled:
            return
        if together:
            cv2.fillPoly(grid, scaled, color=1, shift=_SHIFT)
        else:
            for poly in scaled:
                cv2.fillPoly(grid, [poly], color=1, shift=_SHIFT)
    else:
        for p in polys_world:
            if len(p) >= 3:
                _rasterize_poly_numpy(grid, (p - origin) / res)


def _clip_triangles_to_band(
    world: np.ndarray,
    tris: np.ndarray,
    low: float,
    high: float,
) -> list[np.ndarray]:
    """Return 2D clipped polygons for all triangles intersecting (low, high]."""
    z = world[:, 2]
    tz = z[tris]
    t_min = tz.min(axis=1)
    t_max = tz.max(axis=1)
    active = (t_max > low + _EPS) & (t_min <= high)
    if not np.any(active):
        return []

    act_tris = tris[active]
    act_min = t_min[active]
    act_max = t_max[active]

    # Fast path: triangles completely inside [low, high] need NO clipping!
    inside = (act_min >= low) & (act_max <= high)
    polys: list[np.ndarray] = []
    if np.any(inside):
        xy = world[:, :2]
        for tri_xy in xy[act_tris[inside]]:
            polys.append(tri_xy)

    # Only boundary-straddling triangles need Sutherland-Hodgman Z-clipping
    straddle = ~inside
    if np.any(straddle):
        for tri in act_tris[straddle].tolist():
            verts = [world[i].tolist() for i in tri]
            clipped = _clip_face(_clip_face(verts, low, above=True), high, above=False)
            if len(clipped) >= 3:
                polys.append(np.asarray([(p[0], p[1]) for p in clipped], dtype=np.float64))
    return polys


def _is_map_helper(prim: Usd.Prim) -> bool:
    if prim.GetName() == "Navigation":
        return prim.GetParent().HasCustomDataKey("grid_resolution_m")
    if prim.GetName() == "TestVehicle":
        sibling = prim.GetParent().GetChild("Map")
        return bool(sibling and sibling.HasCustomDataKey("grid_resolution_m"))
    return False


def _world_bbox_xy_z(
    bbox_cache: UsdGeom.BBoxCache,
    prim: Usd.Prim,
    meters: float,
    y_up: bool,
) -> tuple[float, float, float, float, float, float] | None:
    """Return (min_x, min_y, max_x, max_y, min_z, max_z) in Z-up meters, or None if empty."""
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    lo = rng.GetMin() * meters
    hi = rng.GetMax() * meters
    if y_up:
        return (lo[0], -hi[2], hi[0], -lo[2], lo[1], hi[1])
    return (lo[0], lo[1], hi[0], hi[1], lo[2], hi[2])


def _prim_band_geometry(
    prim: Usd.Prim,
    is_mesh: bool,
    time: Usd.TimeCode,
    transforms: UsdGeom.XformCache,
    meters: float,
    y_up: bool,
    low: float,
    high: float,
):
    """Clip one prim to (low, high]: (polygons, section loops, (xy_min, xy_max)), or None if it misses."""
    pts, tris, closed = (_mesh_triangles if is_mesh else _primitive_triangles)(prim, time)
    if len(tris) == 0:
        return None
    world = _transform_points(pts, transforms.GetLocalToWorldTransform(prim), meters, y_up)
    tz = world[:, 2][tris]
    active = (tz.max(axis=1) > low + _EPS) & (tz.min(axis=1) <= high)
    xy = world[np.unique(tris[active]), :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        return None
    polys = _clip_triangles_to_band(world, tris, low, high)
    loops = []
    if float(world[:, 2].min()) < low and (closed or (closed is None and _closed_mesh_fast(tris))):
        loops = _section_loops_tris(world, tris, low + _EPS)
    return polys, loops, (xy.min(axis=0), xy.max(axis=0))


def _rasterize_band(candidates, geometry_args, resolution: float, bounds):
    """Fill every candidate's band projection into one grid sized to ``bounds`` (None: measure only).

    Returns (grid, origin, res, prim_count, seen_lo, seen_hi); ``seen_*`` is the real xy extent of the
    projected geometry, which can exceed ``bounds`` when authored extents are missing or stale.
    """
    grid = origin = None
    res = max(0.01, float(resolution))
    if bounds is not None:
        lo, hi = bounds
        while True:
            origin = np.floor(lo / res) * res
            cols, rows = [math.ceil(float(length) / res) + 2 for length in (hi - origin)]
            if rows * cols <= MAX_GRID_CELLS:
                break
            if res >= 1.0:
                raise ValueError("The scene slice is too large for the map grid. Select a smaller source prim.")
            res = min(1.0, math.ceil(max(res * 1.1, res * math.sqrt(rows * cols / MAX_GRID_CELLS)) * 1000) / 1000)
        grid = np.zeros((rows, cols), dtype=np.uint8)

    seen_lo = np.full(2, math.inf)
    seen_hi = np.full(2, -math.inf)
    prim_count = 0
    for prim, is_mesh in candidates:
        geometry = _prim_band_geometry(prim, is_mesh, *geometry_args)
        if geometry is None:
            continue
        polys, loops, (xy_lo, xy_hi) = geometry
        seen_lo, seen_hi = np.minimum(seen_lo, xy_lo), np.maximum(seen_hi, xy_hi)
        if grid is None:
            continue
        if polys:
            _fill_polygons_into_grid(grid, polys, origin, res)
            prim_count += 1
        if loops:
            loop_arrs = [np.asarray(loop, dtype=np.float64) for loop in loops]
            _fill_polygons_into_grid(grid, loop_arrs, origin, res, together=True)
    return grid, origin, res, prim_count, seen_lo, seen_hi


def collect_stage_slice(
    stage: Usd.Stage,
    time: Usd.TimeCode,
    root_path: str,
    exclude_path: str,
    low: float,
    high: float,
    collision_only: bool,
    resolution: float,
    skip_paths: tuple[str, ...] | list[str] = (),
) -> dict:
    """Project geometry intersecting [low, high] onto the horizontal plane as an occupancy grid.

    Rasterizes inside Isaac Sim so multi-million-face warehouse scenes come back as a compact,
    bit-packed grid (``grid_b64``) instead of geometry. Without occupied cells the grid keys are absent.
    ``skip_paths`` are whole subtrees left out (placed 3D models that come back as models instead).
    """
    root = stage.GetPrimAtPath(_stage_path(root_path, allow_root=True))
    if not root:
        raise ValueError(f"Stage prim not found: {root_path}")
    excluded = _stage_path(exclude_path, allow_root=False) if exclude_path else None
    skipped = {Sdf.Path(p) for p in skip_paths}
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    transforms = UsdGeom.XformCache(time)
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide]
    bbox_cache = UsdGeom.BBoxCache(time, purposes, useExtentsHint=True)
    supported = (UsdGeom.Cube, UsdGeom.Cylinder, UsdGeom.Cone, UsdGeom.Sphere, UsdGeom.Capsule)

    # Pass 1: Collect candidate prims using fast world-space BBox Z-culling
    candidates: list[tuple[Usd.Prim, bool]] = []  # (prim, is_mesh)
    unsupported = 0
    bounds_lo = np.array([math.inf, math.inf], dtype=np.float64)
    bounds_hi = np.array([-math.inf, -math.inf], dtype=np.float64)

    prim_range = iter(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    for prim in prim_range:
        if (excluded is not None and prim.GetPath().HasPrefix(excluded)) or prim.GetPath() in skipped or (
            prim.HasAPI(UsdPhysics.ArticulationRootAPI) or _is_map_helper(prim)
        ):
            prim_range.PruneChildren()
            continue
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        has_collision = bool(collision and collision.GetCollisionEnabledAttr().Get(time))
        visible = (
            imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
        )
        if not has_collision and (collision_only or not visible):
            continue

        is_mesh = prim.IsA(UsdGeom.Mesh)
        if not is_mesh and not any(prim.IsA(shape) for shape in supported):
            unsupported += bool(prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer))
            continue

        # Fast world-space BBox Z-culling before touching any vertex buffers
        wb = _world_bbox_xy_z(bbox_cache, prim, meters, y_up)
        if wb is not None:
            bx0, by0, bx1, by1, bz0, bz1 = wb
            if bz1 <= low + _EPS or bz0 > high:
                continue
            if all(math.isfinite(v) for v in (bx0, by0, bx1, by1)):
                bounds_lo[0] = min(bounds_lo[0], bx0)
                bounds_lo[1] = min(bounds_lo[1], by0)
                bounds_hi[0] = max(bounds_hi[0], bx1)
                bounds_hi[1] = max(bounds_hi[1], by1)

        candidates.append((prim, is_mesh))

    result = {
        "stage": stage.GetRootLayer().GetDisplayName() or "Isaac Sim",
        "up_axis": "Y" if y_up else "Z",
        "root_prim_path": root_path,
        "unsupported": unsupported,
    }
    geometry_args = (time, transforms, meters, y_up, low, high)

    # Authored extents only size the grid. Missing or stale extents are caught by measuring the real
    # geometry while rasterizing; one more pass then covers everything that was clipped.
    lo, hi = bounds_lo, bounds_hi
    for _ in range(2):
        sized = bool(np.isfinite(lo).all() and np.isfinite(hi).all())
        grid, origin, res, prim_count, seen_lo, seen_hi = _rasterize_band(
            candidates, geometry_args, resolution, (lo, hi) if sized else None
        )
        if not np.isfinite(seen_lo).all():
            return {**result, "prim_count": prim_count}
        if sized and (seen_lo >= lo).all() and (seen_hi <= hi).all():
            break
        lo, hi = np.minimum(lo, seen_lo), np.maximum(hi, seen_hi)

    # Trim grid tightly around occupied cells to minimize transfer & memory
    occ_y, occ_x = np.nonzero(grid)
    if len(occ_x) == 0:
        return {**result, "prim_count": prim_count}
    y0, y1 = int(occ_y.min()), int(occ_y.max()) + 1
    x0, x1 = int(occ_x.min()), int(occ_x.max()) + 1
    cropped = np.ascontiguousarray(grid[y0:y1, x0:x1], dtype=np.uint8)
    return {
        **result,
        "prim_count": prim_count,
        "grid_b64": base64.b64encode(zlib.compress(np.packbits(cropped).tobytes(), level=6)).decode("ascii"),
        "grid_shape": [int(cropped.shape[0]), int(cropped.shape[1])],
        "grid_origin": [float(origin[0] + x0 * res), float(origin[1] + y0 * res)],
        "grid_resolution": float(res),
    }
