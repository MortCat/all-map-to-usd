"""Convert raw obstacle point clouds / wall segments into compact 2D geometry.

Supports two reconstruction modes:
1. ``contour_polygon`` (default): exact cell-corner contours (OpenCV ``findContours`` on the
   cell-corner lattice) + hole-safe splitting + ``approxPolyDP`` simplification. Preserves
   circular pillars, curved walls, L-shapes, and local protrusions.
2. ``aabb``: Greedy 2D axis-aligned rectangle merging for crisp grid-faithful boxes.

All raster operations (morphology, connected components, cavity filling) run in OpenCV on
``uint8`` grids where row index grows with +Y (row 0 = min_y).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from allmaptousd.geometry.editing import DRAW, ERASE, EditRaster
from allmaptousd.models import BuildConfig, CanonicalMap, as_points

_AREA = cv2.CC_STAT_AREA
_MAX_SPLIT_DEPTH = 6


def _square_kernel(radius: int) -> np.ndarray:
    return np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)


def _dilate(grid: np.ndarray, radius: int) -> np.ndarray:
    return cv2.dilate(grid, _square_kernel(radius)) if radius > 0 else grid


def _close(grid: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return grid
    # OpenCV's default erosion border counts as occupied; use an empty border so shapes
    # touching the grid edge shrink back like everywhere else.
    return cv2.morphologyEx(
        grid, cv2.MORPH_CLOSE, _square_kernel(radius),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _filter_small_clusters(grid: np.ndarray, min_cells: int) -> np.ndarray:
    """Drop 8-connected occupied clusters smaller than ``min_cells`` (LiDAR dust / outliers)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    keep = stats[:, _AREA] >= min_cells
    keep[0] = False  # background
    return keep[labels].astype(np.uint8)


def _fill_enclosed_solid_obstacles(grid: np.ndarray, res: float, max_area_m2: float) -> np.ndarray:
    """Fill enclosed empty cavities (e.g. hollow laser-scanned racks/pillars) into solid blocks.

    A dilate-only barrier (~0.22 m radius => bridges up to ~0.44 m LiDAR dropout gaps) is used
    strictly for cavity detection; each cavity is then grown back by the same radius so it fills
    up to the original laser perimeter without bloating outside.
    """
    if min(grid.shape) < 3:
        return grid
    seal_radius = max(2, min(8, int(round(0.22 / max(res, 1e-3)))))
    free = (_dilate(grid, seal_radius) == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    max_cells = max(1, int(math.floor(max_area_m2 / (res * res))))
    is_cavity = stats[:, _AREA] <= max_cells
    is_cavity[0] = False  # label 0 = sealed/occupied cells
    border = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    is_cavity[np.unique(border)] = False  # connected to the outside world
    if not is_cavity.any():
        return grid
    return grid | _dilate(is_cavity[labels].astype(np.uint8), seal_radius)


# ---------------------------------------------------------------------------
# Mode 1: Contour Polygons
# ---------------------------------------------------------------------------

def _polygon_signed_area(pts) -> float:
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _corner_lattice(mask: np.ndarray) -> np.ndarray:
    """Map an (h, w) cell mask to the (h+1, w+1) lattice of cell corners touching occupied cells.

    ``findContours`` traces pixel *centers*; tracing the corner lattice instead yields contours
    that lie exactly on the outer cell edges, so 1-cell-thick walls keep their full thickness.
    """
    h, w = mask.shape
    lat = np.zeros((h + 1, w + 1), np.uint8)
    lat[:h, :w] |= mask
    lat[1:, :w] |= mask
    lat[:h, 1:] |= mask
    lat[1:, 1:] |= mask
    return lat


def _mask_to_polygons(
    mask: np.ndarray,
    r0: int,
    c0: int,
    eps_cells: float,
    min_hole_cells: float,
    depth: int = 0,
) -> list[np.ndarray]:
    """Convert a cell mask into simple (hole-free) CCW polygons in cell-corner units (col, row).

    Masks enclosing a real hole (e.g. a room with no doorway) are split across their longer
    axis until every piece is hole-free, because USD extrusion uses hole-less triangulation.
    """
    contours, hierarchy = cv2.findContours(
        _corner_lattice(mask), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []
    is_hole = hierarchy[0][:, 3] >= 0

    has_real_hole = any(
        hole and cv2.contourArea(cnt) > min_hole_cells for cnt, hole in zip(contours, is_hole)
    )
    if has_real_hole and depth < _MAX_SPLIT_DEPTH:
        h, w = mask.shape
        if w >= h:
            mid = w // 2
            halves = ((mask[:, :mid], r0, c0), (mask[:, mid:], r0, c0 + mid))
        else:
            mid = h // 2
            halves = ((mask[:mid], r0, c0), (mask[mid:], r0 + mid, c0))
        out: list[np.ndarray] = []
        for sub, sr, sc in halves:
            if sub.any():
                out.extend(_mask_to_polygons(sub, sr, sc, eps_cells, min_hole_cells, depth + 1))
        return out

    polys: list[np.ndarray] = []
    for cnt, hole in zip(contours, is_hole):
        if hole:
            continue
        raw_area = cv2.contourArea(cnt)
        if raw_area <= 0.25:
            continue
        simp = cv2.approxPolyDP(cnt, eps_cells, closed=True).reshape(-1, 2).astype(np.float64)
        area = _polygon_signed_area(simp) if len(simp) >= 3 else 0.0
        if abs(area) < 0.5 * raw_area:
            # Thin diagonal walls can collapse into a sliver: keep the exact contour instead.
            simp = cnt.reshape(-1, 2).astype(np.float64)
            area = _polygon_signed_area(simp)
        if area < 0:
            simp = simp[::-1]
        polys.append(simp + (c0, r0))
    return polys


def _reconstruct_contour_polygons(
    grid: np.ndarray,
    min_x: float,
    min_y: float,
    res: float,
    eps: float,
) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=4)
    min_hole_cells = max(16.0, 0.5 / max(res * res, 1e-6))  # ignore noise pinholes < 0.5 m^2
    eps_cells = eps / res

    polygons: list[np.ndarray] = []
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        for poly in _mask_to_polygons(mask, int(y), int(x), eps_cells, min_hole_cells):
            # (col, row) cell-corner units -> world meters
            polygons.append(np.round(poly * res + (min_x, min_y), 5))
    return polygons


# ---------------------------------------------------------------------------
# Mode 2: Greedy 2D Axis-Aligned Rectangle Merging (AABB)
# ---------------------------------------------------------------------------

def _merge_grid_to_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge occupied grid cells into disjoint (r0, c0, r1_excl, c1_excl) rectangles."""
    h, w = grid.shape
    active: dict[tuple[int, int], int] = {}
    rects: list[tuple[int, int, int, int]] = []

    for r in range(h):
        row = grid[r]
        padded = np.concatenate(([False], row, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.nonzero(diffs == 1)[0].tolist()
        ends = np.nonzero(diffs == -1)[0].tolist()
        current_spans = set(zip(starts, ends))

        for span in list(active.keys()):
            if span not in current_spans:
                r0 = active.pop(span)
                rects.append((r0, span[0], r, span[1]))

        for span in current_spans:
            if span not in active:
                active[span] = r

    for span, r0 in active.items():
        rects.append((r0, span[0], h, span[1]))

    return rects


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------

def _rasterize_segments(segs: np.ndarray, res: float) -> np.ndarray:
    """Sample (M, 2, 2) segments every half cell into an (N, 2) point array."""
    if not len(segs):
        return as_points([])
    delta = segs[:, 1] - segs[:, 0]
    n = np.maximum(2, np.ceil(np.hypot(delta[:, 0], delta[:, 1]) / (res * 0.5)).astype(np.int64) + 1)
    seg_idx = np.repeat(np.arange(len(segs)), n)
    step = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    t = step / np.repeat(n - 1, n)
    return segs[seg_idx, 0] + delta[seg_idx] * t[:, None]


def _height_groups(canonical: CanonicalMap, config: BuildConfig) -> list[tuple[float, np.ndarray]]:
    """Split the raw inputs into (extrusion height, points) groups.

    Points and layer-less segments use ``wall_height``. CAD segments are grouped by
    ``config.layer_heights`` (layers listed in ``excluded_layers`` or with height <= 0 are dropped),
    so e.g. walls, racks and low obstacles can be extruded to different heights.
    """
    res = float(config.grid_resolution)
    wall_h = float(config.wall_height)
    groups: dict[float, list[np.ndarray]] = {}
    if len(canonical.points):
        groups.setdefault(wall_h, []).append(canonical.points)
    if config.seal_unknown and len(canonical.frontier_points):
        groups.setdefault(wall_h, []).append(canonical.frontier_points)

    segs, seg_layers = canonical.segments, canonical.segment_layers
    if len(segs) and (seg_layers is None or not canonical.layer_names):
        groups.setdefault(wall_h, []).append(_rasterize_segments(segs, res))
    elif len(segs):
        dropped = config.dropped_layers()
        for lid, name in enumerate(canonical.layer_names):
            if name in dropped:
                continue
            height = float(config.layer_heights.get(name, wall_h))
            sel = segs[seg_layers == lid]
            if len(sel):
                groups.setdefault(height, []).append(_rasterize_segments(sel, res))
    return [(h, np.concatenate(parts)) for h, parts in groups.items()]


def _reconstruct_points(
    pts: np.ndarray, config: BuildConfig, edits: EditRaster | None,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Rasterize one point group and rebuild it as (polygons, boxes, effective_resolution)."""
    res = float(config.grid_resolution)
    pad = max(2, config.morph_close_cells + config.dilate_cells + 1) * res
    # The extra half cell puts grid-aligned inputs (occupancy-grid cell centres) in the middle of
    # each cell; otherwise they sit exactly on cell edges and floor() rounding drops ~8% of them.
    min_x, min_y = pts.min(axis=0) - pad - 0.5 * res
    max_x, max_y = pts.max(axis=0) + pad

    def _grid_shape(r: float) -> tuple[int, int]:
        return (
            max(1, int(math.ceil((max_y - min_y) / r)) + 1),
            max(1, int(math.ceil((max_x - min_x) / r)) + 1),
        )

    rows, cols = _grid_shape(res)
    max_cells = 4096 * 4096
    if rows * cols > max_cells:
        res = round(res * math.sqrt((rows * cols) / max_cells), 4)
        rows, cols = _grid_shape(res)

    grid = np.zeros((rows, cols), dtype=np.uint8)
    ci = np.clip(np.floor((pts[:, 0] - min_x) / res).astype(np.int64), 0, cols - 1)
    ri = np.clip(np.floor((pts[:, 1] - min_y) / res).astype(np.int64), 0, rows - 1)
    grid[ri, ci] = 1

    grid = _close(grid, config.morph_close_cells)
    grid = _dilate(grid, config.dilate_cells)
    if config.min_cluster_cells > 1:
        grid = _filter_small_clusters(grid, config.min_cluster_cells)
    if config.fill_solid_obstacles:
        grid = _fill_enclosed_solid_obstacles(grid, res, config.solid_fill_max_area_m2)

    exact_grid = grid
    if edits is not None:
        rs, cs = np.nonzero(grid)
        erased = edits.erased(
            min_x + (cs + 0.5) * res, min_y + (rs + 0.5) * res,
            radius=res / 2 if res > edits.resolution else 0,
        )
        if erased.any():
            er, ec = rs[erased], cs[erased]
            if config.reconstruction_mode == "contour_polygon":
                # Keep ordinary contours away from the eraser. Use exact boxes for affected
                # components so simplification cannot close a narrow erased gap.
                count, labels = cv2.connectedComponents(grid, connectivity=4)
                affected = np.zeros(count, dtype=bool)
                affected[labels[er, ec]] = True
                affected_cells = affected[labels]
                exact_grid = grid & affected_cells
                grid[affected_cells] = 0
            exact_grid[er, ec] = 0

    if config.reconstruction_mode == "contour_polygon":
        polys = _reconstruct_contour_polygons(
            grid, float(min_x), float(min_y), res, float(config.contour_simplify_tol)
        )
        boxes = _grid_boxes(exact_grid, min_x, min_y, res) if exact_grid is not grid else as_points([], (4,))
        return polys, boxes, res

    return [], _grid_boxes(exact_grid, min_x, min_y, res), res


def _grid_boxes(grid: np.ndarray, min_x: float, min_y: float, res: float) -> np.ndarray:
    rects = np.asarray(_merge_grid_to_rectangles(grid.astype(bool)), dtype=np.float64).reshape(-1, 4)
    # (r0, c0, r1, c1) cell indices -> (x0, y0, x1, y1) meters
    return np.round(
        np.column_stack((
            min_x + rects[:, 1] * res,
            min_y + rects[:, 0] * res,
            min_x + rects[:, 3] * res,
            min_y + rects[:, 2] * res,
        )),
        5,
    )


def reconstruct_obstacles(
    canonical: CanonicalMap, config: BuildConfig, edits: EditRaster | None = None,
) -> CanonicalMap:
    """Populate the obstacle polygons/boxes (+ per-item heights) from raw points/segments."""
    polygons: list[np.ndarray] = []
    poly_heights: list[float] = []
    boxes: list[np.ndarray] = [as_points([], (4,))]
    box_heights: list[float] = []
    eff_res = float(config.grid_resolution)

    eraser = edits if edits is not None and np.any(edits.cells == ERASE) else None
    for height, pts in _height_groups(canonical, config):
        g_polys, g_boxes, res = _reconstruct_points(pts, config, eraser)
        polygons += g_polys
        poly_heights += [height] * len(g_polys)
        boxes.append(g_boxes)
        box_heights += [height] * len(g_boxes)
        eff_res = max(eff_res, res)

    if eraser is not None:
        x, y = canonical.points.T
        canonical.points = canonical.points[~eraser.erased(x, y)]
    if edits is not None:
        drawn = edits.cells == DRAW
        drawn_boxes = _grid_boxes(drawn, *edits.origin, edits.resolution)
        samples = np.flatnonzero(drawn)
        stride = max(1, math.ceil(len(samples) / 25000))
        rows, cols = np.unravel_index(samples[::stride], drawn.shape)
        canonical.drawn_points = edits.origin + (np.column_stack((cols, rows)) + 0.5) * edits.resolution
        if canonical.drawing_bounds is not None:
            b = canonical.drawing_bounds
            drawn_boxes[:, [0, 2]] = drawn_boxes[:, [0, 2]].clip(b.min_x, b.max_x)
            drawn_boxes[:, [1, 3]] = drawn_boxes[:, [1, 3]].clip(b.min_y, b.max_y)
            canonical.drawn_points = canonical.drawn_points.clip([b.min_x, b.min_y], [b.max_x, b.max_y])
        boxes.append(drawn_boxes)
        box_heights += [config.wall_height] * len(drawn_boxes)

    canonical.obstacle_polygons = polygons
    canonical.obstacle_polygon_heights = poly_heights
    canonical.obstacle_boxes = np.concatenate(boxes)
    canonical.obstacle_box_heights = box_heights

    extra = canonical.metadata.extra
    extra["reconstructed_polygon_count"] = len(polygons)
    extra["reconstructed_box_count"] = len(box_heights)
    extra["effective_grid_resolution"] = eff_res
    extra["resolution_was_coarsened"] = eff_res != float(config.grid_resolution)
    extra["reconstruction_mode"] = config.reconstruction_mode
    return canonical


# ---------------------------------------------------------------------------
# Ear-Clipping Triangulation for Extruded USD Meshes
# ---------------------------------------------------------------------------

def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    def sign(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]) -> float:
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -1e-9) or (d2 < -1e-9) or (d3 < -1e-9)
    has_pos = (d1 > 1e-9) or (d2 > 1e-9) or (d3 > 1e-9)
    return not (has_neg and has_pos)


def triangulate_polygon(
    vertices: np.ndarray,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """Ear-clipping triangulation of a (K, 2) polygon -> CCW 2D points and triangle index triples."""
    pts = [(float(x), float(y)) for x, y in vertices]
    if len(pts) > 1 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 1e-6:
        pts.pop()
    if len(pts) < 3:
        return pts, []

    if _polygon_signed_area(pts) < 0:
        pts.reverse()

    indices = list(range(len(pts)))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_iters = len(pts) * len(pts) + 10

    while len(indices) > 3 and guard < max_iters:
        guard += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-9:
                continue
            contains_other = False
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(pts[j], a, b, c):
                    contains_other = True
                    break
            if not contains_other:
                triangles.append((i_prev, i_curr, i_next))
                indices.pop(i)
                ear_found = True
                break
        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))

    return pts, triangles
