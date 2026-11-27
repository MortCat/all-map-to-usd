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
from collections.abc import Collection
from typing import NamedTuple

import cv2
import numpy as np

from allmaptousd.geometry.editing import DRAW, ERASE, MAX_EDIT_CELLS, EditRaster
from allmaptousd.models import BuildConfig, CanonicalMap, as_points

_AREA = cv2.CC_STAT_AREA
_MAX_SPLIT_DEPTH = 6


class OccupancyGrid(NamedTuple):
    cells: np.ndarray
    origin: np.ndarray
    resolution: float


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


def _enclosed_cells(
    barrier: np.ndarray, max_cells: int, *, edited_boundary: np.ndarray | None = None,
) -> np.ndarray:
    free = (barrier == 0).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    is_cavity = stats[:, _AREA] <= max_cells
    is_cavity[0] = False  # label 0 = sealed/occupied cells
    border = np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    is_cavity[np.unique(border)] = False  # connected to the outside world
    if edited_boundary is not None:
        touches_edit = np.zeros_like(is_cavity)
        touches_edit[labels[_dilate(edited_boundary, 1) != 0]] = True
        is_cavity &= touches_edit
    return is_cavity[labels].astype(np.uint8)


def _scan_seal_radius(res: float) -> int:
    return max(2, min(8, round(0.22 / max(res, 1e-3))))


def _fill_enclosed_solid_obstacles(grid: np.ndarray, res: float, max_area_m2: float) -> np.ndarray:
    """Detect scan cavities across small dropouts and grow the fill to the original perimeter."""
    if min(grid.shape) < 3:
        return grid
    seal_radius = _scan_seal_radius(res)
    cavities = _enclosed_cells(_dilate(grid, seal_radius), max(1, math.floor(max_area_m2 / (res * res))))
    return grid | _dilate(cavities, seal_radius) if cavities.any() else grid


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
    pts: np.ndarray, config: BuildConfig, edits: EditRaster | None, grids_out: list[OccupancyGrid],
) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Rasterize one point group and rebuild it as (polygons, boxes, effective_resolution).
    The occupied grid it was rebuilt from is appended to ``grids_out``."""
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
    if rows * cols > MAX_EDIT_CELLS:
        res = round(res * math.sqrt((rows * cols) / MAX_EDIT_CELLS), 4)
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

    if grid.any() or exact_grid.any():
        occupied = grid if exact_grid is grid else grid | exact_grid
        grids_out.append(OccupancyGrid(occupied, np.array([min_x, min_y]), res))

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


def _project_grid(source: OccupancyGrid, origin: np.ndarray, res: float, shape: tuple[int, int]) -> np.ndarray:
    """Align grid centers to the joint grid, retaining thin occupied cells when downsampling."""
    scale = source.resolution / res
    offset = (source.origin - origin) / res
    shift = offset + (scale - 1) / 2
    transform = np.array([[scale, 0, shift[0]], [0, scale, shift[1]]])
    rows, cols = shape
    projected = cv2.warpAffine(source.cells, transform, (cols, rows), flags=cv2.INTER_NEAREST)
    if scale < 1:
        y, x = np.nonzero(source.cells)
        xx = np.floor(offset[0] + (x + 0.5) * scale + 1e-7).astype(int).clip(0, cols - 1)
        yy = np.floor(offset[1] + (y + 0.5) * scale + 1e-7).astype(int).clip(0, rows - 1)
        projected[yy, xx] = 1
    return projected


def _fill_narrow_pockets(
    occupied: np.ndarray, erased: np.ndarray, res: float, max_cells: int, vehicle_span: float,
    *, edited_boundary: np.ndarray | None = None,
) -> np.ndarray:
    """Fill small pockets separated from outside by a gap too narrow for the AGV."""
    fill = np.zeros_like(occupied)
    if not math.isfinite(vehicle_span) or vehicle_span <= res:
        return fill
    half_span = vehicle_span / (2 * res)
    if min(occupied.shape) < 2 * half_span:
        return fill
    radius = math.ceil(half_span)
    pad = radius + 1
    if math.prod(size + 2 * pad for size in occupied.shape) > 2 * MAX_EDIT_CELLS:
        return fill

    walls = np.pad(occupied, pad)
    free = 1 - walls
    # The distance transform measures from cell centers. Include half a cell for the wall edge.
    clear = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) >= half_span + 0.5 - 1e-7
    _, labels, stats, _ = cv2.connectedComponentsWithStats(clear.astype(np.uint8), connectivity=4)
    outside = labels[0, 0]
    candidates = np.flatnonzero((stats[:, _AREA] <= max_cells) & (stats[:, _AREA] > 0))
    candidates = candidates[(candidates != 0) & (candidates != outside)]
    if not len(candidates):
        return fill

    outside_free = cv2.floodFill(free, None, (0, 0), 2)[1] == 2
    explicit_cuts = np.pad(erased, pad)
    edit_cells = np.pad(edited_boundary, pad) if edited_boundary is not None else None
    result = np.zeros_like(walls)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for label in candidates:
        x, y, width, height, _ = stats[label]
        region = np.s_[y - radius : y + height + radius, x - radius : x + width + radius]
        pocket = labels[region] == label
        # Exact closed loops already use their original area and fill rules.
        if not np.any(pocket & outside_free[region]):
            continue
        grown = cv2.dilate(pocket.astype(np.uint8), kernel)
        traversed = (grown != 0) & outside_free[region]
        if np.count_nonzero(traversed) > max_cells or np.any(traversed & explicit_cuts[region]):
            continue
        if edit_cells is not None and not np.any(_dilate(grown, 1) & edit_cells[region]):
            continue
        result[region][traversed] = 1
    return result[pad:-pad, pad:-pad]


def _joint_cavity_fill(
    drawn: OccupancyGrid, erased: np.ndarray, sources: list[OccupancyGrid], max_area_m2: float, vehicle_span: float,
) -> OccupancyGrid:
    """Find new solid cells using reconstructed imports and edited boundaries together."""
    if not sources:
        max_cells = math.floor((max_area_m2 + 1e-9) / drawn.resolution**2)
        cavities = _enclosed_cells(drawn.cells, max_cells)
        cavities |= _fill_narrow_pockets(drawn.cells, erased, drawn.resolution, max_cells, vehicle_span)
        return OccupancyGrid(cavities & ~erased, drawn.origin, drawn.resolution)

    grids = [drawn, *sources]
    low = np.min([grid.origin for grid in grids], axis=0)
    high = np.max([grid.origin + np.array(grid.cells.shape[::-1]) * grid.resolution for grid in grids], axis=0)
    res = min(grid.resolution for grid in grids)
    while True:
        origin = drawn.origin + (np.floor((low - drawn.origin) / res) - 1) * res
        cols, rows = map(int, np.ceil((high - origin) / res) + 1)
        if cols * rows <= MAX_EDIT_CELLS:
            break
        res *= max(1.01, math.sqrt(cols * rows / MAX_EDIT_CELLS))

    shape = (rows, cols)
    scan = np.zeros(shape, dtype=np.uint8)
    for source in sources:
        scan |= _project_grid(source, origin, res, shape)
    paint = _project_grid(drawn, origin, res, shape)
    erased = _project_grid(OccupancyGrid(erased.astype(np.uint8), drawn.origin, drawn.resolution), origin, res, shape)
    radius = _scan_seal_radius(res)
    scan_barrier = _dilate(scan, radius)
    occupied = scan | paint
    barrier = (scan_barrier | paint) & (1 - erased)
    max_cells = math.floor((max_area_m2 + 1e-9) / res**2)
    cavities = _enclosed_cells(
        barrier,
        max_cells,
        edited_boundary=paint & (1 - (scan_barrier | erased)),
    )
    # Grow into the widened scan boundary, keeping drawn openings and eraser marks intact.
    fill = cavities | (_dilate(cavities, radius) & scan_barrier) if cavities.any() else cavities
    fill |= _fill_narrow_pockets(occupied, erased, res, max_cells, vehicle_span, edited_boundary=paint)
    return OccupancyGrid(fill & (1 - (occupied | erased)), origin, res)


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
    joint_fill = config.fill_solid_obstacles and edits is not None and np.any(edits.cells == DRAW)
    source_grids: list[OccupancyGrid] = []
    for height, pts in _height_groups(canonical, config):
        g_polys, g_boxes, res = _reconstruct_points(pts, config, eraser, source_grids)
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
        occupied = drawn
        if joint_fill:
            filled = _joint_cavity_fill(
                OccupancyGrid(drawn.astype(np.uint8), edits.origin, edits.resolution),
                edits.cells == ERASE, source_grids, config.solid_fill_max_area_m2,
                min(config.vehicle_proxy.length, config.vehicle_proxy.width),
            )
            if source_grids:
                fill_boxes = _grid_boxes(filled.cells, *filled.origin, filled.resolution)
                boxes.append(fill_boxes)
                box_heights += [config.wall_height] * len(fill_boxes)
                eff_res = max(eff_res, filled.resolution)
            else:
                occupied = drawn | filled.cells.astype(bool)
        drawn_boxes = _grid_boxes(occupied, *edits.origin, edits.resolution)
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
    canonical.obstacle_grids = [tuple(g) for g in source_grids]  # drawn/filled cells are in the boxes

    extra = canonical.metadata.extra
    extra["reconstructed_polygon_count"] = len(polygons)
    extra["reconstructed_box_count"] = len(box_heights)
    extra["effective_grid_resolution"] = eff_res
    extra["resolution_was_coarsened"] = eff_res != float(config.grid_resolution)
    extra["reconstruction_mode"] = config.reconstruction_mode
    return canonical


_MAX_PLACEMENT_CELLS = 1_000_000


def nearest_free_point(
    canonical: CanonicalMap,
    x: float,
    y: float,
    clearance: float,
    res: float,
    *,
    include_zones: bool = True,
    dropped_layers: Collection[str] = (),
) -> tuple[float, float]:
    """Return the point nearest (x, y) that keeps ``clearance`` from every reconstructed obstacle
    (and forbidden zone) and from the map edge. Returns (x, y) itself when it is already clear or
    when the map has no such point."""
    shapes = [*canonical.obstacle_polygons, *(z.vertices for z in canonical.forbidden_zones if include_zones)]
    shapes += [np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for x0, y0, x1, y1 in canonical.obstacle_boxes]
    if not shapes:
        return x, y
    bounds = canonical.compute_bounds(dropped_layers)
    res = max(res, math.sqrt(bounds.width * bounds.height / _MAX_PLACEMENT_CELLS))
    origin = np.array([bounds.min_x, bounds.min_y])
    # Pixel (row, col) is centred on origin + (col, row) * res; 0 = blocked, 255 = free.
    free = np.full((math.ceil(bounds.height / res) + 1, math.ceil(bounds.width / res) + 1), 255, np.uint8)
    for shape in shapes:
        if len(shape) >= 3:  # one at a time: fillPoly cancels out overlapping polygons
            cv2.fillPoly(free, [np.rint((shape - origin) / res * 256).astype(np.int32)], 0, shift=8)
    free[[0, -1], :] = 0
    free[:, [0, -1]] = 0
    # Distances run between pixel centres, so allow one extra cell for the obstacle's true edge.
    clear = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res >= clearance + res

    col, row = np.rint((np.array([x, y]) - origin) / res).astype(int)
    if 0 <= row < clear.shape[0] and 0 <= col < clear.shape[1] and clear[row, col]:
        return x, y
    rows, cols = np.nonzero(clear)
    if not len(rows):
        return x, y
    nearest = np.argmin((cols - col) ** 2 + (rows - row) ** 2)
    px, py = origin + np.array([cols[nearest], rows[nearest]]) * res
    return float(px), float(py)


# ---------------------------------------------------------------------------
# Ear-Clipping Triangulation for Extruded USD Meshes
# ---------------------------------------------------------------------------

def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """True when ``p`` lies inside or on the edges of the counter-clockwise triangle ``abc``."""
    return _cross(a, b, p) >= -1e-9 and _cross(b, c, p) >= -1e-9 and _cross(c, a, p) >= -1e-9


def _split_touching_loops(pts: list[tuple[float, float]]) -> list[list[int]]:
    """Split a contour that touches itself (a repeated vertex) into simple loops of vertex indices.

    Cell contours pinch where two parts meet along a zero-width bridge; ear clipping cannot cross
    that bridge, so each side is triangulated on its own.
    """
    def key(i: int) -> tuple[float, float]:
        return round(pts[i][0], 9), round(pts[i][1], 9)

    loops: list[list[int]] = []
    stack: list[int] = []
    seen: dict[tuple[float, float], int] = {}
    for i in range(len(pts)):
        at = seen.get(key(i))
        if at is None:
            seen[key(i)] = len(stack)
            stack.append(i)
            continue
        loops.append(stack[at:])
        for q in stack[at + 1 :]:
            del seen[key(q)]
        del stack[at + 1 :]
    loops.append(stack)
    return loops


def _clip_ears(pts: list[tuple[float, float]], loop: list[int]) -> list[tuple[int, int, int]]:
    """Ear-clip one simple counter-clockwise loop. Only reflex vertices can sit inside an ear, so only
    they are tested. If a full lap finds no clean ear (self-intersecting input), a convex vertex is
    clipped anyway: overlapping triangles are harmless, a missing one leaves a hole in the collider."""
    idx = list(loop)

    def convex(k: int) -> bool:
        return _cross(pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % len(idx)]]) > 1e-12

    reflex = {idx[k] for k in range(len(idx)) if not convex(k)}
    triangles: list[tuple[int, int, int]] = []
    k = stall = 0
    while len(idx) > 3:
        m = len(idx)
        k %= m
        i, j, n = idx[k - 1], idx[k], idx[(k + 1) % m]
        a, b, c = pts[i], pts[j], pts[n]
        clean = j not in reflex and not any(
            q not in (i, j, n) and pts[q] not in (a, b, c) and _point_in_triangle(pts[q], a, b, c)
            for q in reflex
        )
        if clean or (stall >= m and j not in reflex):
            triangles.append((i, j, n))
            idx.pop(k)
            stall = 0
            for pos in ((k - 1) % (m - 1), k % (m - 1)):  # the two neighbours may have turned convex
                (reflex.discard if convex(pos) else reflex.add)(idx[pos])
            k -= 1
        else:
            k += 1
            stall += 1
            if stall >= 2 * m:  # only reflex or collinear vertices left: nothing with area remains
                break
    if len(idx) == 3 and _cross(*(pts[q] for q in idx)) > 1e-12:
        triangles.append((idx[0], idx[1], idx[2]))
    return triangles


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

    triangles: list[tuple[int, int, int]] = []
    for loop in _split_touching_loops(pts):
        # A clockwise sub-loop is a hole touching the outline at one vertex: leave it open.
        if len(loop) >= 3 and _polygon_signed_area([pts[i] for i in loop]) > 1e-12:
            triangles += _clip_ears(pts, loop)
    return pts, triangles
