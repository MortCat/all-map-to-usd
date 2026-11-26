"""Convert an Isaac Sim stage slice into canonical occupancy-grid points."""

from __future__ import annotations

import math

import cv2
import numpy as np

from allmaptousd.models import CanonicalMap, MapMetadata, as_points

_MAX_CELLS = 4_000_000
_SHIFT = 8


def map_from_stage_slice(scene: dict, requested_resolution: float, low: float, high: float) -> CanonicalMap:
    faces = [as_points(face) for face in scene["projected"]]
    sections = [[as_points(loop) for loop in section] for section in scene["sections"]]
    contours = [*faces, *(loop for section in sections for loop in section)]
    if not contours:
        ignored = scene.get("unsupported", 0)
        suffix = f" ({ignored} unsupported geometry prims skipped)" if ignored else ""
        raise ValueError(f"No obstacles in the selected height range {low:g}–{high:g} m{suffix}.")
    xy = np.concatenate(contours)
    if not np.isfinite(xy).all():
        raise ValueError("The scene contains non-finite coordinates.")
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    extent = hi - lo
    res = requested_resolution
    while True:
        # Keep the edit grid and the occupancy grid aligned to the stage origin.
        origin = np.floor(lo / res) * res
        cols, rows = [math.ceil(length / res) + 2 for length in hi - origin]
        if rows * cols <= _MAX_CELLS:
            break
        if res >= 1.0:
            raise ValueError("The scene slice is too large for the map grid. Select a smaller source prim.")
        res = min(1.0, math.ceil(max(res * 1.1, res * math.sqrt(rows * cols / _MAX_CELLS)) * 1000) / 1000)
    if np.max(np.abs(origin)) > 1e7 or np.max(extent) > 1e7:
        raise ValueError("The stage coordinates are outside the map editor range.")

    def pixels(contour: np.ndarray) -> np.ndarray:
        return np.rint(((contour - origin) / res - 0.5) * (1 << _SHIFT)).astype(np.int32)

    grid = np.zeros((rows, cols), dtype=np.uint8)
    for face in faces:
        cv2.fillPoly(grid, [pixels(face)], color=1, shift=_SHIFT)
    for section in sections:
        # Fill the loops together to preserve the holes of closed shapes.
        cv2.fillPoly(grid, [pixels(loop) for loop in section], color=1, shift=_SHIFT)
    y, x = np.nonzero(grid)
    if not len(x):
        raise ValueError("The selected geometry did not cover any map cells.")
    points = origin + (np.column_stack((x, y)) + 0.5) * res
    metadata = MapMetadata(
        name=scene["stage"],
        source_format="isaac_stage",
        resolution=res,
        extra={
            "isaac_height_range_m": [low, high],
            "isaac_source_prim": scene["root_prim_path"],
            "isaac_up_axis": scene["up_axis"],
            "isaac_projected_prims": scene["prim_count"],
            "isaac_unsupported_prims": scene["unsupported"],
            "isaac_requested_resolution_m": requested_resolution,
        },
    )
    return CanonicalMap(metadata=metadata, points=points)
