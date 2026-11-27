"""Convert an Isaac Sim stage slice (see ``exporters.usd_slice``) into canonical occupancy-grid points."""

from __future__ import annotations

import base64
import zlib

import numpy as np

from allmaptousd.exporters.usd_slice import MAX_GRID_CELLS
from allmaptousd.models import CanonicalMap, MapMetadata


def map_from_stage_slice(scene: dict, requested_resolution: float, low: float, high: float) -> CanonicalMap:
    """Decode the bit-packed occupancy grid rasterized inside Isaac Sim into occupied cell centres."""
    if not scene.get("grid_b64"):
        ignored = scene.get("unsupported", 0)
        suffix = f" ({ignored} unsupported geometry prims skipped)" if ignored else ""
        raise ValueError(f"No obstacles in the selected height range {low:g}–{high:g} m{suffix}.")

    rows, cols = (int(v) for v in scene["grid_shape"])
    origin = np.asarray(scene["grid_origin"], dtype=np.float64)
    res = float(scene["grid_resolution"])
    if not (0 < rows * cols <= MAX_GRID_CELLS and res > 0 and np.isfinite(origin).all()):
        raise ValueError("Isaac Sim returned an invalid occupancy grid.")
    if np.max(np.abs(origin)) > 1e7 or max(rows, cols) * res > 1e7:
        raise ValueError("The stage coordinates are outside the map editor range.")

    bits = np.unpackbits(np.frombuffer(zlib.decompress(base64.b64decode(scene["grid_b64"], validate=True)), np.uint8))
    if len(bits) < rows * cols:
        raise ValueError("Isaac Sim returned an invalid occupancy grid.")
    y, x = np.nonzero(bits[: rows * cols].reshape((rows, cols)))
    if not len(x):
        raise ValueError("The selected geometry did not cover any map cells.")

    return CanonicalMap(
        metadata=MapMetadata(
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
        ),
        points=origin + (np.column_stack((x, y)) + 0.5) * res,
    )
