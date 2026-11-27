"""Save the obstacle map as a ROS / Nav2 occupancy map (``.yaml`` + ``.pgm``, zipped).

The same file is the project save: the YAML carries the standard map_server keys, which Nav2 reads,
plus an ``allmaptousd`` section (ignored by map_server) with the build settings, AGV proxy, placed
3D models, forbidden zones, stations and routes. The imported 3D models themselves are stored as
``models/<id>.usdc``, and the files each one uses (textures, materials) under ``models/<id>/``.
Loading the zip back through the ROS parser restores all of it.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping

import numpy as np
import yaml
from PIL import Image

from allmaptousd.geometry.editing import MAX_EDIT_CELLS
from allmaptousd.model_assets import ModelAsset
from allmaptousd.models import BuildConfig, CanonicalMap
from allmaptousd.parsers.ros_occupancy import PROJECT_KEY

PROJECT_VERSION = 1
_OCCUPIED, _FREE = 0, 254  # map_saver trinary values (unknown would be 205; saved maps have none)


def rasterize_obstacles(canonical: CanonicalMap, config: BuildConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """Occupancy grid of the reconstructed obstacles: (cells, origin_xy, resolution), row 0 = min y.

    It stores the grids the obstacles were traced from (plus the drawn cells), not the traced
    outlines: reopening the file traces the same grid again, so a save/reopen cycle never changes
    the obstacles. The saved grid reuses the lattice those grids were built on.
    """
    res = float(canonical.metadata.extra.get("effective_grid_resolution", config.grid_resolution))
    grids = canonical.obstacle_grids
    on_lattice = [g for g in grids if abs(g[2] - res) < 1e-9]
    phase = np.mod(on_lattice[0][1], res) if on_lattice else np.zeros(2)

    b = canonical.compute_bounds(config.dropped_layers())
    extent = [np.array([b.min_x, b.min_y]), np.array([b.max_x, b.max_y])]
    for cells_g, origin_g, res_g in grids:
        rows_g, cols_g = np.nonzero(cells_g)
        if len(rows_g):
            extent += [origin_g + np.array([cols_g.min(), rows_g.min()]) * res_g,
                       origin_g + np.array([cols_g.max() + 1, rows_g.max() + 1]) * res_g]
    extent = np.array(extent)
    lo = np.floor((extent.min(axis=0) - phase) / res + 1e-6)
    hi = np.ceil((extent.max(axis=0) - phase) / res - 1e-6)
    cols, rows = np.maximum(hi - lo, 1).astype(int)
    if rows * cols > MAX_EDIT_CELLS:
        raise ValueError("The map is too large to save at this grid resolution. Use a coarser resolution.")
    origin = lo * res + phase
    cells = np.zeros((rows, cols), np.uint8)

    centres_x = origin[0] + (np.arange(cols) + 0.5) * res
    centres_y = origin[1] + (np.arange(rows) + 0.5) * res
    for cells_g, origin_g, res_g in grids:  # each saved cell takes the grid cell under its centre
        gc = np.floor((centres_x - origin_g[0]) / res_g).astype(int)
        gr = np.floor((centres_y - origin_g[1]) / res_g).astype(int)
        in_c = np.flatnonzero((gc >= 0) & (gc < cells_g.shape[1]))
        in_r = np.flatnonzero((gr >= 0) & (gr < cells_g.shape[0]))
        if len(in_c) and len(in_r):
            cells[np.ix_(in_r, in_c)] |= cells_g[np.ix_(gr[in_r], gc[in_c])]
    for x0, y0, x1, y1 in canonical.obstacle_boxes:  # drawn and filled cells: on the lattice already
        c0, r0 = np.rint((np.array([x0, y0]) - origin) / res).astype(int)
        c1, r1 = np.rint((np.array([x1, y1]) - origin) / res).astype(int)
        cells[max(r0, 0) : max(r1, r0 + 1), max(c0, 0) : max(c1, c0 + 1)] = 1
    return cells, origin, res


def _project_section(canonical: CanonicalMap, config: BuildConfig, models: Mapping[str, ModelAsset]) -> dict:
    def xy_list(points: np.ndarray) -> list[list[float]]:
        return np.round(points, 4).tolist()

    return {
        "version": PROJECT_VERSION,
        "map_name": canonical.metadata.name,
        "source_format": canonical.metadata.source_format,
        "build": config.model_dump(mode="json"),
        "forbidden_zones": [
            {"name": z.name, "zone_type": z.zone_type, "vertices": xy_list(z.vertices)}
            for z in canonical.forbidden_zones
        ],
        "stations": [
            {"name": s.name, "kind": s.kind, "x": round(s.x, 4), "y": round(s.y, 4), "yaw": round(s.yaw, 6)}
            for s in canonical.stations
        ],
        "paths": [{"name": p.name, "kind": p.kind, "points": xy_list(p.points)} for p in canonical.paths],
        "models": [{"id": m.id, "name": m.name, "file": f"models/{m.id}.usdc"} for m in models.values()],
    }


def export_ros_map_zip(
    canonical: CanonicalMap, config: BuildConfig, stem: str, models: Mapping[str, ModelAsset] | None = None,
) -> bytes:
    """Zip ``<stem>.yaml`` + ``<stem>.pgm``: a Nav2 map that this app reopens with its settings
    (and with the imported 3D models in ``models``)."""
    models = models or {}
    cells, origin, res = rasterize_obstacles(canonical, config)
    image = np.where(cells[::-1] != 0, _OCCUPIED, _FREE).astype(np.uint8)  # image row 0 = max y
    pgm = io.BytesIO()
    Image.fromarray(image).save(pgm, format="PPM")  # an 8-bit grey image is written as binary PGM (P5)

    meta = {
        "image": f"{stem}.pgm",
        "mode": "trinary",
        "resolution": res,
        "origin": [float(origin[0]), float(origin[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        PROJECT_KEY: _project_section(canonical, config, models),
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{stem}.yaml", yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
        zf.writestr(f"{stem}.pgm", pgm.getvalue())
        for model in models.values():
            zf.writestr(f"models/{model.id}.usdc", model.to_usdc())
            for rel, data in model.files.items():
                zf.writestr(f"models/{model.id}/{rel}", data)
    return out.getvalue()
