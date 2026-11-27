"""ROS / ROS 2 Nav2 occupancy grid map parser (.yaml + .pgm/.png or .zip bundle)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from allmaptousd.models import Bounds2D, CanonicalMap, MapMetadata, RoutePath, Station, Zone, as_points
from allmaptousd.parsers.base import MapInputBundle, grid_to_points, load_gray_image

# Extra YAML section written by exporters.ros_map (map_server ignores unknown keys).
PROJECT_KEY = "allmaptousd"


def _classify_cells(
    arr: np.ndarray,
    mode: str,
    negate: int,
    occupied_thresh: float,
    free_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (occupied, free) masks following ROS map_server semantics; the rest is unknown.

    ``arr`` is the grayscale image in [0, 1]. In ``trinary`` / ``scale`` modes occupancy is
    ``1 - pixel`` (or ``pixel`` when ``negate``); in ``raw`` mode the 0-255 pixel value *is*
    the occupancy percentage (0-100), anything above 100 is unknown.
    """
    if mode == "raw":
        value = np.rint(arr * 255.0)
        return (value >= occupied_thresh * 100.0) & (value <= 100.0), value < free_thresh * 100.0
    occ = arr if negate else (1.0 - arr)
    return occ >= occupied_thresh, occ < free_thresh


def _free_space_frontier(occupied: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Cells that close the known free space: unknown cells touching free cells, plus free
    cells on the image border (the map simply ends there). Used by ``BuildConfig.seal_unknown``.
    """
    unknown = ~occupied & ~free
    near_free = cv2.dilate(free.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    border = np.zeros_like(free)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    return (unknown & near_free) | (free & border)


class RosOccupancyParser:
    format_id = "ros_occupancy"
    display_name = "ROS / Nav2 (.yaml + .pgm/.png)"
    supported_extensions = (".yaml", ".yml", ".zip")

    def can_parse(self, bundle: MapInputBundle) -> bool:
        ext = Path(bundle.primary_filename).suffix.lower()
        if ext in (".yaml", ".yml"):
            return True
        if ext == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(bundle.primary_bytes)) as zf:
                    names = [n.lower() for n in zf.namelist()]
                    return any(n.endswith((".yaml", ".yml")) for n in names)
            except Exception:
                return False
        return False

    def _load_yaml_and_image(
        self, bundle: MapInputBundle
    ) -> tuple[dict[str, Any], bytes, str, dict[str, bytes]]:
        """(map yaml, image bytes, default map name, the saved 3D models a project zip carries)."""
        ext = Path(bundle.primary_filename).suffix.lower()

        if ext == ".zip":
            with zipfile.ZipFile(io.BytesIO(bundle.primary_bytes)) as zf:
                yaml_names = [
                    n for n in zf.namelist() if n.lower().endswith((".yaml", ".yml"))
                ]
                if not yaml_names:
                    raise ValueError("ZIP archive does not contain a ROS map .yaml file.")
                yaml_name = yaml_names[0]
                meta = yaml.safe_load(zf.read(yaml_name).decode("utf-8-sig"))
                image_ref = str(meta.get("image", ""))
                # Match by exact path or basename inside zip
                img_bytes = None
                for candidate in zf.namelist():
                    if candidate == image_ref or Path(candidate).name == Path(image_ref).name:
                        img_bytes = zf.read(candidate)
                        break
                if img_bytes is None:
                    raise ValueError(
                        f"Image '{image_ref}' referenced by '{yaml_name}' was not found in ZIP."
                    )
                models = {n: zf.read(n) for n in zf.namelist() if n.startswith("models/") and not n.endswith("/")}
                return meta, img_bytes, Path(yaml_name).stem, models

        meta = yaml.safe_load(bundle.primary_bytes.decode("utf-8-sig"))
        if not isinstance(meta, dict) or "image" not in meta:
            raise ValueError(
                f"'{bundle.primary_filename}' does not look like a valid ROS map YAML (missing 'image')."
            )
        image_ref = str(meta["image"])
        image_basename = Path(image_ref).name

        # 1. Check uploaded companion files
        for fname, fbytes in bundle.companion_files.items():
            if fname == image_ref or Path(fname).name == image_basename:
                return meta, fbytes, Path(bundle.primary_filename).stem, {}

        # 2. Check sibling file on disk if loaded from path
        if bundle.source_dir is not None:
            candidate = bundle.source_dir / image_ref
            if not candidate.exists():
                candidate = bundle.source_dir / image_basename
            if candidate.exists():
                return meta, candidate.read_bytes(), Path(bundle.primary_filename).stem, {}

        raise ValueError(
            f"ROS map image '{image_ref}' is missing. "
            f"Please upload both the .yaml and '{image_basename}' together (or as a .zip)."
        )

    def parse(self, bundle: MapInputBundle) -> CanonicalMap:
        meta, img_bytes, default_name, model_files = self._load_yaml_and_image(bundle)

        resolution = float(meta.get("resolution", 0.05))
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        ox = float(origin[0]) if len(origin) > 0 else 0.0
        oy = float(origin[1]) if len(origin) > 1 else 0.0
        oyaw = float(origin[2]) if len(origin) > 2 else 0.0
        negate = int(meta.get("negate", 0))
        occupied_thresh = float(meta.get("occupied_thresh", 0.65))
        free_thresh = float(meta.get("free_thresh", 0.196))
        mode = str(meta.get("mode", "trinary")).lower()

        arr = load_gray_image(img_bytes, str(meta.get("image", "")))
        height_px, width_px = arr.shape
        occupied, free = _classify_cells(arr, mode, negate, occupied_thresh, free_thresh)
        frontier = _free_space_frontier(occupied, free)

        points = grid_to_points(occupied, resolution, ox, oy, oyaw)
        frontier_points = grid_to_points(frontier, resolution, ox, oy, oyaw)

        canonical = CanonicalMap(
            metadata=MapMetadata(
                name=str(meta.get("map_name") or default_name),
                source_format=self.format_id,
                resolution=resolution,
                extra={
                    "image": str(meta.get("image", "")),
                    "origin": [ox, oy, oyaw],
                    "width_px": width_px,
                    "height_px": height_px,
                    "occupied_thresh": occupied_thresh,
                    "free_thresh": free_thresh,
                    "mode": mode,
                    "negate": negate,
                    "unknown_frontier_cells": len(frontier_points),
                    "raw_point_count": len(points),
                },
            ),
            points=points,
            frontier_points=frontier_points,
        )
        project = meta.get(PROJECT_KEY)
        if isinstance(project, dict):
            _restore_project(canonical, project, resolution, (ox, oy, oyaw), (width_px, height_px), model_files)
        return canonical


def _restore_project(
    canonical: CanonicalMap, project: dict, res: float, origin: tuple[float, float, float], size: tuple[int, int],
    model_files: dict[str, bytes],
) -> None:
    """Bring back what an AllMapToUSD save keeps next to the occupancy image.

    The saved build settings and 3D models travel in ``metadata.extra["project_build"]`` and
    ``["project_models"]`` for ``MapSession`` to apply.
    """
    try:
        canonical.forbidden_zones = [
            Zone(name=str(z["name"]), zone_type=str(z.get("zone_type", "forbidden")), vertices=as_points(z["vertices"]))
            for z in project.get("forbidden_zones") or []
        ]
        canonical.stations = [
            Station(name=str(s["name"]), x=float(s["x"]), y=float(s["y"]), yaw=float(s.get("yaw", 0.0)),
                    kind=str(s.get("kind", "Station")))
            for s in project.get("stations") or []
        ]
        canonical.paths = [
            RoutePath(name=str(p["name"]), points=as_points(p["points"]), kind=str(p.get("kind", "Path")))
            for p in project.get("paths") or []
        ]
        models = [
            {
                "id": str(m["id"]), "name": str(m["name"]), "data": model_files[str(m["file"])],
                "files": {
                    n[len(f"models/{m['id']}/"):]: data for n, data in model_files.items()
                    if n.startswith(f"models/{m['id']}/")
                },
            }
            for m in project.get("models") or []
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The saved '{PROJECT_KEY}' section of this map is damaged: {exc}") from exc
    ox, oy, yaw = origin
    if abs(yaw) < 1e-12:  # keep the full saved canvas, including empty space around the obstacles
        canonical.drawing_bounds = Bounds2D(ox, oy, ox + size[0] * res, oy + size[1] * res)
    canonical.metadata.name = str(project.get("map_name") or canonical.metadata.name)
    canonical.metadata.extra["saved_source_format"] = str(project.get("source_format", ""))
    if isinstance(project.get("build"), dict):
        canonical.metadata.extra["project_build"] = project["build"]
    if models:
        canonical.metadata.extra["project_models"] = models
