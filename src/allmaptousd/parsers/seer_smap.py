"""SEER Robotics (.smap) parser.

Supports SEER JSON-serialized .smap files (conforming to message_map.proto schema):
- header: mapName, resolution, minPos, maxPos
- normalPosList: 2D laser scan obstacle points
- normalLineList / advancedLineList: optional wall/virtual-wall line segments
- advancedAreaList: forbidden/restricted areas treated as virtual obstacles
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from allmaptousd.models import CanonicalMap, MapMetadata, RoutePath, Station, Zone, as_points
from allmaptousd.parsers.base import MapInputBundle

# Matched against whole words (CamelCase, snake_case and spaces split; "NoGo" also joins to "nogo"),
# so "UnrestrictedArea", "Banner" or "SpeedArea" are not mistaken for no-go zones.
_FORBIDDEN_WORD_PREFIXES = ("forbid", "nogo", "restrict", "block", "obstacle", "virtualwall", "danger")
_FORBIDDEN_WORDS = frozenset({"ban", "banned"})
_FORBIDDEN_FLAG_KEYS = frozenset({"forbidden", "isForbidden", "forbid"})
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _words(text: str) -> list[str]:
    words = [w.lower() for w in _WORD.findall(text)]
    return words + [a + b for a, b in zip(words, words[1:])]


def _is_forbidden_area(area: dict[str, Any]) -> bool:
    """Heuristic check for SEER advancedAreaList entries that represent forbidden zones."""
    texts = [str(area.get(key, "")) for key in ("className", "instanceName", "areaType")]
    for prop in area.get("property", []) or []:
        if not isinstance(prop, dict):
            continue
        if prop.get("key") in _FORBIDDEN_FLAG_KEYS:  # an explicit flag wins either way
            if "boolValue" in prop:
                return bool(prop["boolValue"])
            continue
        texts += [str(prop.get("key", "")), str(prop.get("stringValue", ""))]
    if any("禁" in text for text in texts):
        return True
    words = [w for text in texts for w in _words(text)]
    return any(w in _FORBIDDEN_WORDS or w.startswith(_FORBIDDEN_WORD_PREFIXES) for w in words)


def _xy(pos: Any) -> tuple[float, float] | None:
    """{"x": .., "y": ..} or [x, y, ...] -> (x, y); anything else -> None."""
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        return float(pos["x"]), float(pos["y"])
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _xy_list(items: Any) -> list[tuple[float, float]]:
    return [xy for item in items or [] if (xy := _xy(item)) is not None]


def _extract_polygon_points(area: dict[str, Any]) -> list[tuple[float, float]]:
    return _xy_list(
        area.get("posGroup") or area.get("points") or area.get("polygon") or area.get("vertexList")
    )


def _parse_stations(items: list) -> list[Station]:
    """advancedPointList: LocationMark / ChargePoint / ParkPoint / ActionPoint ... with pos + dir."""
    stations: list[Station] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict) or (xy := _xy(item.get("pos"))) is None:
            continue
        stations.append(
            Station(
                name=str(item.get("instanceName") or f"Station_{idx + 1}"),
                x=xy[0],
                y=xy[1],
                yaw=float(item.get("dir") or 0.0),
                kind=str(item.get("className") or "Station"),
            )
        )
    return stations


def _sample_bezier(ctrl: list[tuple[float, float]], n: int = 24) -> np.ndarray:
    """Evaluate a Bezier curve of any degree (de Casteljau) at n+1 samples."""
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = [np.asarray(c, dtype=np.float64) for c in ctrl]
    while len(pts) > 1:
        pts = [(1.0 - t) * a + t * b for a, b in zip(pts[:-1], pts[1:])]
    return pts[0]


def _parse_paths(items: list) -> list[RoutePath]:
    """advancedCurveList: StraightPath / BezierPath / ArcPath ... between two stations.

    Control points (``controlPos1``/``controlPos2``) are sampled as a Bezier curve; ArcPath's
    single control point is approximated by a quadratic Bezier.
    """
    paths: list[RoutePath] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        start = _xy((item.get("startPos") or {}).get("pos"))
        end = _xy((item.get("endPos") or {}).get("pos"))
        if start is None or end is None:
            continue
        ctrl = [c for c in (_xy(item.get("controlPos1")), _xy(item.get("controlPos2"))) if c]
        pts = _sample_bezier([start, *ctrl, end]) if ctrl else np.array([start, end], dtype=np.float64)
        paths.append(
            RoutePath(
                name=str(item.get("instanceName") or f"Path_{idx + 1}"),
                points=pts,
                kind=str(item.get("className") or "Path"),
            )
        )
    return paths


_SEER_TOP_LEVEL_KEYS = frozenset(
    {"normalPosList", "normalLineList", "advancedAreaList", "advancedPointList", "advancedCurveList"}
)


class SeerSmapParser:
    format_id = "seer_smap"
    display_name = "SEER Robotics (.smap)"
    supported_extensions = (".smap", ".json")

    def can_parse(self, bundle: MapInputBundle) -> bool:
        ext = Path(bundle.primary_filename).suffix.lower()
        if ext == ".smap":
            return True
        if ext == ".json":
            # Only claim JSON whose top level carries SEER map lists (not just a keyword somewhere).
            try:
                data = json.loads(bundle.primary_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, ValueError):
                return False
            return isinstance(data, dict) and bool(_SEER_TOP_LEVEL_KEYS & data.keys())
        return False

    def parse(self, bundle: MapInputBundle) -> CanonicalMap:
        try:
            text = bundle.primary_bytes.decode("utf-8-sig")
            data = json.loads(text)
        except Exception as exc:
            raise ValueError(
                f"Failed to decode SEER .smap JSON from '{bundle.primary_filename}': {exc}. "
                "Note: Binary Protobuf .smap files should be converted to JSON first."
            ) from exc

        header = data.get("header") or {}
        map_name = str(header.get("mapName") or Path(bundle.primary_filename).stem)
        resolution = float(header.get("resolution") or 0.05)

        # 1. Obstacle point cloud (normalPosList)
        points = as_points(_xy_list(data.get("normalPosList")))

        # 2. Optional line segments (normalLineList / virtual wall lines)
        segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for line_key in ("normalLineList", "advancedLineList"):
            for item in data.get(line_key) or []:
                if not isinstance(item, dict):
                    continue
                line_obj = item.get("line") if isinstance(item.get("line"), dict) else item
                p0, p1 = _xy(line_obj.get("startPos")), _xy(line_obj.get("endPos"))
                if p0 and p1:
                    segments.append((p0, p1))

        # 3. Forbidden zones (advancedAreaList)
        forbidden_zones: list[Zone] = []
        for idx, area in enumerate(data.get("advancedAreaList") or []):
            if not isinstance(area, dict):
                continue
            if not _is_forbidden_area(area):
                continue
            verts = _extract_polygon_points(area)
            if len(verts) >= 3:
                zone_name = str(
                    area.get("instanceName")
                    or area.get("className")
                    or f"ForbiddenZone_{idx + 1}"
                )
                forbidden_zones.append(
                    Zone(
                        name=zone_name,
                        zone_type=str(area.get("className") or "ForbiddenArea"),
                        vertices=as_points(verts),
                    )
                )

        stations = _parse_stations(data.get("advancedPointList") or [])
        paths = _parse_paths(data.get("advancedCurveList") or [])

        return CanonicalMap(
            metadata=MapMetadata(
                name=map_name,
                source_format=self.format_id,
                resolution=resolution,
                extra={
                    "version": header.get("version", ""),
                    "mapType": header.get("mapType", "2D-Map"),
                    "raw_point_count": len(points),
                    "raw_segment_count": len(segments),
                    "forbidden_zone_count": len(forbidden_zones),
                    "station_count": len(stations),
                    "path_count": len(paths),
                },
            ),
            points=points,
            segments=as_points(segments, (2, 2)),
            forbidden_zones=forbidden_zones,
            stations=stations,
            paths=paths,
        )
