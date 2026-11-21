"""CAD / SICK SMET (.dxf) vector floorplan parser using ezdxf."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import ezdxf
import numpy as np
from ezdxf import path as dxf_path

from allmaptousd.models import CanonicalMap, MapMetadata, as_points
from allmaptousd.parsers.base import MapInputBundle

# AutoCAD $INSUNITS code -> scale factor to meters
_INSUNITS_TO_METERS: dict[int, float] = {
    1: 0.0254,  # Inches
    2: 0.3048,  # Feet
    4: 0.001,   # Millimeters
    5: 0.01,    # Centimeters
    6: 1.0,     # Meters
}

_UNIT_OVERRIDE_TO_METERS: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "ft": 0.3048,
}

# Curve entities converted via ezdxf.path (handles LWPOLYLINE bulges, closed flags, splines).
_GEOMETRY_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "CIRCLE", "ARC", "ELLIPSE", "SPLINE"}
_MAX_BLOCK_DEPTH = 8
_FLATTEN_TOL_M = 0.01  # max chord deviation when sampling curves


def _iter_geometry(entities, parent_layer: str = "0", depth: int = 0) -> Iterator[tuple[str, object]]:
    """Yield (layer, entity) for wall geometry, recursively exploding block references (INSERT).

    Block content drawn on layer "0" inherits the layer of its INSERT (standard CAD behaviour).
    """
    for entity in entities:
        layer = str(entity.dxf.get("layer", "0"))
        if layer == "0":
            layer = parent_layer
        etype = entity.dxftype()
        if etype == "INSERT" and depth < _MAX_BLOCK_DEPTH:
            try:
                yield from _iter_geometry(entity.virtual_entities(), layer, depth + 1)
            except Exception:
                continue
        elif etype in _GEOMETRY_TYPES:
            yield layer, entity


class DxfMapParser:
    format_id = "dxf_cad"
    display_name = "CAD / SICK SMET (.dxf)"
    supported_extensions = (".dxf",)

    def can_parse(self, bundle: MapInputBundle) -> bool:
        return Path(bundle.primary_filename).suffix.lower() == ".dxf"

    def parse(self, bundle: MapInputBundle) -> CanonicalMap:
        try:
            text = (
                bundle.primary_bytes.decode("utf-8-sig", errors="replace")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )
            doc = ezdxf.read(io.StringIO(text))
        except Exception as exc:
            raise ValueError(f"Failed to read DXF file '{bundle.primary_filename}': {exc}") from exc

        unit_opt = str(bundle.options.get("dxf_units", "auto")).lower()
        if unit_opt in _UNIT_OVERRIDE_TO_METERS:
            scale = _UNIT_OVERRIDE_TO_METERS[unit_opt]
            unit_label = unit_opt
        else:
            insunits = int(doc.header.get("$INSUNITS", 6))
            scale = _INSUNITS_TO_METERS.get(insunits, 1.0)
            unit_label = f"insunits:{insunits}"

        flatten_tol = _FLATTEN_TOL_M / max(scale, 1e-9)  # in drawing units
        chunks: list[np.ndarray] = []
        layer_ids: list[np.ndarray] = []
        layer_index: dict[str, int] = {}
        for layer, entity in _iter_geometry(doc.modelspace()):
            try:
                for sub in dxf_path.make_path(entity).sub_paths():
                    pts = np.array([(v.x, v.y) for v in sub.flattening(flatten_tol)], dtype=np.float64)
                    if len(pts) >= 2:
                        chunks.append(np.stack((pts[:-1], pts[1:]), axis=1))
                        lid = layer_index.setdefault(layer, len(layer_index))
                        layer_ids.append(np.full(len(pts) - 1, lid, dtype=np.int32))
            except Exception:
                continue

        if not chunks:
            raise ValueError(
                f"No supported 2D wall geometry (LINE/POLYLINE/CIRCLE/ARC/SPLINE) found in '{bundle.primary_filename}'."
            )
        segments = np.concatenate(chunks) * scale
        segment_layers = np.concatenate(layer_ids)
        layer_names = list(layer_index)

        return CanonicalMap(
            metadata=MapMetadata(
                name=Path(bundle.primary_filename).stem,
                source_format=self.format_id,
                resolution=0.05,
                extra={
                    "dxf_units": unit_label,
                    "scale_to_meters": scale,
                    "raw_segment_count": len(segments),
                    "layers": {
                        name: int((segment_layers == i).sum()) for i, name in enumerate(layer_names)
                    },
                },
            ),
            segments=as_points(segments, (2, 2)),
            segment_layers=segment_layers,
            layer_names=layer_names,
        )
