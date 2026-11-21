"""Parser plugin interface and registry for dependency injection."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from allmaptousd.models import CanonicalMap


def load_gray_image(data: bytes, filename: str) -> np.ndarray:
    """Decode an image into a float32 array in [0, 1] (row 0 = top of image)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("L")
    except Exception as exc:
        raise ValueError(f"Failed to open raster image '{filename}': {exc}") from exc
    return np.asarray(img, dtype=np.float32) / 255.0


def grid_to_points(
    occupied: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float,
    yaw: float = 0.0,
) -> np.ndarray:
    """Convert an image-style occupancy mask (row 0 = top / max Y) into (N, 2) world cell centers."""
    rows, cols = np.nonzero(occupied)
    lx = (cols + 0.5) * resolution
    ly = (occupied.shape[0] - 1 - rows + 0.5) * resolution
    if abs(yaw) > 1e-12:
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = lx * c - ly * s, lx * s + ly * c
    return np.column_stack((origin_x + lx, origin_y + ly)).astype(np.float64)


@dataclass
class MapInputBundle:
    """Unified input container supporting file paths or in-memory uploaded bytes."""

    primary_filename: str
    primary_bytes: bytes
    companion_files: dict[str, bytes] = field(default_factory=dict)
    source_dir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        options: dict[str, Any] | None = None,
    ) -> MapInputBundle:
        p = Path(path)
        return cls(
            primary_filename=p.name,
            primary_bytes=p.read_bytes(),
            source_dir=p.parent,
            options=dict(options or {}),
        )


class MapParser(Protocol):
    """Contract that every vendor-specific map parser implements."""

    format_id: str
    display_name: str
    supported_extensions: tuple[str, ...]

    def can_parse(self, bundle: MapInputBundle) -> bool:
        ...

    def parse(self, bundle: MapInputBundle) -> CanonicalMap:
        ...


class ParserRegistry:
    """Registry allowing vendor parsers to be injected and resolved dynamically."""

    def __init__(self) -> None:
        self._parsers: dict[str, MapParser] = {}

    def register(self, parser: MapParser) -> None:
        self._parsers[parser.format_id] = parser

    def get(self, format_id: str) -> MapParser:
        if format_id not in self._parsers:
            raise KeyError(f"Unknown map parser format_id: {format_id!r}")
        return self._parsers[format_id]

    def list_parsers(self) -> list[dict]:
        return [
            {
                "format_id": p.format_id,
                "display_name": p.display_name,
                "supported_extensions": list(p.supported_extensions),
            }
            for p in self._parsers.values()
        ]

    def resolve(self, bundle: MapInputBundle, format_hint: str | None = None) -> MapParser:
        if format_hint and format_hint != "auto":
            return self.get(format_hint)
        for parser in self._parsers.values():
            if parser.can_parse(bundle):
                return parser
        supported = ", ".join(
            ext for p in self._parsers.values() for ext in p.supported_extensions
        )
        raise ValueError(
            f"No registered parser can handle '{bundle.primary_filename}'. "
            f"Supported extensions: {supported}"
        )

    def parse(self, bundle: MapInputBundle, format_hint: str | None = None) -> CanonicalMap:
        parser = self.resolve(bundle, format_hint=format_hint)
        return parser.parse(bundle)
