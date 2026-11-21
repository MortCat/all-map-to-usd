"""Canonical intermediate representation (IR) for AGV/AMR maps + build/connection configs.

Geometry lives in numpy arrays (meters, world frame) so large SLAM maps with millions of
occupied cells stay cheap to parse, copy, and rebuild. Only user-facing configs that cross
the HTTP API are Pydantic models.
"""

from __future__ import annotations

import copy
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


def _empty(*shape: int) -> np.ndarray:
    return np.zeros((0, *shape), dtype=np.float64)


def as_points(data, ndim_tail: tuple[int, ...] = (2,)) -> np.ndarray:
    """Coerce a sequence of coordinates into a float64 array of shape (N, *ndim_tail)."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return _empty(*ndim_tail)
    return arr.reshape((-1, *ndim_tail))


@dataclass
class Bounds2D:
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    @property
    def center(self) -> tuple[float, float]:
        return 0.5 * (self.min_x + self.max_x), 0.5 * (self.min_y + self.max_y)


@dataclass
class MapMetadata:
    name: str = "UntitledMap"
    source_format: str = "unknown"
    resolution: float = 0.05  # meters per cell
    extra: dict = field(default_factory=dict)


@dataclass
class Zone:
    """Named 2D polygon (CCW or CW) in world meters: vertices shape (K, 2)."""

    name: str
    vertices: np.ndarray
    zone_type: str = "forbidden"


@dataclass
class Station:
    """Named robot pose (charging point, location mark, action point, ...)."""

    name: str
    x: float
    y: float
    yaw: float = 0.0  # radians
    kind: str = "Station"


@dataclass
class RoutePath:
    """Named navigation route sampled as a polyline: points shape (K, 2)."""

    name: str
    points: np.ndarray
    kind: str = "Path"


@dataclass
class CanonicalMap:
    """Unified IR shared across all parsers and exporters.

    Parsers fill the inputs (``points``, ``segments`` [+ ``segment_layers``], ``forbidden_zones``,
    ``stations``, ``paths``); ``geometry.reconstruct_obstacles`` fills the obstacle outputs.
    """

    metadata: MapMetadata = field(default_factory=MapMetadata)
    points: np.ndarray = field(default_factory=lambda: _empty(2))        # (N, 2)
    segments: np.ndarray = field(default_factory=lambda: _empty(2, 2))   # (M, 2, 2)
    # Occupancy-grid sources: cells closing the known free space (used when seal_unknown)
    frontier_points: np.ndarray = field(default_factory=lambda: _empty(2))  # (F, 2)
    # Optional CAD layer index per segment (M,), indexing ``layer_names``
    segment_layers: np.ndarray | None = None
    layer_names: list[str] = field(default_factory=list)
    forbidden_zones: list[Zone] = field(default_factory=list)
    stations: list[Station] = field(default_factory=list)
    paths: list[RoutePath] = field(default_factory=list)
    # Reconstructed outputs; heights are per item (meters)
    obstacle_polygons: list[np.ndarray] = field(default_factory=list)    # each (K, 2)
    obstacle_polygon_heights: list[float] = field(default_factory=list)
    obstacle_boxes: np.ndarray = field(default_factory=lambda: _empty(4))  # (B, 4) x0,y0,x1,y1
    obstacle_box_heights: list[float] = field(default_factory=list)

    def copy(self) -> CanonicalMap:
        # Input arrays are never mutated in place, so sharing them is safe.
        return CanonicalMap(
            metadata=copy.deepcopy(self.metadata),
            points=self.points,
            segments=self.segments,
            frontier_points=self.frontier_points,
            segment_layers=self.segment_layers,
            layer_names=list(self.layer_names),
            forbidden_zones=list(self.forbidden_zones),
            stations=list(self.stations),
            paths=list(self.paths),
        )

    def compute_bounds(self, dropped_layers: Collection[str] = ()) -> Bounds2D:
        """Extent of everything in the map; segments on ``dropped_layers`` (CAD layers the build
        config excludes) are ignored so e.g. an excluded title block does not inflate the bounds."""
        segs = self.segments
        if dropped_layers and self.segment_layers is not None and len(segs):
            drop_ids = [i for i, name in enumerate(self.layer_names) if name in dropped_layers]
            segs = segs[~np.isin(self.segment_layers, drop_ids)]
        parts = [
            self.points,
            segs.reshape(-1, 2),
            self.obstacle_boxes.reshape(-1, 2),
            *self.obstacle_polygons,
            *(z.vertices for z in self.forbidden_zones),
            *(p.points for p in self.paths),
            as_points([(s.x, s.y) for s in self.stations]),
        ]
        parts = [p for p in parts if len(p)]
        if not parts:
            return Bounds2D()
        allp = np.concatenate(parts)
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        return Bounds2D(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


# ---------------------------------------------------------------------------
# API-facing configuration (Pydantic)
# ---------------------------------------------------------------------------

class VehicleProxyConfig(BaseModel):
    """Simple rectangular vehicle footprint proxy for clearance/collision checks."""

    enabled: bool = True
    length: float = 1.2  # X axis (meters)
    width: float = 0.8   # Y axis (meters)
    height: float = 0.6  # Z axis (meters)
    pos_x: float = 0.0
    pos_y: float = 0.0
    yaw_deg: float = 0.0


class BuildConfig(BaseModel):
    """Parameters controlling obstacle reconstruction and USD extrusion."""

    reconstruction_mode: Literal["contour_polygon", "aabb"] = "contour_polygon"
    grid_resolution: float = Field(default=0.05, ge=0.01, le=1.0)
    wall_height: float = Field(default=2.5, ge=0.1, le=20.0)
    forbidden_height: float = Field(default=1.2, ge=0.1, le=20.0)
    morph_close_cells: int = Field(default=2, ge=0, le=5)
    dilate_cells: int = Field(default=0, ge=0, le=5)
    min_cluster_cells: int = Field(default=3, ge=1, le=100)
    fill_solid_obstacles: bool = True
    solid_fill_max_area_m2: float = Field(default=30.0, ge=0.1, le=500.0)
    contour_simplify_tol: float = Field(default=0.04, ge=0.005, le=0.5)
    include_forbidden_as_collision: bool = True
    include_ground: bool = True
    ground_margin: float = Field(default=2.0, ge=0.0, le=50.0)
    include_navigation: bool = True  # stations + route paths as non-colliding USD markers
    # Occupancy grids: turn the unknown/edge frontier of the free space into walls
    seal_unknown: bool = False
    # CAD layers: names to drop entirely, and per-layer extrusion heights (others use wall_height)
    excluded_layers: list[str] = Field(default_factory=list)
    layer_heights: dict[str, float] = Field(default_factory=dict)
    # USD output frame: new origin (map coords) and heading; p_usd = R(-yaw) * (p_map - origin)
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_yaw_deg: float = Field(default=0.0, ge=-360.0, le=360.0)
    vehicle_proxy: VehicleProxyConfig = Field(default_factory=VehicleProxyConfig)

    def dropped_layers(self) -> set[str]:
        """CAD layers left out of the build: explicitly excluded or given a height <= 0."""
        return set(self.excluded_layers) | {n for n, h in self.layer_heights.items() if h <= 0}


class IsaacSimConfig(BaseModel):
    """Connection settings for live Isaac Sim python_server push."""

    host: str = "127.0.0.1"
    port: int = 8226
    use_ssh_tunnel: bool = True
    ssh_user: str = ""  # Optional; leave blank to use ~/.ssh/config
    target_prim_path: str = "/World/AGV_Map"
    push_mode: Literal["reference", "open_stage"] = "reference"
    remote_save_dir: str = "~/allmaptousd_maps"
    add_dome_light: bool = True
    timeout_sec: float = Field(default=30.0, ge=2.0, le=180.0)
    # How the USD file reaches the Isaac host: "inline" (base64 inside the python cell),
    # "file" (local copy for 127.0.0.1 / scp for remote hosts), or "auto" (file above the limit)
    transfer_mode: Literal["auto", "inline", "file"] = "auto"
    inline_limit_mb: float = Field(default=8.0, ge=0.1, le=512.0)
