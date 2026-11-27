"""Application core: parse -> rebuild -> export, shared by the web UI and the CLI.

The package is laid out as a small ports-and-adapters design::

    driving adapters   web.server (FastAPI UI), cli
    core               models (CanonicalMap, BuildConfig), geometry.reconstruct, MapSession
    driven adapters    parsers.*  (one per vendor format, behind the MapParser port)
                       exporters.usd_exporter, exporters.isaac_connector

Adding a vendor format means adding a parser; adding a front end means calling MapSession.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from allmaptousd.exporters.usd_exporter import export_to_usd, export_to_usd_bytes, placed_model_files
from allmaptousd.geometry import reconstruct_obstacles
from allmaptousd.geometry.editing import EditRaster, MapStroke, rasterize_edits
from allmaptousd.model_assets import ModelAsset
from allmaptousd.models import BuildConfig, CanonicalMap
from allmaptousd.parsers import MapInputBundle, ParserRegistry, create_default_registry


class NoMapLoaded(RuntimeError):
    """Raised when rebuilding or exporting before a map was loaded."""


@dataclass
class MapSession:
    """One loaded map: the parsed input (``raw``), the current build config, and its result.

    ``models`` is the library of imported 3D models; it stays when another map is loaded.
    """

    registry: ParserRegistry = field(default_factory=create_default_registry)
    config: BuildConfig = field(default_factory=BuildConfig)
    raw: CanonicalMap | None = None
    built: CanonicalMap | None = None
    edit_resolution: float = 0.05
    edit_revision: str = field(default_factory=lambda: uuid4().hex)
    models: dict[str, ModelAsset] = field(default_factory=dict)
    _strokes: list[MapStroke] = field(default_factory=list, init=False)
    _edit_cursor: int = field(default=0, init=False)
    _edit_raster: EditRaster | None = field(default=None, init=False)

    def load(self, bundle: MapInputBundle, format_hint: str = "auto", center_vehicle: bool = True) -> CanonicalMap:
        raw = self.registry.parse(bundle, format_hint=format_hint)
        saved_models = raw.metadata.extra.pop("project_models", [])
        saved = raw.metadata.extra.pop("project_build", None)
        for item in saved_models:
            if item["id"] not in self.models:
                self.models[item["id"]] = ModelAsset.from_usdc(
                    item["id"], item["name"], item["data"], item.get("files"))
        if saved is None:
            return self.use(raw, center_vehicle)
        # A map saved by this app: its obstacles are already final, so the steps that grow them
        # (closing gaps, dilation, sealing) are switched off; the other settings and the AGV pose return.
        config = BuildConfig.model_validate({
            **saved, "grid_resolution": raw.metadata.resolution,
            "morph_close_cells": 0, "dilate_cells": 0, "seal_unknown": False,
        })
        return self.use(raw, center_vehicle=False, config=config, keep_equipment=True)

    def use(
        self, raw: CanonicalMap, center_vehicle: bool = True, *, config: BuildConfig | None = None,
        keep_equipment: bool = False,
    ) -> CanonicalMap:
        """Adopt an already-parsed map and clear placements unless the map has saved equipment."""
        config = (config or self.config).model_copy(deep=True)
        if not keep_equipment:
            config.equipment = []
        if center_vehicle:
            cx, cy = raw.compute_bounds(config.dropped_layers()).center
            config.vehicle_proxy.pos_x, config.vehicle_proxy.pos_y = round(cx, 3), round(cy, 3)
        built = reconstruct_obstacles(raw.copy(), config)
        self.raw, self.built, self.config = raw, built, config
        self.edit_resolution = config.grid_resolution
        self._strokes = []
        self._edit_cursor = 0
        self._edit_raster = None
        self.edit_revision = uuid4().hex
        return built

    @property
    def can_undo(self) -> bool:
        return self._edit_cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._edit_cursor < len(self._strokes)

    def apply_edits(self, strokes: list[MapStroke], config: BuildConfig) -> CanonicalMap:
        history = self._strokes[:self._edit_cursor] + strokes
        return self._build_edit_history(history, len(history), config)

    def step_edit_history(self, *, redo: bool, config: BuildConfig) -> CanonicalMap:
        if not (self.can_redo if redo else self.can_undo):
            raise ValueError("No stroke to redo." if redo else "No stroke to undo.")
        return self._build_edit_history(self._strokes, self._edit_cursor + (1 if redo else -1), config)

    def _build_edit_history(self, history: list[MapStroke], cursor: int, config: BuildConfig) -> CanonicalMap:
        if self.raw is None:
            raise NoMapLoaded("No map loaded yet.")
        raster = rasterize_edits(history[:cursor], self.edit_resolution, self.raw.drawing_bounds) if cursor else None
        built = reconstruct_obstacles(self.raw.copy(), config, raster)
        self._strokes, self._edit_cursor, self._edit_raster = history, cursor, raster
        self.built, self.config = built, config
        self.edit_revision = uuid4().hex
        return built

    def rebuild(self, config: BuildConfig | None = None) -> CanonicalMap:
        if self.raw is None:
            raise NoMapLoaded("No map loaded yet.")
        config = config or self.config
        self.built = reconstruct_obstacles(self.raw.copy(), config, self._edit_raster)
        self.config = config
        return self.built

    def _require_built(self) -> CanonicalMap:
        if self.built is None:
            raise NoMapLoaded("No map loaded yet.")
        return self.built

    def export_bytes(self, extension: str = ".usd") -> bytes:
        return export_to_usd_bytes(self._require_built(), self.config, extension=extension, models=self.models)

    def export_package(self, extension: str = ".usd") -> tuple[bytes, dict[str, bytes]]:
        """The USD and the files its 3D models use, by their path relative to the USD."""
        return self.export_bytes(extension), placed_model_files(self.config, self.models)

    def export_file(self, path: str | Path) -> Path:
        return export_to_usd(self._require_built(), self.config, path, self.models)
