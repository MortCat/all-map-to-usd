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

from allmaptousd.exporters.usd_exporter import export_to_usd, export_to_usd_bytes
from allmaptousd.geometry import reconstruct_obstacles
from allmaptousd.models import BuildConfig, CanonicalMap
from allmaptousd.parsers import MapInputBundle, ParserRegistry, create_default_registry


class NoMapLoaded(RuntimeError):
    """Raised when rebuilding or exporting before a map was loaded."""


@dataclass
class MapSession:
    """One loaded map: the parsed input (``raw``), the current build config, and its result."""

    registry: ParserRegistry = field(default_factory=create_default_registry)
    config: BuildConfig = field(default_factory=BuildConfig)
    raw: CanonicalMap | None = None
    built: CanonicalMap | None = None

    def load(self, bundle: MapInputBundle, format_hint: str = "auto", center_vehicle: bool = True) -> CanonicalMap:
        return self.use(self.registry.parse(bundle, format_hint=format_hint), center_vehicle)

    def use(self, raw: CanonicalMap, center_vehicle: bool = True) -> CanonicalMap:
        """Adopt an already-parsed map; by default the AGV proxy starts at the map centre."""
        self.raw = raw
        if center_vehicle:
            cx, cy = raw.compute_bounds(self.config.dropped_layers()).center
            self.config.vehicle_proxy.pos_x, self.config.vehicle_proxy.pos_y = round(cx, 3), round(cy, 3)
        return self.rebuild()

    def rebuild(self, config: BuildConfig | None = None) -> CanonicalMap:
        if self.raw is None:
            raise NoMapLoaded("No map loaded yet.")
        if config is not None:
            self.config = config
        self.built = reconstruct_obstacles(self.raw.copy(), self.config)
        return self.built

    def _require_built(self) -> CanonicalMap:
        if self.built is None:
            raise NoMapLoaded("No map loaded yet.")
        return self.built

    def export_bytes(self, extension: str = ".usd") -> bytes:
        return export_to_usd_bytes(self._require_built(), self.config, extension=extension)

    def export_file(self, path: str | Path) -> Path:
        return export_to_usd(self._require_built(), self.config, path)
