"""Default parser registry wiring."""

from allmaptousd.parsers.base import MapInputBundle, MapParser, ParserRegistry
from allmaptousd.parsers.dxf_cad import DxfMapParser
from allmaptousd.parsers.raster_image import RasterImageParser
from allmaptousd.parsers.ros_occupancy import RosOccupancyParser
from allmaptousd.parsers.seer_smap import SeerSmapParser


def create_default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(SeerSmapParser())
    registry.register(RosOccupancyParser())
    registry.register(DxfMapParser())
    registry.register(RasterImageParser())
    return registry


__all__ = [
    "MapInputBundle",
    "MapParser",
    "ParserRegistry",
    "SeerSmapParser",
    "RosOccupancyParser",
    "DxfMapParser",
    "RasterImageParser",
    "create_default_registry",
]
