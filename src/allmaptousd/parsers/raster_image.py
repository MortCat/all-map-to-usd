"""Standalone raster floorplan / SICK SMET image parser (.png, .pgm, .bmp, .jpg)."""

from __future__ import annotations

from pathlib import Path

from allmaptousd.models import CanonicalMap, MapMetadata
from allmaptousd.parsers.base import MapInputBundle, grid_to_points, load_gray_image


class RasterImageParser:
    format_id = "raster_image"
    display_name = "Image Floorplan / SICK Export (.png / .pgm / .bmp)"
    supported_extensions = (".png", ".pgm", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

    def can_parse(self, bundle: MapInputBundle) -> bool:
        return Path(bundle.primary_filename).suffix.lower() in self.supported_extensions

    def parse(self, bundle: MapInputBundle) -> CanonicalMap:
        resolution = max(0.005, min(2.0, float(bundle.options.get("image_resolution", 0.05))))
        occupied_thresh = float(bundle.options.get("occupied_thresh", 0.65))

        arr = load_gray_image(bundle.primary_bytes, bundle.primary_filename)
        height_px, width_px = arr.shape
        occupied = (1.0 - arr) >= occupied_thresh  # dark pixels = walls

        # Center map at world origin (0, 0)
        points = grid_to_points(
            occupied, resolution, -0.5 * width_px * resolution, -0.5 * height_px * resolution
        )
        return CanonicalMap(
            metadata=MapMetadata(
                name=Path(bundle.primary_filename).stem,
                source_format=self.format_id,
                resolution=resolution,
                extra={
                    "width_px": width_px,
                    "height_px": height_px,
                    "occupied_thresh": occupied_thresh,
                    "raw_point_count": len(points),
                },
            ),
            points=points,
        )
