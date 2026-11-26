"""Rasterize deliberate map edits on a fixed world-coordinate grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field, model_validator

from allmaptousd.models import Bounds2D

MAX_EDIT_CELLS = 4096 * 4096
DRAW, ERASE = 1, 2
Coordinate = Annotated[float, Field(ge=-1e7, le=1e7, allow_inf_nan=False)]


class MapStroke(BaseModel):
    tool: Literal["brush", "line", "rectangle", "eraser"]
    points: list[tuple[Coordinate, Coordinate]] = Field(min_length=1, max_length=8192)
    width: float = Field(ge=0.01, le=5.0)
    filled: bool = False

    @model_validator(mode="after")
    def check_shape(self) -> MapStroke:
        if self.tool in ("line", "rectangle") and len(self.points) != 2:
            raise ValueError("A line or rectangle needs two points.")
        return self


@dataclass
class EditRaster:
    cells: np.ndarray
    origin: np.ndarray
    resolution: float

    def erased(self, x: np.ndarray, y: np.ndarray, radius: float = 0) -> np.ndarray:
        """Sample erase cells at points or across coarser reconstruction cells."""
        cx = (x - self.origin[0]) / self.resolution
        cy = (y - self.origin[1]) / self.resolution
        height, width = self.cells.shape
        if radius:
            half = radius / self.resolution
            x0 = np.floor(cx - half + 1e-7).astype(np.int64).clip(0, width)
            x1 = np.ceil(cx + half - 1e-7).astype(np.int64).clip(0, width)
            y0 = np.floor(cy - half + 1e-7).astype(np.int64).clip(0, height)
            y1 = np.ceil(cy + half - 1e-7).astype(np.int64).clip(0, height)
            summed = cv2.integral((self.cells == ERASE).astype(np.uint8))
            return (summed[y1, x1] - summed[y0, x1] - summed[y1, x0] + summed[y0, x0]) > 0
        cols = np.floor(cx + 1e-7).astype(np.int64)
        rows = np.floor(cy + 1e-7).astype(np.int64)
        inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        return inside & (self.cells[np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)] == ERASE)


def rasterize_edits(strokes: list[MapStroke], res: float, bounds: Bounds2D | None) -> EditRaster:
    """Apply each stroke in order. Coordinates on the outer edge snap to the last cell."""
    cell_paths = []
    if bounds:
        limit = np.ceil(np.array([bounds.max_x, bounds.max_y]) / res - 1e-7).astype(np.int64) - 1
    for stroke in strokes:
        cells = np.floor(np.asarray(stroke.points) / res + 1e-7).astype(np.int64)
        if bounds:
            cells = np.clip(cells, 0, limit)
        filled = stroke.tool == "rectangle" and stroke.filled
        thickness = 1 if filled else max(1, math.floor(stroke.width / res + 0.5))
        lo = cells.min(axis=0) - (thickness - 1) // 2
        hi = cells.max(axis=0) + thickness // 2
        if bounds:
            lo, hi = np.maximum(lo, 0), np.minimum(hi, limit)
        cell_paths.append((cells, thickness, lo, hi, filled))

    start = np.min([item[2] for item in cell_paths], axis=0)
    end = np.max([item[3] for item in cell_paths], axis=0)
    width, height = (end - start + 1).tolist()
    if width * height > MAX_EDIT_CELLS:
        raise ValueError("Edits span too many grid cells. Use a coarser resolution or a smaller drawing area.")

    raster = np.zeros((height, width), dtype=np.uint8)
    for stroke, (cells, thickness, lo, hi, filled) in zip(strokes, cell_paths):
        path = (cells - lo).astype(np.int32)
        w, h = (hi - lo + 1).tolist()
        mask = np.zeros((h, w), dtype=np.uint8)
        if stroke.tool == "rectangle":
            cv2.rectangle(mask, tuple(path.min(axis=0)), tuple(path.max(axis=0)), 1, -1 if filled else 1)
        elif len(path) == 1:
            x, y = path[0]
            mask[y, x] = 1
        else:
            cv2.polylines(mask, [path], False, 1, 1)
        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            mask = cv2.dilate(mask, kernel)
        col, row = lo - start
        region = raster[row : row + h, col : col + w]
        region[mask != 0] = ERASE if stroke.tool == "eraser" else DRAW
    return EditRaster(raster, start * res, res)
