"""Generate realistic ROS (.yaml + .pgm) and CAD / SICK SMET (.dxf) sample maps."""

from __future__ import annotations

import math
from pathlib import Path

import ezdxf
import numpy as np
from PIL import Image


def generate_ros_sample(out_dir: Path) -> tuple[Path, Path]:
    """Create a realistic noisy 23m x 16m LiDAR SLAM ROS Nav2 map (.yaml + .pgm).

    A 20m x 14m hall surrounded by unknown (never-scanned) space, with an open west
    loading bay whose scanned apron ends in unknown space (try *Seal Unknown Boundary*),
    two 15-deg tilted pallet racks, one -22-deg tilted workstation, an L-shaped CNC
    machine, two round concrete columns, +/-3 cm laser jitter, scan dropouts, and
    scattered outlier dust.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    res = 0.05
    min_x, max_x = -12.0, 11.0
    min_y, max_y = -8.0, 8.0
    w_px = int(round((max_x - min_x) / res))
    h_px = int(round((max_y - min_y) / res))

    # 205 = unknown grey, 254 = free white, 0 = occupied black
    img = np.full((h_px, w_px), 205, dtype=np.uint8)

    def mark_free(x0: float, y0: float, x1: float, y1: float) -> None:
        c0, c1 = int(round((x0 - min_x) / res)), int(round((x1 - min_x) / res))
        r0, r1 = h_px - 1 - int(round((y1 - min_y) / res)), h_px - 1 - int(round((y0 - min_y) / res))
        img[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = 254

    # Scanned free space: the hall interior plus a short apron outside the loading bay
    mark_free(-10.0, -7.0, 10.0, 7.0)
    mark_free(-11.5, -2.5, -10.0, 2.5)

    def mark_point(x: float, y: float) -> None:
        c = int(round((x - min_x) / res))
        r = h_px - 1 - int(round((y - min_y) / res))
        if 0 <= r < h_px and 0 <= c < w_px:
            img[r, c] = 0

    def add_noisy_seg(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        step: float = 0.04,
        sigma: float = 0.026,
        dropout: float = 0.07,
    ) -> None:
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(dist / step)) + 1)
        for i in range(n):
            if rng.random() < dropout:
                continue
            t = i / (n - 1)
            x = x0 + (x1 - x0) * t + float(rng.normal(0.0, sigma))
            y = y0 + (y1 - y0) * t + float(rng.normal(0.0, sigma))
            mark_point(x, y)

    def add_rotated_rect(
        cx: float, cy: float, length: float, width: float, yaw_deg: float, sigma: float = 0.024
    ) -> None:
        rad = math.radians(yaw_deg)
        c, s = math.cos(rad), math.sin(rad)
        hl, hw = 0.5 * length, 0.5 * width
        corners = [
            (cx + lx * c - ly * s, cy + lx * s + ly * c)
            for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
        ]
        for i in range(4):
            p0, p1 = corners[i], corners[(i + 1) % 4]
            add_noisy_seg(p0[0], p0[1], p1[0], p1[1], sigma=sigma)

    def add_circle(cx: float, cy: float, radius: float, n_pts: int = 80, sigma: float = 0.018) -> None:
        for i in range(n_pts):
            ang = 2.0 * math.pi * i / n_pts
            r = radius + float(rng.normal(0.0, sigma))
            mark_point(cx + r * math.cos(ang), cy + r * math.sin(ang))

    # Outer walls with west loading bay opening
    add_noisy_seg(-10.0, -7.0, 10.0, -7.0, sigma=0.028)
    add_noisy_seg(10.0, -7.0, 10.0, 7.0, sigma=0.028)
    add_noisy_seg(10.0, 7.0, -10.0, 7.0, sigma=0.028)
    add_noisy_seg(-10.0, 7.0, -10.0, 1.8, sigma=0.028)
    add_noisy_seg(-10.0, -1.8, -10.0, -7.0, sigma=0.028)

    # Two 15-degree angled pallet racks
    add_rotated_rect(-4.2, 3.2, 4.4, 1.6, yaw_deg=15.0)
    add_rotated_rect(3.5, 3.4, 4.8, 1.5, yaw_deg=15.0)

    # One -22-degree angled packing station
    add_rotated_rect(4.2, -2.6, 3.8, 1.8, yaw_deg=-22.0)

    # One L-shaped CNC machine
    l_poly = [
        (-6.2, -4.8),
        (-2.2, -4.8),
        (-2.2, -3.4),
        (-4.6, -3.4),
        (-4.6, -1.2),
        (-6.2, -1.2),
    ]
    for i in range(len(l_poly)):
        p0, p1 = l_poly[i], l_poly[(i + 1) % len(l_poly)]
        add_noisy_seg(p0[0], p0[1], p1[0], p1[1], sigma=0.024)

    # Two circular concrete columns
    add_circle(-0.2, 0.8, 0.45)
    add_circle(0.4, -4.4, 0.45)

    # Scattered outlier laser dust
    for _ in range(55):
        mark_point(float(rng.uniform(-9.2, 9.2)), float(rng.uniform(-6.2, 6.2)))

    pgm_path = out_dir / "sample_factory.pgm"
    yaml_path = out_dir / "sample_factory.yaml"
    Image.fromarray(img, mode="L").save(pgm_path)

    yaml_content = (
        "image: sample_factory.pgm\n"
        f"resolution: {res}\n"
        f"origin: [{min_x}, {min_y}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path, pgm_path


def generate_dxf_sample(out_path: Path) -> Path:
    """Create a clean CAD / SICK SMET vector floorplan (.dxf, meters)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6  # Meters
    msp = doc.modelspace()

    # Outer walls with loading bay opening
    msp.add_line((-10.0, -7.0), (10.0, -7.0))
    msp.add_line((10.0, -7.0), (10.0, 7.0))
    msp.add_line((10.0, 7.0), (-10.0, 7.0))
    msp.add_line((-10.0, 7.0), (-10.0, 1.8))
    msp.add_line((-10.0, -1.8), (-10.0, -7.0))

    # Tilted racks as closed LWPOLYLINEs
    def add_rot_poly(cx: float, cy: float, length: float, width: float, yaw_deg: float) -> None:
        rad = math.radians(yaw_deg)
        c, s = math.cos(rad), math.sin(rad)
        hl, hw = 0.5 * length, 0.5 * width
        pts = [
            (cx + lx * c - ly * s, cy + lx * s + ly * c)
            for lx, ly in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))
        ]
        msp.add_lwpolyline(pts, close=True)

    add_rot_poly(-4.2, 3.2, 4.4, 1.6, 15.0)
    add_rot_poly(3.5, 3.4, 4.8, 1.5, 15.0)
    add_rot_poly(4.2, -2.6, 3.8, 1.8, -22.0)

    # L-shaped CNC machine
    msp.add_lwpolyline(
        [
            (-6.2, -4.8),
            (-2.2, -4.8),
            (-2.2, -3.4),
            (-4.6, -3.4),
            (-4.6, -1.2),
            (-6.2, -1.2),
        ],
        close=True,
    )

    # Round concrete pillars
    msp.add_circle((-0.2, 0.8), radius=0.45)
    msp.add_circle((0.4, -4.4), radius=0.45)

    doc.saveas(out_path)
    return out_path


if __name__ == "__main__":
    # The bundled sample ships inside the package so it also works from a regular (non-editable) install.
    base = Path(__file__).resolve().parents[1] / "src" / "allmaptousd" / "samples"
    generate_ros_sample(base)
    print(f"Generated ROS sample map in {base}")
