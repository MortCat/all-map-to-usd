"""Command-line batch converter: ``allmaptousd convert <map> -o <out.usd>``.

Examples::

    allmaptousd convert maps/factory.yaml -o out/factory.usd
    allmaptousd convert plant.dxf -o plant.usda --dxf-units mm --layer-height RACK=2.0 --exclude-layer TEXT
    allmaptousd convert site.smap -o site.usd --config build.json --origin 12.5 -3 90
    allmaptousd formats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from allmaptousd.models import BuildConfig
from allmaptousd.parsers import MapInputBundle, create_default_registry
from allmaptousd.pipeline import MapSession


def _layer_height(text: str) -> tuple[str, float]:
    name, sep, value = text.rpartition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected LAYER=HEIGHT, got {text!r}")
    return name, float(value)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="allmaptousd", description="AGV/AMR map -> Isaac Sim USD converter")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("formats", help="List supported map formats")

    cv = sub.add_parser("convert", help="Convert a map file to .usd / .usda")
    cv.add_argument("input", type=Path, help="Map file (.smap/.json, .yaml/.zip, .dxf, .png/.pgm/...)")
    cv.add_argument("-o", "--output", type=Path, help="Output .usd/.usda (default: <input>.usd)")
    cv.add_argument("--format", default="auto", help="Parser format_id (default: auto detect)")
    cv.add_argument("--config", type=Path, help="BuildConfig JSON; CLI flags below override it")
    cv.add_argument("--mode", choices=["contour_polygon", "aabb"], help="Reconstruction mode")
    cv.add_argument("--resolution", type=float, help="Grid resolution (m)")
    cv.add_argument("--wall-height", type=float, help="Default extrusion height (m)")
    cv.add_argument("--image-resolution", type=float, default=0.05, help="Raster image m/px")
    cv.add_argument("--dxf-units", default="auto", choices=["auto", "m", "cm", "mm", "in", "ft"])
    cv.add_argument("--seal-unknown", action="store_true", help="ROS: seal unknown frontier as walls")
    cv.add_argument("--layer-height", type=_layer_height, action="append", default=[],
                    metavar="LAYER=H", help="Per-CAD-layer height (repeatable)")
    cv.add_argument("--exclude-layer", action="append", default=[], metavar="LAYER",
                    help="Drop a CAD layer (repeatable)")
    cv.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                    help="USD output origin and heading in map coordinates")
    cv.add_argument("--no-navigation", action="store_true", help="Skip stations / route paths")
    cv.add_argument("--no-vehicle", action="store_true", help="Skip the AGV proxy model")
    return ap


def _build_config(args: argparse.Namespace, data: dict) -> BuildConfig:
    data = dict(data)
    overrides = {
        "reconstruction_mode": args.mode,
        "grid_resolution": args.resolution,
        "wall_height": args.wall_height,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    if args.layer_height:
        data["layer_heights"] = {**data.get("layer_heights", {}), **dict(args.layer_height)}
    if args.exclude_layer:
        data["excluded_layers"] = [*data.get("excluded_layers", []), *args.exclude_layer]
    if args.origin:
        data["origin_x"], data["origin_y"], data["origin_yaw_deg"] = args.origin
    if args.no_navigation:
        data["include_navigation"] = False
    if args.seal_unknown:
        data["seal_unknown"] = True
    config = BuildConfig.model_validate(data)
    if args.no_vehicle:
        config.vehicle_proxy.enabled = False
    return config


def convert(args: argparse.Namespace) -> Path:
    file_cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    config = _build_config(args, file_cfg)
    bundle = MapInputBundle.from_path(
        args.input,
        options={
            "image_resolution": args.image_resolution,
            "dxf_units": args.dxf_units,
        },
    )
    session = MapSession(config=config)
    # The AGV proxy starts at the map centre unless the config file places it.
    canonical = session.load(bundle, args.format, center_vehicle="vehicle_proxy" not in file_cfg)
    output = session.export_file(args.output or args.input.with_suffix(".usd"))
    extra = canonical.metadata.extra
    print(
        f"{args.input.name} [{canonical.metadata.source_format}] -> {output}  "
        f"({extra.get('reconstructed_polygon_count', 0)} polygons, "
        f"{extra.get('reconstructed_box_count', 0)} boxes, "
        f"{len(canonical.forbidden_zones)} zones, {len(canonical.stations)} stations, "
        f"{len(canonical.paths)} paths)"
    )
    if extra.get("resolution_was_coarsened"):
        print(f"warning: grid coarsened to {extra['effective_grid_resolution']} m (map too large)",
              file=sys.stderr)
    return output


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "formats":
        for p in create_default_registry().list_parsers():
            print(f"{p['format_id']:15s} {', '.join(p['supported_extensions']):30s} {p['display_name']}")
        return 0
    try:
        convert(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
