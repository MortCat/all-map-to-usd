"""FastAPI backend for AllMapToUSD."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from allmaptousd.exporters import check_isaac_connection, push_usd_to_isaac
from allmaptousd.models import BuildConfig, CanonicalMap, IsaacSimConfig
from allmaptousd.parsers import MapInputBundle
from allmaptousd.pipeline import MapSession, NoMapLoaded

ROOT_DIR = Path(__file__).resolve().parents[3]
SAMPLE_DIR = ROOT_DIR / "sample_maps"
EXPORT_DIR = ROOT_DIR / "exports"
STATIC_DIR = Path(__file__).resolve().parent / "static"  # index.html, app.css, js/*.js
ISAAC_CONFIG_FILE = ROOT_DIR / "isaac_config.json"
SAMPLES = {
    "ros_factory": {
        "name": "ROS.yaml (Noisy SLAM Occupancy Grid)",
        "format": "ros_occupancy",
        "file": "sample_factory.yaml",
    },
}


_BAD_FILENAME_CHARS = frozenset('\\/:*?"<>|')


def safe_filename(name: str, fallback: str = "map_layout") -> str:
    """Replace path separators, Windows-reserved and control characters; keeps non-ASCII names."""
    cleaned = "".join(
        "_" if ch in _BAD_FILENAME_CHARS or ord(ch) < 32 else ch for ch in name
    ).strip(" .")
    return cleaned or fallback


def _round_list(arr: np.ndarray) -> list:
    return np.round(arr, 4).tolist()


def build_viewer_payload(canonical: CanonicalMap, config: BuildConfig) -> dict:
    """Serialize CanonicalMap into compact JSON for the 2D CAD canvas viewer."""
    raw_pts = canonical.points
    stride = max(1, len(raw_pts) // 25000)
    bounds = canonical.compute_bounds(config.dropped_layers())

    return {
        "metadata": asdict(canonical.metadata),
        "bounds": asdict(bounds),
        "config": config.model_dump(),
        "raw_points": _round_list(raw_pts[::stride]),
        "raw_point_total": len(raw_pts),
        "obstacle_boxes": _round_list(canonical.obstacle_boxes),
        "obstacle_polygons": [_round_list(poly) for poly in canonical.obstacle_polygons],
        "obstacle_polygon_heights": canonical.obstacle_polygon_heights,
        "obstacle_box_heights": canonical.obstacle_box_heights,
        "forbidden_zones": [
            {"name": z.name, "zone_type": z.zone_type, "vertices": _round_list(z.vertices)}
            for z in canonical.forbidden_zones
        ],
        "stations": [
            {"name": s.name, "kind": s.kind, "x": round(s.x, 4), "y": round(s.y, 4), "yaw": round(s.yaw, 4)}
            for s in canonical.stations
        ],
        "paths": [
            {"name": p.name, "kind": p.kind, "points": _round_list(p.points)} for p in canonical.paths
        ],
        "layers": canonical.layer_names,
    }


class AppState:
    """Process-wide UI state: the loaded map session and the persisted Isaac Sim settings."""

    def __init__(self) -> None:
        self.session = MapSession()
        self.isaac_config = self._load_isaac_config()

    @staticmethod
    def _load_isaac_config() -> IsaacSimConfig:
        try:
            return IsaacSimConfig.model_validate_json(ISAAC_CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return IsaacSimConfig()

    def save_isaac_config(self, cfg: IsaacSimConfig) -> None:
        self.isaac_config = cfg
        ISAAC_CONFIG_FILE.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    def payload(self) -> dict:
        return build_viewer_payload(self.session.built, self.session.config)


state = AppState()


app = FastAPI(title="AllMapToUSD", version="1.1.0")


@app.exception_handler(NoMapLoaded)
def _no_map_loaded(_request: Request, exc: NoMapLoaded) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/parsers")
def list_parsers() -> dict:
    return {"parsers": state.session.registry.list_parsers()}


@app.get("/api/samples")
def list_samples() -> dict:
    return {"samples": [{"id": sid, **info} for sid, info in SAMPLES.items()]}


@app.post("/api/load-sample/{sample_id}")
def load_sample(sample_id: str) -> dict:
    if sample_id not in SAMPLES:
        raise HTTPException(status_code=404, detail=f"Unknown sample_id: {sample_id}")
    state.session.load(MapInputBundle.from_path(SAMPLE_DIR / SAMPLES[sample_id]["file"]))
    return state.payload()


@app.post("/api/upload")
async def upload_map(
    files: list[UploadFile] = File(...),
    format_hint: str = Form("auto"),
    image_resolution: float = Form(0.05),
    dxf_units: str = Form("auto"),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    file_map: dict[str, bytes] = {}
    for uf in files:
        fname = Path(uf.filename or "uploaded_map").name
        file_map[fname] = await uf.read()

    priority_exts = (".smap", ".yaml", ".yml", ".dxf", ".zip", ".json")
    primary_name = next(
        (name for name in file_map if Path(name).suffix.lower() in priority_exts),
        next(iter(file_map.keys())),
    )
    companions = {k: v for k, v in file_map.items() if k != primary_name}
    bundle = MapInputBundle(
        primary_filename=primary_name,
        primary_bytes=file_map[primary_name],
        companion_files=companions,
        options={
            "image_resolution": float(image_resolution),
            "dxf_units": dxf_units,
        },
    )

    try:
        state.session.load(bundle, format_hint=format_hint)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return state.payload()


@app.post("/api/rebuild")
def rebuild_map(config: BuildConfig) -> dict:
    state.session.rebuild(config)
    return state.payload()


@app.post("/api/export/usd")
def export_usd(config: BuildConfig, ext: str = "usd") -> Response:
    built = state.session.rebuild(config)
    clean_ext = "usda" if ext.lower() == "usda" else "usd"
    data = state.session.export_bytes(f".{clean_ext}")
    filename = f"{safe_filename(built.metadata.name.replace(' ', '_'))}.{clean_ext}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    saved_path = EXPORT_DIR / filename
    saved_path.write_bytes(data)
    # HTTP headers are latin-1 only: send an ASCII fallback plus an RFC 5987 UTF-8 filename.
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": (
                f'attachment; filename="map_layout.{clean_ext}"; '
                f"filename*=UTF-8''{quote(filename)}"
            ),
            "X-Saved-Path": quote(str(saved_path), safe=":/\\"),
        },
    )


@app.get("/api/isaac/config")
def get_isaac_config() -> IsaacSimConfig:
    return state.isaac_config


@app.post("/api/isaac/config")
def update_isaac_config(cfg: IsaacSimConfig) -> IsaacSimConfig:
    state.save_isaac_config(cfg)
    return state.isaac_config


class IsaacPushRequest(BaseModel):
    build_config: BuildConfig
    isaac_config: IsaacSimConfig


@app.post("/api/isaac/test")
def test_isaac(cfg: IsaacSimConfig) -> dict:
    state.save_isaac_config(cfg)
    return check_isaac_connection(cfg)


@app.post("/api/isaac/push")
def push_to_isaac(req: IsaacPushRequest) -> dict:
    built = state.session.rebuild(req.build_config)
    state.save_isaac_config(req.isaac_config)
    return push_usd_to_isaac(state.session.export_bytes(".usd"), built.metadata.name, req.isaac_config)
