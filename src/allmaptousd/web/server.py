"""FastAPI backend for AllMapToUSD."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Annotated, Literal
from urllib.parse import quote
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from allmaptousd import __version__
from allmaptousd.exporters import check_isaac_connection, push_usd_to_isaac
from allmaptousd.exporters.isaac_connector import pull_stage_slice, push_equipment_to_isaac
from allmaptousd.exporters.ros_map import export_ros_map_zip
from allmaptousd.exporters.usd_exporter import equipment_floor_samples, export_equipment_usda, placed_model_files
from allmaptousd.geometry.editing import MapStroke
from allmaptousd.geometry.isaac_map import map_from_stage_slice
from allmaptousd.model_assets import ModelSource, import_model, open_model_files, open_model_source, write_tree
from allmaptousd.models import (
    Bounds2D,
    BuildConfig,
    CanonicalMap,
    EquipmentPlacement,
    IsaacSimConfig,
    MapMetadata,
)
from allmaptousd.parsers import MapInputBundle
from allmaptousd.pipeline import MapSession, NoMapLoaded

PACKAGE_DIR = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PACKAGE_DIR / "samples"
STATIC_DIR = PACKAGE_DIR / "web" / "static"  # index.html, app.css, js/*.js


def _data_dir() -> Path:
    """Where exports and settings go: $ALLMAPTOUSD_HOME, else the project root of a source checkout
    (start_app.cmd keeps exports/ there), else ~/AllMapToUSD for a regular pip install."""
    if home := os.environ.get("ALLMAPTOUSD_HOME"):
        return Path(home).expanduser()
    checkout = PACKAGE_DIR.parents[1]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src" / "allmaptousd").is_dir():
        return checkout
    return Path.home() / "AllMapToUSD"


DATA_DIR = _data_dir()
EXPORT_DIR = DATA_DIR / "exports"
ISAAC_CONFIG_FILE = DATA_DIR / "isaac_config.json"
MapRevision = Annotated[str | None, Header()]
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
    stride = max(1, (len(raw_pts) + 24999) // 25000)
    bounds = canonical.compute_bounds(config.dropped_layers())

    return {
        "metadata": asdict(canonical.metadata),
        "bounds": asdict(bounds),
        "drawing_bounds": asdict(canonical.drawing_bounds) if canonical.drawing_bounds is not None else None,
        "config": config.model_dump(),
        "raw_points": _round_list(raw_pts[::stride]),
        "drawn_points": _round_list(canonical.drawn_points),
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
        self.map_lock = RLock()
        self.isaac_config = self._load_isaac_config()
        # Uploaded 3D model files waiting for the user to pick parts (newest last, a few kept)
        self.model_sources: dict[str, ModelSource] = {}

    def add_model_source(self, source: ModelSource) -> str:
        token = uuid4().hex
        with self.map_lock:
            self.model_sources[token] = source
            while len(self.model_sources) > 3:
                self.model_sources.pop(next(iter(self.model_sources))).close()
        return token

    def model_source(self, token: str) -> ModelSource:
        source = self.model_sources.get(token)
        if source is None:
            raise HTTPException(status_code=404, detail="The uploaded model has expired. Choose the file again.")
        return source

    @staticmethod
    def _load_isaac_config() -> IsaacSimConfig:
        try:
            return IsaacSimConfig.model_validate_json(ISAAC_CONFIG_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return IsaacSimConfig()

    def save_isaac_config(self, cfg: IsaacSimConfig) -> None:
        self.isaac_config = cfg
        ISAAC_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        ISAAC_CONFIG_FILE.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")

    def payload(self) -> dict:
        return {
            **build_viewer_payload(self.session.built, self.session.config),
            "edit_revision": self.session.edit_revision,
            "edit_resolution": self.session.edit_resolution,
            "can_undo": self.session.can_undo,
            "can_redo": self.session.can_redo,
            "models": [m.summary() for m in self.session.models.values()],
        }

    def check_map_revision(self, revision: str | None) -> None:
        if revision is not None and revision != self.session.edit_revision:
            raise HTTPException(status_code=409, detail="The current map has changed. Reload the map to continue.")

    def build_usd(
        self, config: BuildConfig, extension: str, revision: str | None,
    ) -> tuple[str, bytes, dict[str, bytes]]:
        """(map name, USD, files the placed models use by their path relative to the USD)."""
        with self.map_lock:
            self.check_map_revision(revision)
            built = self.session.rebuild(config)
            try:
                return built.metadata.name, *self.session.export_package(extension)
            except ValueError as exc:  # e.g. a placed model that is no longer loaded
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    def build_ros_map(self, config: BuildConfig, revision: str | None) -> tuple[str, bytes]:
        with self.map_lock:
            self.check_map_revision(revision)
            built = self.session.rebuild(config)
            stem = safe_filename(built.metadata.name.replace(" ", "_"))
            return stem, export_ros_map_zip(built, self.session.config, stem, self.session.models)


state = AppState()


app = FastAPI(title="AllMapToUSD", version=__version__)

# Host names the UI may be reached under. A web page on another site that rebinds its DNS name to
# 127.0.0.1 still sends its own name here, so it cannot drive the API (which can run code in Isaac
# Sim). run_app.py adds the --host it binds to; "*" turns the check off.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_REQUEST_BYTES = 1 << 30  # 1 GiB: far above real map uploads, stops a runaway request filling the disk


@app.middleware("http")
async def _guard_requests(request: Request, call_next):
    if "*" not in ALLOWED_HOSTS and request.url.hostname not in ALLOWED_HOSTS:
        return JSONResponse(status_code=400, content={"detail": "Invalid host header."})
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_REQUEST_BYTES:
        return JSONResponse(status_code=413, content={"detail": "Upload is larger than 1 GiB."})
    return await call_next(request)


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
    with state.map_lock:
        state.session.load(MapInputBundle.from_path(SAMPLE_DIR / SAMPLES[sample_id]["file"]))
        return state.payload()


class EmptyMapRequest(BaseModel):
    width: float = Field(ge=0.1, le=1000)
    height: float = Field(ge=0.1, le=1000)
    resolution: float = Field(ge=0.01, le=1.0)
    config: BuildConfig


class EditHistoryRequest(BaseModel):
    revision: str
    config: BuildConfig


class MapEditRequest(EditHistoryRequest):
    strokes: list[MapStroke] = Field(min_length=1, max_length=500)


@app.post("/api/empty-map")
def create_empty_map(req: EmptyMapRequest) -> dict:
    if min(req.width, req.height) < req.resolution:
        raise HTTPException(status_code=422, detail="Map dimensions must each be at least one grid cell.")
    raw = CanonicalMap(
        metadata=MapMetadata(name="Empty Map", source_format="empty", resolution=req.resolution),
        drawing_bounds=Bounds2D(0, 0, req.width, req.height),
    )
    with state.map_lock:
        state.session.use(raw, config=req.config.model_copy(update={"grid_resolution": req.resolution}))
        return state.payload()


@app.post("/api/map-edits")
def apply_map_edits(req: MapEditRequest) -> dict:
    with state.map_lock:
        state.check_map_revision(req.revision)
        try:
            state.session.apply_edits(req.strokes, req.config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return state.payload()


@app.post("/api/map-edits/{direction}")
def step_map_history(direction: Literal["undo", "redo"], req: EditHistoryRequest) -> dict:
    with state.map_lock:
        state.check_map_revision(req.revision)
        try:
            state.session.step_edit_history(redo=direction == "redo", config=req.config)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
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
        with state.map_lock:
            state.session.load(bundle, format_hint=format_hint)
            return state.payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/rebuild")
def rebuild_map(config: BuildConfig, x_map_revision: MapRevision = None) -> dict:
    with state.map_lock:
        state.check_map_revision(x_map_revision)
        state.session.rebuild(config)
        return state.payload()


class ExportTarget(BaseModel):
    """Where an export is saved: ``name`` (default: the map name) in ``folder`` (default: exports/)."""

    name: str = ""
    folder: str = ""
    overwrite: bool = True   # replace a file of the same name (the Save As dialog asks first)
    download: bool = True    # also send the file back (the UI only saves it)


def _export_folder(folder: str) -> Path:
    if not folder.strip():
        return EXPORT_DIR
    path = Path(folder.strip()).expanduser()
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail=f"Enter the full path of the folder, not '{folder}'.")
    return path


def _export_name(name: str, default_stem: str, ext: str) -> str:
    stem = name.strip()
    if stem.lower().endswith(f".{ext}"):
        stem = stem[: -len(ext) - 1]
    return f"{safe_filename(stem) if stem else default_stem}.{ext}"


def _saved_download(
    target: ExportTarget, default_stem: str, ext: str, data: bytes, media_type: str,
    files: dict[str, bytes] | None = None,
) -> Response:
    """Save the file as ``target`` says and send it back (or, without ``target.download``, where it was
    saved). With ``files`` (paths relative to the file) they are saved next to it too, and the
    download is a zip of all of them."""
    filename = _export_name(target.name, default_stem, ext)
    folder = _export_folder(target.folder)
    saved_path = folder / filename
    if saved_path.exists() and not target.overwrite:
        raise HTTPException(status_code=409, detail=f"{saved_path} already exists.")
    try:
        write_tree(folder, {filename: data, **(files or {})})
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Cannot save to {folder}: {exc}") from exc
    if not target.download:
        return JSONResponse({"saved_path": str(saved_path), "files": len(files or {})})
    if files:
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, data)
            for rel, blob in files.items():
                zf.writestr(rel, blob)
        filename, data, media_type = f"{Path(filename).stem}_usd.zip", package.getvalue(), "application/zip"
    ext = Path(filename).suffix
    # HTTP headers are latin-1 only: send an ASCII fallback plus an RFC 5987 UTF-8 filename.
    return Response(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=\"map_layout{ext}\"; filename*=UTF-8''{quote(filename)}",
            "X-Saved-Path": quote(str(saved_path), safe=":/\\"),
        },
    )


@app.post("/api/export/usd")
def export_usd(
    config: BuildConfig, ext: str = "usd", target: Annotated[ExportTarget, Depends()] = ExportTarget(),
    x_map_revision: MapRevision = None,
) -> Response:
    clean_ext = "usda" if ext.lower() == "usda" else "usd"
    name, data, files = state.build_usd(config, f".{clean_ext}", x_map_revision)
    stem = safe_filename(name.replace(" ", "_"))
    return _saved_download(target, stem, clean_ext, data, "application/octet-stream", files)


@app.post("/api/export/ros-map")
def export_ros_map(
    config: BuildConfig, target: Annotated[ExportTarget, Depends()] = ExportTarget(),
    x_map_revision: MapRevision = None,
) -> Response:
    """The map as a Nav2 occupancy map (.yaml + .pgm in a zip), which is also the project save."""
    try:
        stem, data = state.build_ros_map(config, x_map_revision)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _saved_download(target, stem, "zip", data, "application/zip")


# A native "Save As" dialog, in its own process so Tk never runs on a server thread. It asks before
# replacing an existing file. Arguments: folder, file name, extension, file type label.
_SAVE_DIALOG = """
import sys, tkinter
from tkinter import filedialog
folder, name, ext, label = sys.argv[1:5]
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
print(filedialog.asksaveasfilename(parent=root, title="Save As", initialdir=folder, initialfile=name,
                                   defaultextension=ext, filetypes=[(label, "*" + ext)]))
"""
_EXPORT_TYPES = {"usd": "USD (binary)", "usda": "USD (text)", "zip": "Map save (ROS zip)"}


class SaveAsRequest(BaseModel):
    ext: Literal["usd", "usda", "zip"]
    name: str = ""    # suggested file name (without extension)
    folder: str = ""  # folder to start in (default: exports/)


@app.post("/api/export/save-as")
def choose_export_path(req: SaveAsRequest, request: Request) -> dict:
    """Ask where to save an export in a dialog on this computer. Returns the chosen ``folder`` and
    ``name``, or empty strings when the user cancels."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(status_code=400, detail="The Save As dialog only opens on the computer running the app.")
    start = Path(req.folder.strip()).expanduser() if req.folder.strip() else EXPORT_DIR
    while not start.is_dir() and start != start.parent:
        start = start.parent
    name = safe_filename(req.name.strip()) if req.name.strip() else "map_layout"
    try:
        done = subprocess.run(
            [sys.executable, "-c", _SAVE_DIALOG, str(start if start.is_dir() else DATA_DIR), f"{name}.{req.ext}",
             f".{req.ext}", _EXPORT_TYPES[req.ext]],
            capture_output=True, text=True, encoding="utf-8", timeout=1800,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # file names in any language
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(status_code=500, detail=f"The Save As dialog could not be opened: {exc}") from exc
    if done.returncode != 0:
        raise HTTPException(status_code=500, detail="The Save As dialog could not be opened.")
    chosen = done.stdout.strip()
    if not chosen:
        return {"folder": "", "name": ""}
    path = Path(chosen)
    return {"folder": str(path.parent), "name": path.name}


# --- 3D models (equipment) -------------------------------------------------------------------

class ModelSelectRequest(BaseModel):
    token: str = Field(min_length=1, max_length=64)
    main: str = Field(min_length=1, max_length=4096)


class ModelImportRequest(BaseModel):
    token: str = Field(min_length=1, max_length=64)
    paths: list[str] = Field(default_factory=list, max_length=4000)  # empty: the whole model
    unit: Literal["auto", "m", "cm", "mm", "in"] = "auto"
    up_axis: Literal["auto", "z", "y"] = "auto"


@app.post("/api/models/inspect")
def inspect_model(files: list[UploadFile] = File(...)) -> dict:
    """Open an uploaded model and list its parts for the import dialog. The upload is one USD / STEP /
    STL file, a .zip, or a folder (each file named by its path in the folder)."""
    try:
        if len(files) == 1:
            source = open_model_source(files[0].filename or "model", files[0].file.read())
        else:
            source = open_model_files({f.filename or f"file_{i}": f.file.read() for i, f in enumerate(files)})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": state.add_model_source(source), **source.describe()}


@app.post("/api/models/select")
def select_model_file(req: ModelSelectRequest) -> dict:
    """Use another model file of the same upload (a folder or zip holding several)."""
    source = state.model_source(req.token)
    try:
        source.select(req.main)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": req.token, **source.describe()}


@app.post("/api/models/import")
def import_model_parts(req: ModelImportRequest) -> dict:
    source = state.model_source(req.token)
    try:
        asset = import_model(source, req.paths, req.unit, req.up_axis)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with state.map_lock:
        state.session.models[asset.id] = asset
    return asset.summary()


@app.get("/api/models")
def list_models() -> list[dict]:
    return [m.summary() for m in state.session.models.values()]


@app.get("/api/models/{model_id}/preview")
def model_preview(model_id: str) -> dict:
    asset = state.session.models.get(model_id)
    if asset is None:
        raise HTTPException(status_code=404, detail="This model is no longer loaded.")
    return asset.preview()


@app.get("/api/models/{model_id}/textures/{index}")
def model_texture(model_id: str, index: int) -> Response:
    """A texture of the model's preview (PNG, at most 512 px)."""
    asset = state.session.models.get(model_id)
    textures = asset.preview_mesh.textures if asset is not None else []
    if not 0 <= index < len(textures):
        raise HTTPException(status_code=404, detail="No such texture.")
    return Response(content=textures[index], media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict:
    with state.map_lock:
        state.session.models.pop(model_id, None)
    return {"deleted": model_id}


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
def push_to_isaac(req: IsaacPushRequest, x_map_revision: MapRevision = None) -> dict:
    name, data, files = state.build_usd(req.build_config, ".usd", x_map_revision)
    state.save_isaac_config(req.isaac_config)
    return push_usd_to_isaac(data, name, req.isaac_config, files)


@app.post("/api/isaac/push-models")
def push_models_to_isaac(req: IsaacPushRequest, x_map_revision: MapRevision = None) -> dict:
    """Write only the placed 3D models into the scene open in Isaac Sim (obstacles stay untouched)."""
    with state.map_lock:
        state.check_map_revision(x_map_revision)
        name = state.session.raw.metadata.name if state.session.raw is not None else "map"
        models = dict(state.session.models)
    try:
        layer = export_equipment_usda(req.build_config, models)
    except ValueError as exc:  # a placed model that is no longer loaded
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    samples = equipment_floor_samples(req.build_config, models)
    files = placed_model_files(req.build_config, models)
    state.save_isaac_config(req.isaac_config)
    return push_equipment_to_isaac(layer, name, req.isaac_config, samples, files)


def _restored_placements(scene: dict, models: dict) -> list[EquipmentPlacement]:
    """Placed models found in the pulled stage, as placements (their models are loaded in the app)."""
    placements = []
    for item in scene.get("equipment") or []:
        try:
            yaw = (float(item["yaw_deg"]) + 180.0) % 360.0 - 180.0
            placement = EquipmentPlacement(
                model_id=item["model_id"], name=str(item.get("name") or "Model")[:128] or "Model",
                x=round(float(item["x"]), 4), y=round(float(item["y"]), 4), yaw_deg=round(yaw, 3),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if placement.model_id in models and np.isfinite([placement.x, placement.y]).all():
            placements.append(placement)
    return placements[:500]


@app.post("/api/isaac/pull")
def pull_from_isaac(req: IsaacPushRequest, x_map_revision: MapRevision = None) -> dict:
    with state.map_lock:
        state.check_map_revision(x_map_revision)
        revision = state.session.edit_revision
        known_models = sorted(state.session.models)
    settings = req.isaac_config.pull
    height = req.build_config.vehicle_proxy.height
    if not np.isfinite(height) or height <= settings.floor_clearance_m:
        raise HTTPException(status_code=422, detail="AGV height must be greater than the floor clearance.")
    low = settings.floor_elevation_m + settings.floor_clearance_m
    high = settings.floor_elevation_m + height
    try:
        scene = pull_stage_slice(
            req.isaac_config,
            low,
            high,
            resolution=req.build_config.grid_resolution,
            known_models=known_models,
        )
        raw = map_from_stage_slice(scene, req.build_config.grid_resolution, low, high)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Keep the stage's meter coordinates and preserve its sampled cells on the initial rebuild.
    config = req.build_config.model_copy(update={
        "grid_resolution": raw.metadata.resolution,
        "reconstruction_mode": "aabb",
        "morph_close_cells": 0,
        "dilate_cells": 0,
        "min_cluster_cells": 1,
        "fill_solid_obstacles": False,
        "seal_unknown": False,
        "origin_x": 0.0,
        "origin_y": 0.0,
        "origin_yaw_deg": 0.0,
    })
    with state.map_lock:
        state.check_map_revision(revision)
        state.save_isaac_config(req.isaac_config)
        # Placed models come back as movable models (the stage frame is the map frame: origin 0).
        config.equipment = _restored_placements(scene, state.session.models)
        raw.metadata.extra["isaac_restored_models"] = len(config.equipment)
        raw.metadata.extra["isaac_unknown_models"] = int(scene.get("unknown_equipment") or 0)
        state.session.use(raw, config=config, keep_equipment=True)
        return state.payload()
