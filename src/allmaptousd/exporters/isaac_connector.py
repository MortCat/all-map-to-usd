"""Live Isaac Sim python_server client for pushing layouts and pulling stage geometry.

Speaks the exact TCP protocol of ``isaacsim.code_editor.python_server`` (default port 8226).
When targeting a remote host IP (e.g. 192.168.3.158) whose python_server is bound to
remote loopback 127.0.0.1:8226, this connector transparently establishes and reuses an
SSH local port-forwarding tunnel so the user always enters the real remote IP.
"""

from __future__ import annotations

import atexit
import base64
import gzip
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from allmaptousd.model_assets import write_tree
from allmaptousd.models import IsaacSimConfig

_LOCAL_HOSTS = ("", "127.0.0.1", "localhost", "::1")
# Key-based, non-interactive OpenSSH (uses ~/.ssh/config); shared by the tunnel, ssh mkdir and scp.
_SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=5")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console flash on Windows


def _is_local_host(host: str) -> bool:
    return host.strip() in _LOCAL_HOSTS


def _ssh_target(host: str, user: str = "") -> str:
    return f"{user.strip()}@{host.strip()}" if user.strip() else host.strip()


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-]", "_", name.strip())
    return cleaned or "agv_map_layout"


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _pick_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _SshTunnelManager:
    """Manage background SSH port-forwarding tunnels keyed by (host, remote_port, user)."""

    def __init__(self) -> None:
        self._tunnels: dict[tuple[str, int, str], tuple[int, subprocess.Popen]] = {}
        atexit.register(self.close_all)

    def ensure_tunnel(
        self,
        remote_host: str,
        remote_port: int,
        ssh_user: str = "",
        wait_sec: float = 6.0,
    ) -> int:
        key = (remote_host, int(remote_port), ssh_user.strip())
        existing = self._tunnels.get(key)
        if existing is not None:
            local_port, proc = existing
            if proc.poll() is None and _is_port_open("127.0.0.1", local_port, timeout=0.4):
                return local_port
            self._tunnels.pop(key, None)

        local_port = _pick_free_local_port()
        target = _ssh_target(remote_host, ssh_user)
        cmd = [
            "ssh", "-N", "-L", f"127.0.0.1:{local_port}:127.0.0.1:{int(remote_port)}", *_SSH_OPTS,
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
            target,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)

        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                err = (proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "").strip()
                raise OSError(f"SSH tunnel failed ({target}): {err or f'exit {proc.returncode}'}")
            if _is_port_open("127.0.0.1", local_port, timeout=0.25):
                self._tunnels[key] = (local_port, proc)
                return local_port
            time.sleep(0.15)

        proc.terminate()
        raise OSError(f"SSH tunnel timed out ({target} -> 127.0.0.1:{remote_port})")

    def close_all(self) -> None:
        for _, proc in self._tunnels.values():
            if proc.poll() is None:
                proc.terminate()
        self._tunnels.clear()


_tunnel_mgr = _SshTunnelManager()


def _resolve_endpoint(cfg: IsaacSimConfig) -> tuple[str, int, str]:
    """Return (connect_host, connect_port, transport_label) for the given config."""
    host = cfg.host.strip() or "127.0.0.1"
    port = int(cfg.port)
    if _is_local_host(host):
        return host, port, "local"

    # Try direct TCP first in case the remote Isaac Sim binds to 0.0.0.0
    if _is_port_open(host, port, timeout=0.5):
        return host, port, "direct"

    if not cfg.use_ssh_tunnel:
        return host, port, "direct"

    # Auto-establish an isolated SSH tunnel to remote_host's 127.0.0.1:<port>
    local_port = _tunnel_mgr.ensure_tunnel(host, port, ssh_user=cfg.ssh_user)
    return "127.0.0.1", local_port, "ssh_tunnel"


def send_python_cell(
    host: str,
    port: int,
    source: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Send a Python snippet to Isaac Sim's python_server and parse its JSON reply."""
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(source.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)

        chunks: list[bytes] = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)

    raw = b"".join(chunks).decode("utf-8", errors="replace").strip()
    if not raw:
        return {"status": "error", "evalue": "Empty response from Isaac Sim python_server."}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"status": "ok", "output": raw}


def _last_json_line(resp: dict[str, Any]) -> dict[str, Any]:
    """The python cells print a JSON summary as their last line; return it (or {} if absent)."""
    for line in reversed(str(resp.get("output") or "").strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def build_pull_script(cfg: IsaacSimConfig, low: float, high: float) -> str:
    """Run the projection against the active stage without authoring any USD changes."""
    projection = Path(__file__).with_name("usd_slice.py").read_text(encoding="utf-8")
    arguments = (
        cfg.pull.root_prim_path, cfg.pull.exclude_prim_path, low, high, cfg.pull.geometry == "collision_only",
    )
    return projection + f"""

import base64, gzip, json
import omni.timeline
import omni.usd

def _pull_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise ValueError("No active stage in Isaac Sim.")
    seconds = omni.timeline.get_timeline_interface().get_current_time()
    time = Usd.TimeCode(seconds * stage.GetTimeCodesPerSecond())
    result = collect_stage_slice(stage, time, *{arguments!r})
    data = json.dumps(result, separators=(",", ":"), allow_nan=False).encode("utf-8")
    print(json.dumps({{"stage_slice": base64.b64encode(gzip.compress(data, compresslevel=3)).decode("ascii")}}))

_pull_stage()
"""


def pull_stage_slice(cfg: IsaacSimConfig, low: float, high: float) -> dict[str, Any]:
    """Read the stage slice over the same direct or SSH connection used by Push."""
    try:
        host, port, _ = _resolve_endpoint(cfg)
        reply = send_python_cell(host, port, build_pull_script(cfg, low, high), timeout=cfg.timeout_sec)
    except OSError as exc:
        raise RuntimeError(f"Cannot pull from Isaac Sim ({cfg.host}:{cfg.port}): {exc}") from exc
    if reply.get("status") == "error":
        error = reply.get("evalue") or "No stage data returned."
        if reply.get("ename") == "ValueError":
            raise ValueError(error)
        raise RuntimeError(f"Isaac Sim pull failed: {error}")
    try:
        payload = _last_json_line(reply)["stage_slice"]
        data = json.loads(gzip.decompress(base64.b64decode(payload, validate=True)))
        if not isinstance(data, dict) or not {"stage", "prim_count", "projected", "sections"} <= data.keys():
            raise ValueError("Invalid stage projection")
        return data
    except (KeyError, OSError, EOFError, TypeError, ValueError) as exc:
        raise RuntimeError("Isaac Sim did not return a valid stage projection.") from exc


def check_isaac_connection(cfg: IsaacSimConfig) -> dict[str, Any]:
    """Verify live Isaac Sim python_server connectivity and inspect the active Stage."""
    probe_code = (
        "import json\n"
        "try:\n"
        "    import omni.usd\n"
        "    stage = omni.usd.get_context().get_stage()\n"
        "    stage_url = stage.GetRootLayer().identifier if stage else None\n"
        "    prim_count = len(list(stage.Traverse())) if stage else 0\n"
        "    print(json.dumps({'stage_url': stage_url, 'prim_count': prim_count}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'stage_url': None, 'prim_count': 0, 'note': str(exc)}))\n"
    )
    try:
        conn_host, conn_port, transport = _resolve_endpoint(cfg)
        resp = send_python_cell(conn_host, conn_port, probe_code, timeout=min(cfg.timeout_sec, 6.0))
    except OSError as exc:
        return {
            "reachable": False,
            "host": cfg.host,
            "port": int(cfg.port),
            "message": (
                f"Cannot connect to {cfg.host}:{cfg.port} ({exc}). "
                "Make sure isaacsim.code_editor.python_server is enabled."
            ),
        }

    if resp.get("status") == "error":
        return {
            "reachable": False,
            "host": cfg.host,
            "port": int(cfg.port),
            "message": f"Isaac Sim error: {resp.get('ename', '')}: {resp.get('evalue', '')}",
            "raw": resp,
        }

    info = _last_json_line(resp)

    stage_url = info.get("stage_url") or "(Unsaved / Empty Stage)"
    prim_count = info.get("prim_count", 0)
    via = " (SSH Tunnel)" if transport == "ssh_tunnel" else ""
    return {
        "reachable": True,
        "host": cfg.host,
        "port": int(cfg.port),
        "transport": transport,
        "stage_url": stage_url,
        "prim_count": prim_count,
        "message": f"Connected {cfg.host}:{cfg.port}{via} · Stage: {stage_url} ({prim_count} prims)",
    }


def _home_relative(path: str) -> str:
    """'~/maps' -> 'maps' so ssh/scp resolve it against the remote home directory."""
    path = path.strip() or "allmaptousd_maps"
    return path[2:] if path.startswith("~/") else path


def deliver_files(files: dict[str, bytes], cfg: IsaacSimConfig) -> str:
    """Copy ``files`` (paths relative to ``remote_save_dir``; the first one is the main file) to the
    Isaac host. Local hosts get direct file writes; remote hosts get ``ssh mkdir -p`` + ``scp``
    (key-based login from ~/.ssh/config, same as the tunnel). Returns a description of the copy."""
    main = next(iter(files))
    if _is_local_host(cfg.host):
        target_dir = Path(os.path.expanduser(cfg.remote_save_dir)).resolve()
        write_tree(target_dir, files)
        return f"local copy -> {target_dir / main}"

    remote_dir = _home_relative(cfg.remote_save_dir)
    target = _ssh_target(cfg.host, cfg.ssh_user)
    run_kw = dict(capture_output=True, text=True, timeout=max(cfg.timeout_sec, 120.0), creationflags=_NO_WINDOW)

    mk = subprocess.run(["ssh", *_SSH_OPTS, target, f"mkdir -p -- {shlex.quote(remote_dir)}"], **run_kw)
    if mk.returncode != 0:
        raise OSError(f"ssh mkdir failed ({target}): {mk.stderr.strip() or mk.returncode}")

    with tempfile.TemporaryDirectory() as tmp:
        write_tree(Path(tmp), files)
        tops = sorted({rel.split("/")[0] for rel in files})  # folders included: one scp for all
        cmd = ["scp", "-q", "-r", *_SSH_OPTS, *(str(Path(tmp) / t) for t in tops), f"{target}:{remote_dir}/"]
        cp = subprocess.run(cmd, **run_kw)
    if cp.returncode != 0:
        raise OSError(f"scp failed ({target}): {cp.stderr.strip() or cp.returncode}")
    extra = f" (+{len(files) - 1} model files)" if len(files) > 1 else ""
    return f"scp -> {target}:{remote_dir}/{main}{extra}"


# Writes files that travel inside a python cell into the save folder (paths stay inside it).
_WRITE_FILES = """
def _write_files(save_dir, packed):
    for rel, blob in packed:
        path = (save_dir / rel).resolve()
        if save_dir not in path.parents:
            raise ValueError(f"Refusing to write outside {save_dir}: {rel}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.decompress(base64.b64decode(blob)))
""".strip()


def _pack_files(files: dict[str, bytes] | None) -> list[tuple[str, str]]:
    return [(rel, base64.b64encode(zlib.compress(data, 6)).decode("ascii")) for rel, data in (files or {}).items()]


def _deliver_and_run(
    cfg: IsaacSimConfig, files: dict[str, bytes], build_script: Callable[[bool], str], timeout: float,
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    """Ship ``files`` (the main file first) to the Isaac host and run the cell ``build_script(inline)``.

    The files travel inside the cell (inline), or are copied beforehand by :func:`deliver_files` when
    ``transfer_mode`` says so or they exceed ``inline_limit_mb``. Returns (failure result or None,
    cell response, transport, delivery)."""
    size = sum(len(d) for d in files.values())
    inline = not (cfg.transfer_mode == "file" or (
        cfg.transfer_mode == "auto" and size > cfg.inline_limit_mb * 1024 * 1024))
    delivery, transport = "inline", ""
    try:
        if not inline:
            delivery = deliver_files(files, cfg)
        script = build_script(inline)
        conn_host, conn_port, transport = _resolve_endpoint(cfg)
        resp = send_python_cell(conn_host, conn_port, script, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        failure = {"ok": False, "message": f"Cannot deliver to Isaac Sim ({cfg.host}:{cfg.port}, {delivery}): {exc}"}
        return failure, {}, transport, delivery
    if resp.get("status") == "error":
        return {
            "ok": False,
            "message": f"Isaac Sim execution failed [{resp.get('ename', 'Error')}]: {resp.get('evalue', '')}",
            "traceback": "\n".join(resp.get("traceback") or []),
        }, resp, transport, delivery
    return None, resp, transport, delivery


_FRAME_PUSHED_MAP = """
def _frame_pushed_map(stage, mounted_at):
    from omni.kit.viewport.utility import frame_viewport_prims, get_active_viewport
    from pxr import Gf, Usd

    viewport = get_active_viewport()
    if viewport is None or viewport.stage != stage:
        return False
    paths = []
    for suffix in ("Map/Obstacles", "Map/ForbiddenZones"):
        prim = stage.GetPrimAtPath(f"{mounted_at}/{suffix}")
        if prim and prim.GetChildren():
            paths.append(str(prim.GetPath()))
    if not paths:
        paths = [f"{mounted_at}/Environment/Ground"]
        if not stage.GetPrimAtPath(paths[0]):
            paths = [mounted_at]

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bounds = Gf.Range3d()
    for path in paths:
        bounds.UnionWith(cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange())
    if bounds.IsEmpty():
        return False
    center = bounds.GetMidpoint()
    size = max(bounds.GetSize().GetLength(), 1.0)
    y_up = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    up = Gf.Vec3d(0, 1, 0) if y_up else Gf.Vec3d(0, 0, 1)
    direction = Gf.Vec3d(0.7, 1.1, 0.9) if y_up else Gf.Vec3d(0.7, -0.9, 1.1)
    session = stage.GetSessionLayer()
    camera_path = Sdf.Path("/AllMapToUSDPreviewCamera")
    with Usd.EditContext(stage, session):
        # Use a viewport-only camera so framing never overwrites the current scene camera.
        while stage.GetPrimAtPath(camera_path) and not session.GetPrimAtPath(camera_path):
            camera_path = Sdf.Path(str(camera_path) + "_")
        camera = UsdGeom.Camera.Define(stage, camera_path)
        transform = Gf.Matrix4d(1).SetLookAt(center + direction * size, center, up).GetInverse()
        camera.MakeMatrixXform().Set(transform)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(max(0.001, size * 0.0001), max(100000, size * 100)))
        viewport.camera_path = camera_path
        return frame_viewport_prims(viewport, prims=paths)
""".strip()


def build_push_script(
    usd_bytes: bytes | None,
    map_name: str,
    cfg: IsaacSimConfig,
    files: dict[str, bytes] | None = None,
) -> str:
    """Build a Python cell that mounts the USD in the live stage.

    With ``usd_bytes`` the file is embedded (base64) and written on the Isaac host first, along with
    ``files`` (textures etc. of the 3D models, paths relative to the USD); with ``None`` they are
    expected to already be in ``remote_save_dir`` (see deliver_files).
    """
    safe_name = _sanitize_filename(map_name)
    write_line = (
        f"_usd_path.write_bytes(base64.b64decode({base64.b64encode(usd_bytes).decode('ascii')!r}))"
        if usd_bytes is not None
        else "assert _usd_path.is_file(), f'USD not found on Isaac host: {_usd_path}'"
    )
    packed = _pack_files(files if usd_bytes is not None else None)
    write_files = f"{_WRITE_FILES}\n\n_write_files(_save_dir, {packed!r})" if packed else ""
    target_prim = cfg.target_prim_path.strip() or "/World/AGV_Map"
    if not target_prim.startswith("/"):
        target_prim = f"/{target_prim}"

    return f"""
import base64, json, os, zlib
from pathlib import Path
import omni.usd
from pxr import Sdf, UsdGeom, UsdLux

_save_dir = Path(os.path.expanduser({cfg.remote_save_dir!r})).resolve()
_save_dir.mkdir(parents=True, exist_ok=True)
_usd_path = (_save_dir / "{safe_name}.usd").resolve()
{write_line}
{write_files}
_usd_str = _usd_path.as_posix()

_existing_layer = Sdf.Layer.Find(_usd_str)
if _existing_layer is not None:
    _existing_layer.Reload(force=True)

_ctx = omni.usd.get_context()
_mode = {cfg.push_mode!r}
_target_prim = {target_prim!r}
_add_light = {bool(cfg.add_dome_light)!r}

if _mode == "open_stage":
    if not _ctx.open_stage(_usd_str):
        raise RuntimeError(f"Cannot open USD stage: {{_usd_str}}")
    _stage = _ctx.get_stage()
    _mounted_at = "/World"
else:
    _stage = _ctx.get_stage()
    if _stage is None:
        _ctx.new_stage()
        _stage = _ctx.get_stage()
    if not _stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(_stage, "/World")
    _prim = _stage.DefinePrim(_target_prim, "Xform")
    _prim.GetReferences().ClearReferences()
    _prim.GetReferences().AddReference(_usd_str)
    _mounted_at = _target_prim

_lighting = UsdGeom.Imageable(_stage.GetPrimAtPath(f"{{_mounted_at}}/Lighting"))
if _lighting and _mode == "reference":
    _target_path = Sdf.Path(_mounted_at)
    _has_scene_lights = any(
        UsdLux.LightAPI(prim) and not prim.GetPath().HasPrefix(_target_path)
        and UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        for prim in _stage.Traverse()
    )
    _lighting.CreateVisibilityAttr().Set(
        UsdGeom.Tokens.inherited if _add_light and not _has_scene_lights else UsdGeom.Tokens.invisible
    )

{_FRAME_PUSHED_MAP}

_framed = False
try:
    _framed = bool(_frame_pushed_map(_stage, _mounted_at))
except Exception as _frame_error:
    print(f"Viewport framing unavailable: {{_frame_error}}")

_stage_url = _stage.GetRootLayer().identifier if _stage else ""
print(json.dumps({{
    "ok": True,
    "remote_usd_path": _usd_str,
    "mounted_prim": _mounted_at,
    "push_mode": _mode,
    "stage_url": _stage_url,
    "viewport_framed": _framed,
}}))
""".strip()


def push_usd_to_isaac(
    usd_bytes: bytes,
    map_name: str,
    cfg: IsaacSimConfig,
    files: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Transfer the generated USD (and ``files``, the textures etc. of its 3D models by their path
    relative to the USD) to Isaac Sim and mount/open it in the live Stage."""
    failure, resp, transport, delivery = _deliver_and_run(
        cfg, {f"{_sanitize_filename(map_name)}.usd": usd_bytes, **(files or {})},
        lambda inline: build_push_script(usd_bytes if inline else None, map_name, cfg, files),
        cfg.timeout_sec,
    )
    if failure:
        return failure
    parsed = _last_json_line(resp)

    via = " (SSH Tunnel)" if transport == "ssh_tunnel" else ""
    return {
        "ok": bool(parsed.get("ok", True)),
        "transport": transport,
        "remote_usd_path": parsed.get("remote_usd_path", ""),
        "mounted_prim": parsed.get("mounted_prim", cfg.target_prim_path),
        "push_mode": parsed.get("push_mode", cfg.push_mode),
        "stage_url": parsed.get("stage_url", ""),
        "viewport_framed": bool(parsed.get("viewport_framed")),
        "elapsed_seconds": resp.get("elapsed_seconds"),
        "delivery": delivery,
        "message": (
            f"Mounted to {cfg.host}{via} · Prim: {parsed.get('mounted_prim', cfg.target_prim_path)} "
            f"(File: {parsed.get('remote_usd_path', '')}, {delivery})"
            + (" · View framed" if parsed.get("viewport_framed") else "")
        ),
    }


# How far the automatic floor search looks around the Floor Elevation setting (metres).
FLOOR_SEARCH_BELOW_M = 1.0
FLOOR_SEARCH_ABOVE_M = 0.5
