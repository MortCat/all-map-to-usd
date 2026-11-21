"""First-run setup and launcher for AllMapToUSD (standard library only; started by start_app.cmd).

* ``.venv`` is ready and matches pyproject.toml -> start the app straight away.
* ``.venv`` is missing, broken or outdated     -> show a small setup window that creates the
  virtual environment, installs the app with ``pip``, then starts it.

Without tkinter the same steps run in the console. The full pip output goes to ``setup.log``.
"""

from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
WINDOWS = os.name == "nt"
VENV_PY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
VENV_PYW = VENV / "Scripts/pythonw.exe" if WINDOWS else VENV_PY  # no console window for the app
STAMP = VENV / ".allmaptousd-setup"
LOG_FILE = ROOT / "setup.log"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # keep child consoles hidden under pythonw
MIN_PYTHON = (3, 11)


# ---------------------------------------------------------------------------
# Setup logic (no UI)
# ---------------------------------------------------------------------------

def requirements_hash() -> str:
    """Setup is redone whenever the declared dependencies change."""
    return hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()


def venv_works() -> bool:
    if not VENV_PY.exists():
        return False
    try:
        return subprocess.run([str(VENV_PY), "-c", "import sys"], creationflags=NO_WINDOW,
                              capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def needs_setup() -> bool:
    try:
        stamp_ok = STAMP.read_text(encoding="utf-8").strip() == requirements_hash()
    except OSError:
        stamp_ok = False
    return not (stamp_ok and venv_works())


def launch_app(args: list[str]) -> None:
    """Start the app detached from this launcher; its output goes to app.log (pythonw has no console)."""
    exe = VENV_PYW if VENV_PYW.exists() else VENV_PY
    with (ROOT / "app.log").open("w", encoding="utf-8") as log:
        subprocess.Popen([str(exe), str(ROOT / "run_app.py"), *args], cwd=ROOT, creationflags=NO_WINDOW,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


class SetupError(RuntimeError):
    pass


_running: list[subprocess.Popen] = []  # the pip/venv process currently running, for Cancel / close


def stop_running() -> None:
    for proc in _running:
        if proc.poll() is None:
            proc.terminate()


# (title, weight in the overall progress bar)
STEPS = [
    ("Creating the Python environment (.venv)", 10),
    ("Updating pip", 10),
    ("Downloading and installing packages", 70),
    ("Checking the installation", 10),
]
EXPECTED_PACKAGES = 30  # rough count of "Collecting ..." lines, only used for the progress bar


def run_setup(
    on_step: Callable[[int], None],
    on_line: Callable[[str], None],
    on_fraction: Callable[[float], None],
    cancelled: threading.Event,
) -> None:
    """Create/repair .venv and install the app; raises SetupError with a short reason on failure."""
    log = LOG_FILE.open("w", encoding="utf-8")

    def run(cmd: list[str], what: str, count_packages: bool = False) -> None:
        log.write(f"\n$ {' '.join(cmd)}\n")
        env = {**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        _running[:] = [proc]
        collected = 0
        for line in proc.stdout:
            log.write(line)
            line = line.strip()
            if not line:
                continue
            on_line(line)
            if count_packages and line.startswith("Collecting "):
                collected += 1
                on_fraction(min(0.85, collected / EXPECTED_PACKAGES))
        if proc.wait() != 0:
            raise SetupError("Cancelled." if cancelled.is_set() else f"{what} failed (see setup.log).")

    try:
        if sys.version_info < MIN_PYTHON:
            raise SetupError(f"Python {'.'.join(map(str, MIN_PYTHON))}+ is required, found {sys.version.split()[0]}.")

        on_step(0)
        if not venv_works():
            # --clear also repairs a .venv whose base Python was moved or uninstalled
            run([sys.executable, "-m", "venv", "--clear", str(VENV)], "Creating .venv")
        on_step(1)
        run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip"], "Updating pip")
        on_step(2)
        run([str(VENV_PY), "-m", "pip", "install", "-e", ".[desktop]"], "Installing packages", count_packages=True)
        on_step(3)
        run([str(VENV_PY), "-c", "import allmaptousd.web.server, pxr, cv2, webview; print('All modules import.')"],
            "Checking the installation")
        STAMP.write_text(requirements_hash(), encoding="utf-8")
    finally:
        log.close()


# ---------------------------------------------------------------------------
# Setup window (tkinter) - colours follow the app's dark theme (web/static/app.css)
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#0b0f17", "surface": "#121824", "surface2": "#182030", "border": "#263247",
    "text": "#e8eef8", "dim": "#94a3b8", "muted": "#64748b",
    "accent": "#38bdf8", "ok": "#22c55e", "danger": "#ef4444",
}


def run_window(app_args: list[str]) -> None:
    import tkinter as tk
    from tkinter import ttk

    c = COLORS
    root = tk.Tk()
    root.title("AllMapToUSD Setup")
    root.configure(bg=c["bg"])
    root.resizable(False, False)
    w, h = 560, 300
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Bar.Horizontal.TProgressbar", troughcolor=c["surface2"], background=c["accent"],
                    bordercolor=c["border"], lightcolor=c["accent"], darkcolor=c["accent"], thickness=8)

    font = ("Segoe UI", 10) if WINDOWS else ("TkDefaultFont", 10)
    frame = tk.Frame(root, bg=c["bg"], padx=22, pady=18)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Setting up AllMapToUSD", bg=c["bg"], fg=c["text"],
             font=(font[0], 14, "bold"), anchor="w").pack(fill="x")
    tk.Label(frame, text="First run: preparing the Python environment. This takes a few minutes "
                         "and only happens once (or after an update).",
             bg=c["bg"], fg=c["dim"], font=font, anchor="w", justify="left",
             wraplength=w - 44).pack(fill="x", pady=(4, 14))

    step_var = tk.StringVar(value="Starting…")
    step_label = tk.Label(frame, textvariable=step_var, bg=c["bg"], fg=c["accent"],
                          font=(font[0], 10, "bold"), anchor="w")
    step_label.pack(fill="x")
    bar = ttk.Progressbar(frame, style="Bar.Horizontal.TProgressbar", maximum=100)
    bar.pack(fill="x", pady=(6, 6))
    detail_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=detail_var, bg=c["surface"], fg=c["muted"], font=("Consolas", 9), anchor="w",
             padx=8, pady=6, highlightthickness=1, highlightbackground=c["border"]).pack(fill="x")

    buttons = tk.Frame(frame, bg=c["bg"])
    buttons.pack(fill="x", side="bottom")

    def button(text: str, cmd: Callable[[], None], primary: bool = False) -> tk.Button:
        return tk.Button(buttons, text=text, command=cmd, relief="flat", cursor="hand2", font=font, padx=14, pady=3,
                         bg=c["accent"] if primary else c["surface2"], fg=c["bg"] if primary else c["text"],
                         activebackground=c["border"], activeforeground=c["text"], bd=0)

    events: queue.Queue = queue.Queue()
    cancelled = threading.Event()
    state = {"step": 0}

    def progress(step: int, fraction: float = 0.0) -> float:
        done = sum(wt for _, wt in STEPS[:step])
        return done + STEPS[step][1] * fraction

    def worker() -> None:
        try:
            run_setup(
                on_step=lambda i: events.put(("step", i)),
                on_line=lambda s: events.put(("line", s)),
                on_fraction=lambda f: events.put(("fraction", f)),
                cancelled=cancelled,
            )
            events.put(("done", None))
        except SetupError as exc:
            events.put(("error", str(exc)))
        except Exception as exc:  # unexpected: still report it in the window
            events.put(("error", f"{type(exc).__name__}: {exc}"))

    def show_buttons(*specs: tuple[str, Callable[[], None], bool]) -> None:
        for child in buttons.winfo_children():
            child.destroy()
        for text, cmd, primary in reversed(specs):
            button(text, cmd, primary).pack(side="right", padx=(8, 0))

    def start() -> None:
        cancelled.clear()
        step_label.configure(fg=c["accent"])
        bar["value"] = 0
        show_buttons(("Cancel", cancel, False))
        threading.Thread(target=worker, daemon=True).start()

    def cancel() -> None:
        cancelled.set()
        stop_running()
        step_var.set("Cancelling…")

    def open_log() -> None:
        if WINDOWS:
            os.startfile(LOG_FILE)  # opens setup.log in the default text editor
        else:
            subprocess.Popen(["xdg-open", str(LOG_FILE)])

    def poll() -> None:
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "step":
                    state["step"] = value
                    step_var.set(f"Step {value + 1} of {len(STEPS)} · {STEPS[value][0]}")
                    bar["value"] = progress(value)
                elif kind == "fraction":
                    bar["value"] = progress(state["step"], value)
                elif kind == "line":
                    detail_var.set(value if len(value) <= 72 else value[:69] + "…")
                elif kind == "done":
                    bar["value"] = 100
                    step_label.configure(fg=c["ok"])
                    step_var.set("Ready · starting AllMapToUSD…")
                    detail_var.set("Setup finished.")
                    show_buttons()
                    launch_app(app_args)
                    root.after(1200, root.destroy)
                    return
                elif kind == "error":
                    step_label.configure(fg=c["danger"])
                    step_var.set(f"Setup failed · {value}")
                    show_buttons(("Open log", open_log, False), ("Retry", start, True), ("Close", close, False))
        except queue.Empty:
            pass
        root.after(100, poll)

    def close() -> None:
        cancelled.set()
        stop_running()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    start()
    root.after(100, poll)
    root.mainloop()


def run_console(app_args: list[str]) -> int:
    """Fallback when tkinter is unavailable (start_app.cmd then runs us with a console)."""
    print("Setting up AllMapToUSD (first run). This takes a few minutes and only happens once.")
    try:
        run_setup(
            on_step=lambda i: print(f"\n[{i + 1}/{len(STEPS)}] {STEPS[i][0]}"),
            on_line=lambda s: print("   ", s),
            on_fraction=lambda f: None,
            cancelled=threading.Event(),
        )
    except SetupError as exc:
        print(f"\nSetup failed: {exc}\nFull log: {LOG_FILE}")
        input("Press Enter to close...")
        return 1
    launch_app(app_args)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if "--force-setup" in args:
        args.remove("--force-setup")
    elif not needs_setup():
        launch_app(args)
        return 0
    try:
        import tkinter  # noqa: F401
    except ImportError:
        return run_console(args)
    run_window(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
