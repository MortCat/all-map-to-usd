"""Entry point for AllMapToUSD (Desktop WebView App or Local Browser mode)."""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import uvicorn  # noqa: E402

from allmaptousd.web import server as web_server  # noqa: E402

_ANY_INTERFACE = {"0.0.0.0", "::", ""}
_STARTUP_TIMEOUT_SEC = 15.0


def _family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _find_free_port(host: str, preferred: int = 8765) -> int:
    """``preferred`` unless something already answers there or it cannot be bound on ``host``."""
    probe = "127.0.0.1" if host in _ANY_INTERFACE else host
    with socket.socket(_family(probe), socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        in_use = s.connect_ex((probe, preferred)) == 0
    with socket.socket(_family(host), socket.SOCK_STREAM) as s:
        try:
            if in_use:
                raise OSError
            s.bind((host, preferred))
        except OSError:
            s.bind((host, 0))
        return int(s.getsockname()[1])


def _start_server(server: uvicorn.Server) -> threading.Thread:
    """Run uvicorn in the background and return once it accepts connections."""
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + _STARTUP_TIMEOUT_SEC
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            server.should_exit = True
            raise SystemExit(f"AllMapToUSD server did not start on {server.config.host}:{server.config.port}.")
        time.sleep(0.05)
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="AllMapToUSD Launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in system web browser instead of pywebview desktop window.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run only the FastAPI server without opening a window or browser.",
    )
    args = parser.parse_args()

    port = _find_free_port(args.host, args.port)
    # Accept requests addressed to the bound host too; on every interface any name has to be allowed.
    web_server.ALLOWED_HOSTS.add("*" if args.host in _ANY_INTERFACE else args.host)
    browse_host = "127.0.0.1" if args.host in _ANY_INTERFACE else args.host
    url = f"http://[{browse_host}]:{port}" if ":" in browse_host else f"http://{browse_host}:{port}"

    if args.headless:
        print(f"AllMapToUSD server running at {url}")
        uvicorn.run(web_server.app, host=args.host, port=port, log_level="info")
        return

    server = uvicorn.Server(
        uvicorn.Config(web_server.app, host=args.host, port=port, log_level="warning")
    )
    thread = _start_server(server)

    if args.browser:
        print(f"Opening AllMapToUSD in browser: {url}")
        webbrowser.open(url)
        try:
            while thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            server.should_exit = True
        return

    try:
        import webview  # type: ignore

        webview.create_window(
            "AllMapToUSD — AGV/AMR Map to Isaac Sim USD",
            url,
            width=1360,
            height=860,
            min_size=(1024, 680),
        )
        webview.start()
    except Exception as exc:
        print(f"pywebview unavailable ({exc}), falling back to browser: {url}")
        webbrowser.open(url)
        try:
            while thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
