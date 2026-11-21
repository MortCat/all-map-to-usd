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

from allmaptousd.web.server import app  # noqa: E402


def _find_free_port(preferred: int = 8765) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


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

    port = _find_free_port(args.port)
    url = f"http://{args.host}:{port}"

    if args.headless:
        print(f"AllMapToUSD server running at {url}")
        uvicorn.run(app, host=args.host, port=port, log_level="info")
        return

    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.6)

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
