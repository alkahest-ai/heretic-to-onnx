from __future__ import annotations

import argparse
import json
import os
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class BrowserChatHandler(SimpleHTTPRequestHandler):
    def _read_local_hf_token(self) -> str:
        env_token = os.environ.get("HF_TOKEN", "").strip()
        if env_token:
            return env_token

        token_path = Path.home() / ".cache" / "huggingface" / "token"
        try:
            return token_path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _maybe_redirect_root(self) -> bool:
        if self.path in {"", "/"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/browser-chat/")
            self.end_headers()
            return True
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return True
        return False

    def _maybe_send_local_hf_token(self) -> bool:
        if self.path != "/__hf_token":
            return False

        token = self._read_local_hf_token()
        if not token:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "token": ""})
            return True

        self._send_json(HTTPStatus.OK, {"ok": True, "token": token})
        return True

    def do_GET(self) -> None:  # noqa: N802
        if self._maybe_send_local_hf_token():
            return
        if self._maybe_redirect_root():
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path == "/__hf_token":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if self._maybe_redirect_root():
            return
        super().do_HEAD()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, default=4173)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    handler = partial(BrowserChatHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Serving {root} at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
