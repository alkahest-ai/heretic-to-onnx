from __future__ import annotations

import argparse
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class BrowserChatHandler(SimpleHTTPRequestHandler):
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

    def do_GET(self) -> None:  # noqa: N802
        if self._maybe_redirect_root():
            return
        super().do_GET()

    def do_HEAD(self) -> None:  # noqa: N802
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
