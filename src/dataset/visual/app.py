from __future__ import annotations

import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    from .chart_specs import build_dashboard
    from .log_loader import build_dir_signature, load_log_bundle, parse_rank_query
except ImportError:
    from chart_specs import build_dashboard
    from log_loader import build_dir_signature, load_log_bundle, parse_rank_query


CACHE_LOCK = threading.Lock()
CACHE: Dict[str, Dict[str, Any]] = {}


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(value) for value in obj]
    return str(obj)


def get_cached_bundle(log_dir: Path):
    key = str(log_dir.resolve())
    signature = build_dir_signature(log_dir)
    with CACHE_LOCK:
        cache_item = CACHE.get(key)
        if cache_item and cache_item.get("signature") == signature:
            return cache_item["bundle"], signature, True

    bundle = load_log_bundle(log_dir)
    with CACHE_LOCK:
        CACHE[key] = {
            "signature": signature,
            "bundle": bundle,
        }
    return bundle, signature, False


class DashboardRequestHandler(BaseHTTPRequestHandler):
    assets_dir: Path = Path(__file__).resolve().parent / "assets"
    project_root: Path = Path(__file__).resolve().parents[3]
    default_log_dir: Path = project_root / "runs/DThinkVLN-P-7B-GRPO-16-sample/reward_logs"

    def log_message(self, fmt: str, *args) -> None:
        # Keep default server quiet; explicit prints are easier to read.
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self._serve_asset("index.html")
            return
        if path.startswith("/assets/"):
            rel = path[len("/assets/") :]
            self._serve_asset(rel)
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/data":
            self._handle_api_data(parsed.query)
            return

        self._send_error(HTTPStatus.NOT_FOUND, f"Unsupported path: {path}")

    def _resolve_log_dir(self, log_dir_raw: Optional[str]) -> Path:
        if not log_dir_raw:
            return self.default_log_dir
        candidate = Path(log_dir_raw)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        return candidate

    def _handle_api_data(self, query: str) -> None:
        params = parse_qs(query)
        log_dir_raw = params.get("log_dir", [str(self.default_log_dir)])[0]
        ranks_raw = params.get("ranks", ["all"])[0]

        log_dir = self._resolve_log_dir(log_dir_raw)
        if not log_dir.exists() or not log_dir.is_dir():
            self._send_error(HTTPStatus.BAD_REQUEST, f"Invalid log_dir: {log_dir}")
            return

        try:
            bundle, signature, cache_hit = get_cached_bundle(log_dir)
            ranks = parse_rank_query(ranks_raw, bundle.available_ranks)
            dashboard = build_dashboard(bundle, ranks)
            dashboard["meta"]["cache_hit"] = bool(cache_hit)
            dashboard["meta"]["cache_signature"] = signature
            dashboard["meta"]["available_ranks"] = bundle.available_ranks
            dashboard["meta"]["rank_query"] = ranks_raw
            dashboard["meta"]["requested_log_dir"] = str(log_dir)
            self._send_json(_json_safe(dashboard))
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to build dashboard: {exc}")

    def _serve_asset(self, rel_path: str) -> None:
        rel = Path(rel_path).as_posix().lstrip("/")
        target = (self.assets_dir / rel).resolve()

        if not str(target).startswith(str(self.assets_dir.resolve())):
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        if not target.exists() or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, f"Asset not found: {rel}")
            return

        suffix = target.suffix.lower()
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".svg": "image/svg+xml",
            ".png": "image/png",
        }.get(suffix, "application/octet-stream")

        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        payload = {"error": message, "status": int(status)}
        self._send_json(payload, status=status)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    default_log_dir = project_root / "runs/DThinkVLN-P-7B-GRPO-16-sample/reward_logs"

    parser = argparse.ArgumentParser(description="DThinkVLN reward log visual dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=18091, help="Bind port")
    parser.add_argument(
        "--log-dir",
        default=str(default_log_dir),
        help="Default reward_logs directory (can be overridden by query param)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    DashboardRequestHandler.default_log_dir = Path(args.log_dir).resolve()
    DashboardRequestHandler.assets_dir = Path(__file__).resolve().parent / "assets"
    DashboardRequestHandler.project_root = Path(__file__).resolve().parents[3]

    server = ThreadingHTTPServer((args.host, args.port), DashboardRequestHandler)
    print(f"Dashboard server running at http://{args.host}:{args.port}")
    print(f"Default log dir: {DashboardRequestHandler.default_log_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
