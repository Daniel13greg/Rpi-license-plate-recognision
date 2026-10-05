"""Local HTTP API of the Pi.

GET  /health                          liveness, no token needed, no plates
GET  /                                status page with live camera images (for aiming)
GET  /api/v1/status                   bays, cameras, delivery queue
GET  /api/v1/bays                     all bays
GET  /api/v1/bays/{id}                one bay: state, current plate, last reads
POST /api/v1/bays/{id}/read?wait=10   plate of the car in the bay (waits up to `wait` s)
GET  /api/v1/bays/{id}/snapshot.jpg   current camera image (?annotate=0 for the raw frame)
POST /api/v1/bays/{id}/simulate       inject an event, body {"plate": "ABC123"} (integration tests)
GET  /api/v1/events?limit=50&bay=1    recent events and their delivery status

When ``api.token`` is set, every route except /health needs
``Authorization: Bearer <token>`` or ``?token=<token>``.
"""

from __future__ import annotations

import hmac
import json
import logging
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import cv2

from carwash_lpr import __version__, plates
from carwash_lpr.config import ApiConfig
from carwash_lpr.events import PlateEvent

if TYPE_CHECKING:
    from carwash_lpr.service import Service

log = logging.getLogger(__name__)

_BAY_ROUTE = re.compile(r"^/api/v1/bays/([A-Za-z0-9_-]+)(/read|/snapshot\.jpg|/simulate)?$")
_MAX_BODY = 64 * 1024


class ApiServer:
    def __init__(self, cfg: ApiConfig, service: "Service"):
        handler = type("Handler", (_Handler,), {"service": service, "token": cfg.token})
        self.httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler)
        self.httpd.daemon_threads = True
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="api", daemon=True)

    @property
    def port(self) -> int:
        return self.httpd.server_address[1]

    def start(self) -> None:
        self._thread.start()
        log.info("API listening on %s:%d", *self.httpd.server_address[:2])

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    service: "Service"
    token: str
    _body: bytes = b""
    server_version = f"carwash-lpr/{__version__}"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("api %s - %s", self.address_string(), format % args)

    # --- plumbing ----------------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, indent=2).encode(), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _authorized(self, query: dict) -> bool:
        if not self.token:
            return True
        header = self.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else query.get("token", [""])[0]
        return hmac.compare_digest(supplied.encode(), self.token.encode())

    def _read_body(self) -> bytes:
        # Always consumed, even when unused, so a kept-alive connection stays in sync.
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY:
            self.close_connection = True
            raise ValueError("request body too large")
        return self.rfile.read(length) if length > 0 else b""

    def _read_json(self) -> dict:
        if not self._body.strip():
            return {}
        data = json.loads(self._body)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _route(self, method: str) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"
        try:
            self._body = self._read_body() if method == "POST" else b""
            if path == "/health" and method == "GET":
                health = self.service.health()
                self._json(HTTPStatus.OK if health["status"] == "ok" else HTTPStatus.SERVICE_UNAVAILABLE, health)
                return
            if not self._authorized(query):
                self._error(HTTPStatus.UNAUTHORIZED, "missing or wrong API token")
                return
            if path == "/" and method == "GET":
                self._send(HTTPStatus.OK, _STATUS_PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/api/v1/status" and method == "GET":
                self._json(HTTPStatus.OK, self.service.status())
            elif path == "/api/v1/bays" and method == "GET":
                self._json(HTTPStatus.OK, [bay.status() for bay in self.service.bays.values()])
            elif path == "/api/v1/events" and method == "GET":
                limit = min(max(int(query.get("limit", ["50"])[0]), 1), 500)
                bay = query.get("bay", [None])[0]
                self._json(HTTPStatus.OK, self.service.outbox.recent(limit, bay))
            elif match := _BAY_ROUTE.match(path):
                self._bay_route(method, match.group(1), match.group(2) or "", query)
            else:
                self._error(HTTPStatus.NOT_FOUND, "no such endpoint")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            log.exception("API error on %s %s", method, self.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

    def _bay_route(self, method: str, bay_id: str, action: str, query: dict) -> None:
        worker = self.service.bays.get(bay_id)
        if worker is None:
            self._error(HTTPStatus.NOT_FOUND, f"no bay {bay_id!r}")
            return
        if action == "" and method == "GET":
            self._json(HTTPStatus.OK, worker.status())
        elif action == "/read" and method == "POST":
            wait = min(max(float(query.get("wait", ["10"])[0]), 0.0), 60.0)
            request = worker.request_read("http")
            result = request.wait(wait) if wait > 0 else None
            if result is None:
                status = "pending" if wait == 0 else "timeout"
                code = HTTPStatus.ACCEPTED if wait == 0 else HTTPStatus.OK
                self._json(code, {"bay_id": bay_id, "status": status, "plate": None})
            else:
                self._json(HTTPStatus.OK, result)
        elif action == "/snapshot.jpg" and method == "GET":
            annotate = query.get("annotate", ["1"])[0] not in ("0", "false", "no")
            image = worker.snapshot(annotate)
            if image is None:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "no camera image yet")
                return
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            self._send(HTTPStatus.OK, jpeg.tobytes(), "image/jpeg")
        elif action == "/simulate" and method == "POST":
            body = self._read_json()
            event_type = body.get("event_type", "plate_recognized")
            if event_type not in ("plate_recognized", "plate_unrecognized", "vehicle_left", "test"):
                raise ValueError(f"unknown event_type {event_type!r}")
            found = plates.interpret(str(body.get("plate", ""))) if body.get("plate") else None
            event = PlateEvent(
                event_type=event_type,
                bay_id=bay_id,
                bay_name=worker.bay.name,
                trigger="simulated",
                plate=found.plate if found else None,
                plate_display=found.display if found else None,
                plate_format=found.format if found else None,
                country=found.country if found else None,
                confidence=1.0 if found else None,
                votes=1 if found else 0,
            )
            self.service.emit(event)
            self._json(HTTPStatus.ACCEPTED, event.payload(self.service.cfg.device_id, self.service.cfg.site_id))
        else:
            self._error(HTTPStatus.METHOD_NOT_ALLOWED, f"{method} not supported here")

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        self._route("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._route("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._route("POST")


_STATUS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Car wash LPR</title>
<style>
body{font-family:system-ui,sans-serif;margin:16px;background:#f4f5f7;color:#1d2330}
h1{font-size:20px;margin:0 0 12px}
.bays{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:16px}
.bay{background:#fff;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.15)}
.bay img{width:100%;border-radius:4px;background:#222;min-height:120px}
.plate{font:700 26px/1.2 monospace;letter-spacing:2px}
.muted{color:#667085;font-size:13px}
.bad{color:#b42318}
</style></head><body>
<h1>Car wash plate recognition</h1><div class="muted" id="summary"></div><div class="bays" id="bays"></div>
<script>
const token = new URLSearchParams(location.search).get("token");
const q = token ? "?token=" + encodeURIComponent(token) : "";
const amp = token ? "&token=" + encodeURIComponent(token) : "";
function esc(s){return String(s ?? "").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function refresh(){
  try{
    const r = await fetch("/api/v1/status" + q); const s = await r.json();
    document.getElementById("summary").textContent =
      `device ${s.device_id} · queue ${JSON.stringify(s.outbox)} · webhook ${s.webhook.last_error ? "error: " + s.webhook.last_error : "ok"}`;
    const root = document.getElementById("bays");
    for (const b of s.bays){
      let el = document.getElementById("bay-" + b.id);
      if(!el){el = document.createElement("div"); el.className = "bay"; el.id = "bay-" + b.id;
        el.innerHTML = '<img alt="camera"><div class="info"></div>'; root.appendChild(el);}
      el.querySelector("img").src = `/api/v1/bays/${encodeURIComponent(b.id)}/snapshot.jpg?t=${Date.now()}${amp}`;
      const reads = b.last_reads.map(x => `${esc(x.text)} ${(x.confidence*100).toFixed(0)}% ${x.width_px}px`).join(", ");
      const ev = b.last_event ? `${esc(b.last_event.event_type)} ${esc(b.last_event.plate_display || "")} ${esc(b.last_event.timestamp)}` : "none";
      el.querySelector(".info").innerHTML =
        `<div><b>${esc(b.name)}</b> · ${esc(b.mode)} · ${esc(b.state)}</div>
         <div class="plate">${esc(b.plate_display || "—")}</div>
         <div class="muted ${b.camera.status === "ok" ? "" : "bad"}">camera ${esc(b.camera.status)} ${esc(b.camera.error || "")}</div>
         <div class="muted">reads: ${reads || "none"}</div><div class="muted">last event: ${ev}</div>`;
    }
  }catch(e){document.getElementById("summary").textContent = "error: " + e;}
}
refresh(); setInterval(refresh, 2000);
</script></body></html>
"""
