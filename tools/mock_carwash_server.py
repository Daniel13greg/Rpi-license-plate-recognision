#!/usr/bin/env python3
"""A stand-in for the car wash system: receives plate events and "applies" accounts.

Use it to test a Raspberry Pi before the real system is connected, and as a reference
for implementing the receiving side (see docs/INTEGRATION.md). Standard library only.

    python3 tools/mock_carwash_server.py --port 9000 --plates tools/registered_plates.json \
        --token my-token --hmac-secret my-secret

Point the Pi at it with  webhook.url: http://<this-computer>:9000/api/lpr/events
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CYRILLIC = str.maketrans("АВЕКМНОРСТУХІ", "ABEKMHOPCTYXI")


def normalize(plate: str) -> str:
    """The same canonical form the Pi sends: A-Z and 0-9 only ("c ab-123" -> "CAB123")."""
    text = unicodedata.normalize("NFKD", plate.upper())
    text = "".join(c for c in text if not unicodedata.combining(c)).translate(_CYRILLIC)
    return re.sub(r"[^A-Z0-9]", "", text)


class State:
    def __init__(self, accounts: dict[str, dict], token: str, secret: str, max_skew: int):
        self.accounts = {normalize(plate): account for plate, account in accounts.items()}
        self.token = token
        self.secret = secret
        self.max_skew = max_skew
        self.seen_events: set[str] = set()
        self.active: dict[str, str] = {}  # bay -> plate with an active session


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status: int, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if state.token and self.headers.get("Authorization") != f"Bearer {state.token}":
                return self.reply(401, {"error": "bad token"})
            if state.secret:
                timestamp = self.headers.get("X-LPR-Timestamp", "")
                expected = "sha256=" + hmac.new(state.secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(expected, self.headers.get("X-LPR-Signature", "")):
                    return self.reply(401, {"error": "bad signature"})
                if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > state.max_skew:
                    return self.reply(401, {"error": "timestamp too old (check the clocks)"})
            event = json.loads(body)
            event_id = event.get("event_id")
            if event_id in state.seen_events:  # a retry of something already handled
                return self.reply(200, {"status": "duplicate"})
            state.seen_events.add(event_id)

            kind, bay = event.get("event_type"), event.get("bay_id")
            plate = normalize(event.get("plate") or "")
            stamp = event.get("timestamp", "")
            if kind == "plate_recognized":
                account = state.accounts.get(plate)
                if account is None:
                    print(f"{stamp} bay {bay}: {event.get('plate_display')} is not registered -> pay at the terminal")
                    return self.reply(200, {"status": "unknown_plate"})
                if state.active.get(bay) == plate:
                    print(f"{stamp} bay {bay}: {plate} already has an active session")
                    return self.reply(200, {"status": "already_active"})
                state.active[bay] = plate
                print(f"{stamp} bay {bay}: {event.get('plate_display')} -> account {account.get('name')} "
                      f"(balance {account.get('balance')}) APPLIED  [confidence {event.get('confidence')}]")
                return self.reply(200, {"status": "applied", "account": account.get("id")})
            if kind == "plate_unrecognized":
                print(f"{stamp} bay {bay}: a car is there but the plate could not be read -> manual payment")
            elif kind == "vehicle_left":
                if state.active.get(bay) == plate:
                    del state.active[bay]
                print(f"{stamp} bay {bay}: vehicle {event.get('plate_display') or ''} left")
            else:
                print(f"{stamp} bay {bay}: {kind} {event.get('plate_display') or ''}")
            return self.reply(200, {"status": "ok"})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--plates", help='JSON file: {"KCA123": {"id": 1, "name": "Ion", "balance": 150}, ...}')
    parser.add_argument("--token", default="", help="expected bearer token")
    parser.add_argument("--hmac-secret", default="", help="expected HMAC secret")
    parser.add_argument("--max-skew", type=int, default=300, help="accepted clock difference in seconds")
    args = parser.parse_args()
    accounts = {}
    if args.plates:
        with open(args.plates, encoding="utf-8") as f:
            accounts = json.load(f)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(State(accounts, args.token, args.hmac_secret, args.max_skew)))
    print(f"mock car wash system on http://{args.host}:{args.port} with {len(accounts)} registered plates")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
