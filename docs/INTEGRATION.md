# Integrating with the car wash system

This guide is for the developers of the system that keeps customer accounts and
controls the bays. The Raspberry Pi tells that system which plate is in which bay, in
one or both of these ways:

* **Push (webhook):** the Pi sends an HTTP request to your endpoint for every event.
  This is the recommended way.
* **Pull (REST API):** your system asks the Pi for the plate in a bay, for example when
  the customer presses *Start* on the bay terminal.

## 1. Webhook requests

```
POST https://carwash.example.md/api/lpr/events       (webhook.url / webhook.method)
Content-Type: application/json; charset=utf-8
User-Agent: carwash-lpr/1.0.0
Idempotency-Key: 4a7cf106168b4dd68935059d5ea4237d      same value as event_id
X-LPR-Event-Id: 4a7cf106168b4dd68935059d5ea4237d
Authorization: Bearer <CARWASH_API_TOKEN>              if webhook.bearer_token is set
X-LPR-Timestamp: 1791198187                             if webhook.hmac_secret is set
X-LPR-Signature: sha256=5d0c...                         if webhook.hmac_secret is set
```

### Body

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | `carwash-lpr/1` |
| `event_id` | string | Unique per event; retries reuse it. Deduplicate on it. |
| `event_type` | string | `plate_recognized`, `plate_unrecognized`, `vehicle_left` or `test` |
| `timestamp` | string | ISO 8601 local time with offset, e.g. `2026-10-05T14:03:07.123+03:00` |
| `device_id` / `site_id` | string / null | Which Pi and which car wash (from the Pi's config) |
| `bay_id` / `bay_name` | string | The bay, as configured on the Pi (`"2"`, `"Box 2"`) |
| `trigger` | string | What started the read: `continuous`, `gpio` (sensor), `http` (pull API), `simulated`, `test` |
| `plate` | string / null | Canonical plate: `A-Z0-9` only, no spaces, e.g. `KCA123`, `BLAB123` |
| `plate_display` | string / null | For screens: `KCA 123`, `BL AB 123` |
| `plate_format` | string / null | `md_standard`, `md_short`, `md_regional`, `md_transnistria`, `md_police`, `md_military`, `md_diplomatic`, `md_temporary`, `md_president`, `md_four_letter`, `ro`, `ro_bucharest`, `ua`, `unknown` |
| `country` | string / null | `MD`, `RO`, `UA`, or empty when unknown |
| `confidence` | number / null | 0–1, average OCR confidence over the frames that agreed |
| `votes` | integer | How many frames agreed on the plate |
| `candidates` | array | Up to 5 plates seen in the read window, best first: `{plate, plate_display, votes, confidence}` |
| `ocr_region` | string / null | Country the OCR model thinks the plate is from (informative only) |
| `duration_seconds` | number | Only in `vehicle_left`: how long the car was there |
| `images` | object | Only if `webhook.include_images`: `plate_jpeg_base64`, `frame_jpeg_base64` |

### Event types

* **`plate_recognized`**: a car is in the bay and its plate is known. Sent once per
  visit.
* **`plate_unrecognized`**: trigger mode only. A car is there (sensor or pull request)
  but no plate could be read within `window_seconds`. `plate` is null; `candidates` may
  hold weak guesses. Use it to show "please pay at the terminal".
* **`vehicle_left`**: the car has gone. This is reliable in trigger mode with a sensor;
  in continuous mode it is best effort and off by default. `plate` may be null if the
  car was never recognised.
* **`test`**: sent by `carwash-lpr send-test` or the simulate endpoint. Never charge
  anything on it.

### Your response

| Your HTTP status | What the Pi does |
|---|---|
| 2xx | Delivered. The body is stored in the Pi's event log (useful for debugging). |
| 408, 425, 429, 5xx, timeout, connection error | Retried with backoff 1, 2, 4, 8, 16, 30 s... (`Retry-After` is honoured, up to 60 s) |
| any other 4xx | Rejected; not retried. Logged as an error on the Pi. |

* Answer within `webhook.timeout_seconds` (5 s); do slow work asynchronously.
* An unregistered plate is a normal outcome. Answer 200 (e.g. `{"status":"unknown_plate"}`),
  not 404.
* Events not delivered within `max_event_age_seconds` (default 120 s) are dropped, not
  delivered late. A "car entered bay 2" message arriving ten minutes later could start
  a session for whoever is in bay 2 by then.
* Events from one bay arrive in order. An event never overtakes an older event from the
  same bay.

## 2. Verifying requests

Use HTTPS. Check the bearer token, and if `hmac_secret` is configured, the signature:

```
X-LPR-Signature = "sha256=" + hex( HMAC-SHA256( secret, X-LPR-Timestamp + "." + raw_body ) )
```

Compute it over the **raw request bytes**, not over re-serialised JSON. Reject
timestamps more than a few minutes off; that stops replayed requests, and it needs the
Pi's clock to be synchronised (NTP).

**Python**

```python
import hashlib, hmac, time

def verify(headers, raw_body: bytes, secret: str, max_skew: int = 300) -> bool:
    timestamp = headers.get("X-LPR-Timestamp", "")
    if not timestamp.isdigit() or abs(time.time() - int(timestamp)) > max_skew:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, headers.get("X-LPR-Signature", ""))
```

**PHP**

```php
function lpr_verify(string $rawBody, string $secret, int $maxSkew = 300): bool {
    $timestamp = $_SERVER['HTTP_X_LPR_TIMESTAMP'] ?? '';
    if (!ctype_digit($timestamp) || abs(time() - (int)$timestamp) > $maxSkew) {
        return false;
    }
    $expected = 'sha256=' . hash_hmac('sha256', $timestamp . '.' . $rawBody, $secret);
    return hash_equals($expected, $_SERVER['HTTP_X_LPR_SIGNATURE'] ?? '');
}

$raw = file_get_contents('php://input');
if (!lpr_verify($raw, getenv('LPR_HMAC_SECRET'))) { http_response_code(401); exit; }
$event = json_decode($raw, true);
```

**Node.js (Express)**

```js
const crypto = require("crypto");

function verify(req, secret, maxSkew = 300) {
  const ts = req.get("X-LPR-Timestamp") || "";
  if (!/^\d+$/.test(ts) || Math.abs(Date.now() / 1000 - Number(ts)) > maxSkew) return false;
  const expected = "sha256=" + crypto.createHmac("sha256", secret).update(ts + ".").update(req.body).digest("hex");
  const given = req.get("X-LPR-Signature") || "";
  return given.length === expected.length && crypto.timingSafeEqual(Buffer.from(given), Buffer.from(expected));
}

app.post("/api/lpr/events", express.raw({ type: "application/json" }), (req, res) => {
  if (!verify(req, process.env.LPR_HMAC_SECRET)) return res.sendStatus(401);
  const event = JSON.parse(req.body);
  // ...
  res.json({ status: "ok" });
});
```

## 3. Matching plates to accounts

The Pi always sends the canonical form: upper case, Latin letters and digits only. Store
registered plates in the same form, so that `"c ab 123"`, `"CAB-123"` and `"САВ123"`
(Cyrillic) all become `CAB123`:

```python
import re, unicodedata
CYRILLIC = str.maketrans("АВЕКМНОРСТУХІ", "ABEKMHOPCTYXI")

def normalize_plate(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.upper())
    text = "".join(c for c in text if not unicodedata.combining(c)).translate(CYRILLIC)
    return re.sub(r"[^A-Z0-9]", "", text)
```

```php
function normalize_plate(string $p): string {
    $p = strtr(mb_strtoupper($p, 'UTF-8'), ['А'=>'A','В'=>'B','Е'=>'E','К'=>'K','М'=>'M','Н'=>'H','О'=>'O',
        'Р'=>'P','С'=>'C','Т'=>'T','У'=>'Y','Х'=>'X','І'=>'I','Ș'=>'S','Ş'=>'S','Ț'=>'T','Ţ'=>'T','Ă'=>'A','Â'=>'A','Î'=>'I']);
    return preg_replace('/[^A-Z0-9]/', '', $p);
}
```

```js
const CYR = { "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I" };
const normalizePlate = (s) =>
  s.toUpperCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/./g, (c) => CYR[c] || c).replace(/[^A-Z0-9]/g, "");
```

The pre-2015 Chișinău plate `C AB 123` and a current `CAB 123` are the same canonical
`CAB123`. Spaces never matter.

## 4. Recommended logic on your side

1. **Deduplicate** on `event_id`. A retried request carries the same id.
2. On `plate_recognized`, look the plate up by **exact match only**. Never by
   similarity: dirt or foam on a plate can turn `VYH 698` into a different valid number,
   and a "nearest registered plate" search would then charge someone else.
3. If the plate is registered and that bay has no active session, start one for the
   account, and **show the plate and the customer's name on the bay terminal** so
   the driver can cancel. You may also require `confidence >= 0.9` before charging
   without a confirmation tap.
4. If the same plate already has an active session in that bay, ignore the event (for
   example, a car that stood in the bay longer than `repeat_cooldown_seconds` with its
   plate hidden by foam).
5. If the plate is not registered, or on `plate_unrecognized`, the terminal falls back
   to normal payment.
6. On `vehicle_left`, end the session if your process needs that.
7. Keep the raw events for a while. If a customer disputes a charge, the Pi's snapshot of
   that `event_id` is in `/var/lib/carwash-lpr/snapshots/<date>/<bay>/` (kept for
   `snapshot_retention_days`).

`tools/mock_carwash_server.py` implements this logic in about 150 lines of standard-library
Python and is a usable reference.

## 5. Adapting to an existing API

If your system already has an endpoint, the Pi can shape its request to fit, without
any code on your side:

```yaml
webhook:
  url: "https://pos.example.md/api/boxes/{bay_id}/vehicle"   # placeholders work in the URL
  method: POST
  headers: {X-Api-Key: "${POS_API_KEY}"}
  event_types: [plate_recognized]           # only send what the API understands
  payload_template:
    licensePlate: "{plate}"
    box: "{bay_id}"
    detectedAt: "{timestamp}"
    note: "LPR {plate_display} ({confidence})"
```

Placeholders are the body fields from section 1. A value that is exactly one placeholder
keeps its JSON type (number, null, array); placeholders inside longer text are rendered
as text, with null as an empty string. An unknown placeholder stops the service at
start-up with an error that lists the valid names.

## 6. Pull API on the Pi

Every endpoint except `/health` needs `Authorization: Bearer <LPR_API_TOKEN>` (see
`/etc/carwash-lpr/env` on the Pi) or `?token=...`. The port is 8080 by default.

**Ask for the plate in a bay**: waits up to `wait` seconds (0–60, default 10):

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" "http://carwash-pi-01:8080/api/v1/bays/2/read?wait=10"
```
```json
{"bay_id": "2", "status": "recognized", "plate": "KCA123", "plate_display": "KCA 123",
 "confidence": 0.97, "event_id": "4a7cf106168b4dd68935059d5ea4237d", "candidates": [...]}
```

`status` is `recognized`, `unrecognized` (the read window ended without a plate),
`timeout` (`wait` ran out) or `pending` (with `wait=0`: the read was started and its
result will come as a webhook event). In continuous mode, a car whose plate was
seen in the last 10 s is answered immediately. In trigger mode, the call opens a read
window like the sensor does, and produces the usual webhook event. Its `event_id` is in
the response, so the two can be matched.

| Endpoint | Purpose |
|---|---|
| `GET /health` | `{"status": "ok"}` (HTTP 200) or `degraded` (HTTP 503, e.g. a camera is down). No plates, no token needed. |
| `GET /api/v1/status` | Everything: bays, cameras, delivery queue, last webhook error |
| `GET /api/v1/bays/{id}` | Bay state (`empty`/`occupied` or `idle`/`reading`/`done`), current plate, last reads, last event |
| `GET /api/v1/bays/{id}/snapshot.jpg` | Current camera image with the search area and plates drawn (`?annotate=0` for the raw image) |
| `GET /api/v1/events?limit=50&bay=2` | Recent events with delivery status, HTTP code and your response body |
| `POST /api/v1/bays/{id}/simulate` | Inject an event to test your side: `{"plate": "KCA 123", "event_type": "plate_recognized"}` |

## 7. Testing the integration

* From the Pi: `sudo carwash-lpr send-test --plate "KCA 123" --type plate_recognized` sends one
  event with the real headers and signature, and prints your response.
* Without access to the Pi's shell:
  `curl -X POST -H "Authorization: Bearer $TOKEN" -d '{"plate":"KCA 123"}' http://<pi>:8080/api/v1/bays/1/simulate`
* Without your system: run `python3 tools/mock_carwash_server.py --plates tools/registered_plates.json`
  and point `webhook.url` at it.
