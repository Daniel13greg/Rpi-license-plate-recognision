# Car wash licence plate recognition for Raspberry Pi (Moldova)

A Raspberry Pi with a camera watches a car wash bay, reads the Moldovan licence plate of
the car that drives in, and sends it to the car wash system. The car wash system looks
the plate up among its registered customers and applies that customer's account to the
bay.

```
 camera ──► Raspberry Pi ─────────────────────────────────────────►  car wash system
            plate detector (YOLOv9) → OCR → Moldovan plate rules       POST /your/endpoint
            → several frames must agree → "car arrived in bay 2"       {"bay_id": "2",
            → stored locally, delivered with retries (HTTPS + HMAC)     "plate": "KCA123", ...}
                                                                        → registered? apply account
```

* **Moldovan plates.** Current `ABC 123` plates (including green EV plates and RMG/RMP
  government plates), personalised `ION 7`, pre-2015 district plates `BL AB 123` /
  `C AB 123`, Transnistrian-region `A 123 BC`, and special series (`MAI 1234`, `FA 1234`,
  `RM 0001`, `CD 123 AB`, `H 1234`). Romanian and Ukrainian plates, which are common in
  Moldova, are also understood. Classic OCR mix-ups (0/O, 1/I, 8/B, 5/S...) are fixed
  only where the plate layout requires a letter or a digit.
* **Reliable reads.** A plate is reported only when several frames agree. Each car is
  reported once, not every second it stands in the bay.
* **Two ways to know a car is there.** With no sensor, the plate alone decides. In
  trigger mode, a presence sensor (induction loop, light barrier) on a GPIO pin or an
  HTTP call opens the read window. If no plate can be read, `plate_unrecognized` is
  sent so the terminal can ask for normal payment.
* **No lost messages.** Events go to a local SQLite queue first and are retried until
  delivered. They are dropped after 2 minutes, because a late "car entered" message
  could charge the wrong car.
* **Easy to integrate.** The Pi can push a JSON webhook (signed, idempotent, with a
  template to match an existing API). The car wash system can also ask the Pi through
  its REST API: `POST /api/v1/bays/2/read`.
* **Unattended operation.** Runs as a systemd service with a watchdog, reconnects lost
  cameras, keeps short-lived JPEG snapshots of each event, and has a status page for
  aiming the camera.
* **Cameras.** Raspberry Pi Camera Module (Picamera2 or `rpicam-vid`), IP cameras
  (RTSP), USB webcams. One Pi can serve several bays.

The neural networks are [open-image-models](https://github.com/ankandrew/open-image-models)
(plate detector) and [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr)
(OCR, trained on plates from 65+ countries including Moldova). Both run on the Pi's CPU
with ONNX Runtime. No cloud service is needed.

## Quick start

On a Raspberry Pi 4 or 5 running **Raspberry Pi OS 64-bit** (Bookworm or newer):

```bash
sudo apt install -y git
git clone https://github.com/Daniel13greg/Rpi-license-plate-recognision.git
cd Rpi-license-plate-recognision
sudo ./deploy/install.sh
```

The installer sets up a virtualenv in `/opt/carwash-lpr`, the configuration in
`/etc/carwash-lpr`, data in `/var/lib/carwash-lpr`, downloads the models (~11 MB), and
starts the `carwash-lpr` service. Then:

1. Put the car wash system's endpoint and secrets in `/etc/carwash-lpr/env`:
   ```
   CARWASH_WEBHOOK_URL=https://carwash.example.md/api/lpr/events
   CARWASH_API_TOKEN=...
   CARWASH_HMAC_SECRET=...
   ```
2. Describe the bays and cameras in `/etc/carwash-lpr/config.yaml`. The
   [example](config/config.example.yaml) documents every option.
3. `carwash-lpr check-config && sudo systemctl restart carwash-lpr`
4. Open `http://<pi-address>:8080/?token=<LPR_API_TOKEN from the env file>` and aim the
   camera. The live image shows the search area and each plate with its read and its
   width in pixels.
5. Test the link to the car wash system: `carwash-lpr send-test --plate "KCA 123"`.

[docs/INSTALLATION.md](docs/INSTALLATION.md) covers hardware, camera placement, IP
cameras, sensor wiring and a go-live checklist. [docs/INTEGRATION.md](docs/INTEGRATION.md)
is for the developers of the car wash system.

### Try it without a camera

```bash
carwash-lpr demo-images /tmp/demo --plates "KCA 123" "BL AB 123" "ION 7"
python3 tools/mock_carwash_server.py --port 9000 --plates tools/registered_plates.json
```

Then set a bay camera to `{type: images, path: /tmp/demo, fps: 6, loop: false}` and
`webhook.url: http://127.0.0.1:9000/api/lpr/events`, and run `carwash-lpr run -c <config>`.
The mock server prints which account it would apply:

```
bay 1: KCA 123 -> account Ion Popescu (balance 150 MDL) APPLIED  [confidence 1.0]
bay 1: BL AB 123 -> account Maria Rusu (balance 80 MDL) APPLIED  [confidence 1.0]
```

Check a photo directly with `carwash-lpr recognize photo.jpg --annotate out/`.

## What gets sent

```json
{
  "schema": "carwash-lpr/1",
  "event_id": "4a7cf106168b4dd68935059d5ea4237d",
  "event_type": "plate_recognized",
  "timestamp": "2026-10-05T14:03:07.123+03:00",
  "device_id": "carwash-pi-01",
  "site_id": "chisinau-botanica",
  "bay_id": "2",
  "bay_name": "Box 2",
  "trigger": "continuous",
  "plate": "KCA123",
  "plate_display": "KCA 123",
  "plate_format": "md_standard",
  "country": "MD",
  "confidence": 0.97,
  "votes": 4,
  "candidates": [{"plate": "KCA123", "plate_display": "KCA 123", "votes": 4, "confidence": 0.97}],
  "ocr_region": "Moldova"
}
```

`plate` is always canonical: upper case `A-Z0-9` with no spaces (`"c ab-123"` →
`CAB123`). The car wash system should store registered plates the same way.
Event types: `plate_recognized`, `plate_unrecognized` (a car is there but its plate
could not be read; trigger mode), `vehicle_left` (optional), and `test`. Headers,
signature, retries and the pull API are covered in [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Modes

| | `continuous` (default) | `trigger` |
|---|---|---|
| Knows a car is there from | its plate being read | a GPIO presence sensor or `POST /api/v1/bays/{id}/read` |
| Reports | `plate_recognized` once per visit | `plate_recognized`, or `plate_unrecognized` after `window_seconds` |
| Car left | plate unseen for `absence_timeout_seconds` (best effort, off by default) | sensor released (`vehicle_left`) |
| CPU | recognition while something moves | recognition only during the read window |

In continuous mode the same plate is not reported again within `repeat_cooldown_seconds`
(5 min), counted from when it was last seen. A car standing in the bay is reported once,
even when foam hides its plate for a while.

## Accuracy: what to expect

Tests in this repository:

* Synthetic Moldovan plates (standard, district, personalised, special, green EV) are read
  correctly by the full pipeline, from camera frames through to the webhook.
* A real photo of a Moldovan plate (`VYH 698`, rear of a car in daylight) was read with
  confidence 1.00 and the OCR's own country guess "Moldova". Degraded copies were still
  read correctly when the plate was only 50 px wide, very dark and noisy, blurred, heavily
  JPEG-compressed, or tilted by 10°.
* **Failure modes.** Strong motion blur produced a low-confidence misread, which the
  confidence threshold rejects. Opaque drops painted over the digits produced
  *confident* wrong reads (`VHG 98`, `VYH 600`). Dirt, foam or snow covering characters
  is therefore the real risk. Multi-frame voting helps when water moves, but not when mud
  sticks to the plate.

So the car wash system should **apply an account only for an exact match with a
registered plate**, and should show the plate (and customer name) on the bay terminal
before charging. Accuracy depends mostly on the camera: aim for plates ≥ 100 px wide, at
less than 30° to the camera, lit by an IR illuminator in dark bays, and read as the car
drives in, before it gets wet. See [docs/INSTALLATION.md](docs/INSTALLATION.md).

To improve accuracy on site, the snapshots in `/var/lib/carwash-lpr/snapshots` can be
used to fine-tune the OCR with
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr). Point
`recognizer.ocr_model_path` / `ocr_config_path` at the result.

## Operation

| Task | Command |
|---|---|
| Logs | `journalctl -u carwash-lpr -f` (start with `-v` or set `log_level: DEBUG` to see every read) |
| Status page | `http://<pi>:8080/?token=...` |
| Health (for monitoring) | `curl http://<pi>:8080/health`: HTTP 200 or 503 |
| Recent events and delivery status | `curl -H "Authorization: Bearer $TOKEN" http://<pi>:8080/api/v1/events` |
| Validate config | `carwash-lpr check-config` |
| Single camera image (service stopped) | `carwash-lpr snapshot --bay 1 --annotate -o snap.jpg` |
| How a text is interpreted | `carwash-lpr plate "8LAB123"` |
| Update | `git pull && sudo ./deploy/install.sh` (keeps config and data) |
| Uninstall | `sudo ./deploy/uninstall.sh [--purge]` |

**Privacy.** Plates and snapshots are personal data under Moldova's Law 195/2024 on
personal data protection (GDPR-aligned, in force since 23 August 2026). Keep
`snapshot_retention_days` short, tell customers about the cameras (signs at the bays),
set `api.token`, and keep the Pi on a private network.

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest            # the model tests download ~11 MB on first run
```

CI runs the tests on Python 3.11 with numpy 1.24 (as on Raspberry Pi OS Bookworm) and on
Python 3.13 with current libraries.

```
carwash_lpr/
  plates.py      Moldovan (and RO/UA) plate layouts, normalisation, OCR fixes
  recognizer.py  plate detector + OCR (ONNX Runtime)
  voting.py      multi-frame agreement
  bay.py         per-bay logic: continuous and trigger modes, read requests
  camera.py      Picamera2, rpicam-vid, OpenCV (RTSP/USB/file) and image-folder sources
  outbox.py      SQLite event queue      sender.py   webhook delivery (retries, HMAC)
  api.py         local REST API + status page   service.py  wiring, watchdog
  config.py      YAML config + validation       cli.py      command line
deploy/          install.sh, uninstall.sh, systemd unit
tools/           mock car wash server for testing the integration
```
