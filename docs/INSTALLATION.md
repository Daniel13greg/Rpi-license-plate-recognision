# Installation and hardware guide

## 1. Hardware

| Part | Recommendation |
|---|---|
| Computer | **Raspberry Pi 5** (4 or 8 GB) with the active cooler: two camera connectors, room for several bays. A Pi 4 (2 GB+) works for one or two bays (consider the faster `yolo-v9-t-384` detector). Recognition runs continuously, so without cooling the CPU throttles and slows down. |
| OS | Raspberry Pi OS **64-bit** (Bookworm or newer). Lite is enough. 32-bit systems are not supported (no ONNX Runtime). |
| Storage | High-endurance microSD (32 GB+) or an NVMe SSD on a Pi 5. The event database and snapshots are written all day. |
| Power | The official power supply (27 W for the Pi 5). A UPS HAT protects the SD card from power cuts. |
| Network | Ethernet if possible. Give the Pi a fixed address (DHCP reservation). |
| Camera | See below. One camera per bay. |
| Light | An 850 nm infrared illuminator for bays that are dark or used at night. Plates are retro-reflective and light up under IR. |
| Sensor (optional) | For trigger mode: the relay output of an induction loop detector, or a photo-electric (light barrier) sensor with relay output. |

**Choosing a camera**

* **IP camera (RTSP)**: the usual choice for wash bays. Pick an IP66/IP67 housing with
  built-in IR and PoE, ideally a varifocal lens (2.8–12 mm) or an "LPR" model, so the
  plate fills enough of the image. The Pi can sit in the technical room, connected over
  the network.
* **Raspberry Pi Camera Module 3** (or the NoIR version plus an IR illuminator): the
  cheapest option. It needs a waterproof housing, and the Pi must be close to it (camera
  cables are short). The Pi 5 takes two camera modules.
* **USB webcam**: fine for testing on a desk, not for a wet bay.

Never put the Pi, a camera without housing, or the cables where the spray reaches.

## 2. Where to put the camera

This matters more than any software setting.

* **Read the plate as the car drives in**, before water and foam cover it. A good spot is
  high on the back wall of the bay (2–2.5 m), looking at the entrance and the front plate
  of the incoming car. Moldovan cars carry front and rear plates. If cars reverse into
  your bays, watch the rear plate instead.
* **Plate size: 100–200 pixels wide** in the image (60 px minimum). Plate width in pixels
  ≈ image width × 0.52 m ÷ width of the scene at the plate's distance. Example: a Camera
  Module 3 (66° horizontal field of view) at 4 m sees a 5.2 m wide scene, so a 1920 px
  image shows the plate about 190 px wide. The Wide model (102°) would give about 100 px.
* **Angle:** keep the plate within about 30° of straight-on, both sideways and up/down.
* **No glare:** don't point the camera at the open gate if the sun shines straight in,
  or at bright lamps.
* **Focus** at the reading distance. For Camera Module 3: `af_mode: manual` and
  `lens_position: 0.25` for 4 m (dioptres = 1 ÷ metres). Continuous autofocus can hunt
  in spray.
* **Exposure:** a short shutter (1/500 s or faster, e.g. `exposure_time_us: 2000`)
  avoids motion blur, but needs enough light, so use IR.
* **Search area (ROI):** limit `roi` to where plates appear inside the bay. Exclude the
  street, the queue in front of the gate and the neighbouring bay.

## 3. Prepare the Raspberry Pi

1. In Raspberry Pi Imager choose *Raspberry Pi OS Lite (64-bit)*. In its settings, set a
   hostname (e.g. `carwash-pi-01`), a user, SSH and the time zone `Europe/Chisinau`.
2. Boot, log in over SSH, and update: `sudo apt update && sudo apt full-upgrade -y`
3. Camera module check: `rpicam-hello --list-cameras`, then `rpicam-still -o test.jpg`
4. Clock check: `timedatectl` must say `System clock synchronized: yes`. The Pi needs
   network time for correct event timestamps and signatures. A Pi 4 has no battery-backed
   clock; a Pi 5 has one but needs its optional battery.

## 4. Install

```bash
sudo apt install -y git
git clone https://github.com/Daniel13greg/Rpi-license-plate-recognision.git
cd Rpi-license-plate-recognision
sudo ./deploy/install.sh
```

The installer:

* installs Python, Picamera2, `rpicam-apps` and GPIO libraries from apt;
* creates the system user `carwash-lpr` (groups `video`, `gpio`);
* builds `/opt/carwash-lpr/venv`, keeping the system's numpy so Picamera2 keeps working;
* writes `/etc/carwash-lpr/config.yaml` (from the example) and `/etc/carwash-lpr/env`
  (secrets, with a random API token);
* downloads the models to `/var/lib/carwash-lpr/models`;
* installs and starts the `carwash-lpr` systemd service.

To update later, run `git pull && sudo ./deploy/install.sh`. Configuration and data are
kept.

## 5. Configure

Secrets go in `/etc/carwash-lpr/env`, everything else in
`/etc/carwash-lpr/config.yaml` (every option is explained in the
[example](../config/config.example.yaml)). After changes:

```bash
sudo carwash-lpr check-config && sudo systemctl restart carwash-lpr
journalctl -u carwash-lpr -f
```

### Camera types

```yaml
# Raspberry Pi camera module (second module on a Pi 5: index 1)
camera: {type: picamera2, index: 0, width: 1920, height: 1080, fps: 10, af_mode: manual, lens_position: 0.25}

# The same through rpicam-vid; use this if "import picamera2" fails
camera: {type: rpicam, index: 0, width: 1920, height: 1080, fps: 10}

# IP camera; keep the password in /etc/carwash-lpr/env as CAM1_PASSWORD
camera: {type: opencv, url: "rtsp://admin:${CAM1_PASSWORD}@192.168.1.64:554/Streaming/Channels/101"}

# USB webcam
camera: {type: opencv, url: /dev/video0, width: 1280, height: 720, fps: 10}
```

Typical RTSP addresses (main stream; check your camera's manual):

* Hikvision: `rtsp://user:pass@IP:554/Streaming/Channels/101` (`102` = sub-stream)
* Dahua: `rtsp://user:pass@IP:554/cam/realmonitor?channel=1&subtype=0`
* Others: see the camera's ONVIF / RTSP settings. Test the URL in VLC first.

Use the camera's main stream (1080p) unless plates are already 150 px wide or more in the
sub-stream.

### Aim with the status page

Open `http://<pi-address>:8080/?token=<LPR_API_TOKEN>`. Every 2 seconds it shows each
bay's camera image with the search area (orange) and every detected plate (green)
labelled with the read, its confidence and its **width in pixels**. Drive a car in
slowly and adjust the camera, zoom, focus and `roi` until the plate reads well and is at
least 100 px wide where the car is first fully inside the search area.

## 6. Presence sensor (trigger mode)

The Pi's GPIO pins take **3.3 V at most**. A 12/24 V sensor output connected directly
destroys the pin. Use one of these:

* **Relay (dry) contact**, as found on loop detectors and many light barriers: one wire to
  GPIO17 (physical pin 11), the other to GND (physical pin 9). Contact closed = car present:

  ```yaml
  mode: trigger
  trigger: {gpio_pin: 17, pull_up: true, active_high: false}
  ```

* **12/24 V NPN/PNP outputs**: go through an optocoupler module or a relay, then wire it
  as above.

Check the wiring with `pinctrl get 17`: the level must change when a car (or your hand)
triggers the sensor. Tune `activate_delay_seconds` (ignores blips) and
`release_delay_seconds` (gaps between wheels or people walking by) to your sensor.

Without a sensor, trigger mode still works through the API: the bay terminal calls
`POST /api/v1/bays/{id}/read` when the customer presses *Start*.

## 7. Several bays on one Pi

One Raspberry Pi 5 can serve 4–5 bays:

* **Use IP cameras.** A Pi 5 has two camera connectors (a Pi 4 has one) and their cables
  are short. Put one PoE IP camera in each bay, cabled to a PoE switch next to the Pi,
  and add one entry per bay under `bays:`.
* **Set each camera's stream to H.264, 8–10 frames per second**, 1080p (or 720p if
  plates are still at least 120 px wide). The Pi 5 has no hardware H.264 decoder, so
  decoding the streams costs CPU in proportion to their frame rate; 25 fps would waste
  most of it.
* **Keep the CPU savers on:** motion gating, a tight `roi` per bay, `process_fps` of 2–4,
  and `presence.recheck_interval_seconds: 2`. With rechecking, a car that has already been
  identified is only re-read every 2 seconds, even while it is being washed (spray looks
  like motion). Trigger mode with presence sensors uses the least CPU of all.
* All bays share one plate reader and take turns in arrival order, so when several cars
  arrive at once each bay gets an equal share.

**Measure before buying everything.** On the Pi:

```bash
sudo carwash-lpr benchmark --bays 5     # the plate reader alone
sudo systemctl stop carwash-lpr
sudo carwash-lpr benchmark --cameras    # the configured cameras, decoding included
sudo systemctl start carwash-lpr
```

The report shows how many plate reads per second each bay gets when all bays are busy at
the same moment, and how long identifying a car takes. Two or more reads per second per
bay is good: a car is identified about a second after its plate becomes readable. With a
single camera bought so far, point all five test bays at the same camera URL: most IP
cameras serve several streams at once, so this measures the full decoding load.

If it is too slow, use the faster `yolo-v9-t-384-license-plate-end2end` detector, lower
the camera frame rates, or split the bays over two Pis. Two Pis also mean that one
failure only sends some bays back to manual payment. Watch the temperature with
`vcgencmd measure_temp`: above about 80 °C the Pi slows itself down, so use the active
cooler.

## 8. Go-live checklist

- [ ] `sudo carwash-lpr check-config` passes; `api.token` is set; the webhook URL is `https://`.
- [ ] `sudo carwash-lpr send-test --plate "KCA 123"` is delivered (HTTP 2xx).
- [ ] `timedatectl` shows a synchronised clock.
- [ ] 10 or more test drives per bay with different cars. Check `/api/v1/events` and the
      snapshots in `/var/lib/carwash-lpr/snapshots`.
- [ ] A test in the dark (IR) and one with a dirty or wet car.
- [ ] Unplug the network for 30 s during a test drive: the event arrives after reconnecting.
- [ ] `sudo reboot`: the service comes back by itself (`systemctl status carwash-lpr`).
- [ ] The car wash system matches plates exactly and shows the plate on the terminal.
- [ ] Signs at the bays say that cameras read licence plates; retention is configured.

## 9. Troubleshooting

| Symptom | What to check |
|---|---|
| Log says `picamera2 is not available` | `sudo apt install python3-picamera2`, re-run the installer, or switch to `type: rpicam` |
| Camera `reconnecting` (IP camera) | URL, user and password (try them in VLC), network, `rtsp_transport: udp` |
| No plates at all | Status page: is the plate inside the orange area, and wider than `min_plate_width`? Set `log_level: DEBUG` to log every read. |
| Plates read wrongly | Focus, motion blur (shorter exposure, more light), angle, dirty lens. Raise `voting.min_reads` to 3. |
| Plate read but no event | Its category is not in `plates.accept`, or `voting.min_confidence` is too strict for the camera |
| Same car reported twice | Increase `presence.repeat_cooldown_seconds`, or use trigger mode with a sensor |
| Events not delivered | `curl -H "Authorization: Bearer $TOKEN" localhost:8080/api/v1/events` shows `last_error` and your server's response. 401 means the token or HMAC secret differ; `expired` means the server was unreachable for longer than `max_event_age_seconds`. |
| High CPU or temperature | Lower `process_fps`, keep `motion.enabled`, tighten `roi`, use the 384 detector, add cooling |
| Service restarts by itself | `journalctl -u carwash-lpr` shows the reason; the watchdog restarts the service if a camera or thread hangs |
