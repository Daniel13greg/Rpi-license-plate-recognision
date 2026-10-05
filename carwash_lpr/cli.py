"""Command line interface: ``carwash-lpr <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from carwash_lpr import __version__, plates
from carwash_lpr.config import (
    CATEGORIES, Config, ConfigError, PlatesConfig, RecognizerConfig, VotingConfig, load_config, redacted,
)

log = logging.getLogger("carwash_lpr")


def default_config() -> str:
    return os.environ.get("CARWASH_LPR_CONFIG", "/etc/carwash-lpr/config.yaml")


def default_env_file() -> str:
    return os.environ.get("CARWASH_LPR_ENV_FILE", "/etc/carwash-lpr/env")


def load_env_file(path: str) -> bool:
    """Load the service's secrets (systemd EnvironmentFile syntax) without overriding the
    environment, so commands run by hand see the same ${NAME} values as the service."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        hint = "; run with sudo" if isinstance(exc, PermissionError) else ""
        print(f"note: cannot read {path} (secrets for the configuration): {exc}{hint}", file=sys.stderr)
        return False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def setup_logging(level: str = "INFO", verbose: bool = False) -> None:
    # journald already timestamps every line
    fmt = "%(levelname)s %(name)s: %(message)s"
    if "JOURNAL_STREAM" not in os.environ:
        fmt = "%(asctime)s " + fmt
    logging.basicConfig(level=logging.DEBUG if verbose else getattr(logging, level), format=fmt, force=True)
    for noisy in ("urllib3", "open_image_models", "fast_plate_ocr", "picamera2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _pick_bay(cfg: Config, bay_id: str | None):
    if bay_id is None:
        return cfg.bays[0]
    try:
        return cfg.bay(bay_id)
    except KeyError:
        raise ConfigError(f"no bay with id {bay_id!r}") from None


def _models_dir(cfg: Config | None) -> Path:
    if cfg is not None:
        return Path(cfg.data_dir) / "models"
    return Path.home() / ".cache" / "carwash-lpr" / "models"


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    setup_logging(cfg.log_level, args.verbose)
    from carwash_lpr.service import Service

    Service(cfg).run()
    return 0


def cmd_check_config(args) -> int:
    cfg = load_config(args.config)
    print(json.dumps(redacted(cfg), indent=2, ensure_ascii=False))
    if not cfg.webhook.url:
        print("warning: webhook.url is empty, events will only be kept locally", file=sys.stderr)
    if cfg.api.enabled and not cfg.api.token:
        print("warning: api.token is empty, anyone on the network can use the API", file=sys.stderr)
    print(f"{args.config}: OK", file=sys.stderr)
    return 0


def cmd_download_models(args) -> int:
    from carwash_lpr.recognizer import ensure_models

    cfg = load_config(args.config) if args.config else None
    rec = cfg.recognizer if cfg else RecognizerConfig()
    for path in ensure_models(rec, _models_dir(cfg)):
        print(path)
    return 0


def cmd_recognize(args) -> int:
    import cv2

    from carwash_lpr.recognizer import PlateRecognizer, draw_overlay
    from carwash_lpr.voting import observe

    cfg = load_config(args.config) if args.config else None
    rec_cfg = cfg.recognizer if cfg else RecognizerConfig()
    plates_cfg = cfg.plates if cfg else PlatesConfig(accept=list(CATEGORIES))
    recognizer = PlateRecognizer(rec_cfg, _models_dir(cfg))
    exit_code = 0
    for name in args.images:
        frame = cv2.imread(name, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"{name}: cannot read image", file=sys.stderr)
            exit_code = 1
            continue
        started = time.perf_counter()
        reads = recognizer.recognize(frame)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"{name}: {len(reads)} plate(s) in {elapsed:.0f} ms")
        labels = []
        for read in reads:
            found = plates.interpret(read.text)
            accepted = observe(read, plates_cfg, 0.0) is not None
            label = found.display if found else read.text
            labels.append(label)
            details = f"{found.format}, {found.category}, {found.fixes} fix(es)" if found else "not a plate"
            region = f", OCR region {read.region} {read.region_confidence:.0%}" if read.region else ""
            print(
                f"  {label:12} raw={read.text!r} confidence={read.confidence:.2f} min={read.min_char_confidence:.2f} "
                f"width={read.width}px ({details}{region}){'' if accepted else ' [ignored]'}"
            )
        if args.annotate:
            out_dir = Path(args.annotate)
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / Path(name).name
            cv2.imwrite(str(out), draw_overlay(frame, None, reads, labels))
            print(f"  annotated image: {out}")
    return exit_code


def cmd_plate(args) -> int:
    for text in args.text:
        found = plates.candidates(text)
        best = plates.interpret(text)
        if best is None:
            print(f"{text!r}: not a plate")
            continue
        print(f"{text!r} -> {best.plate} ({best.display}), {best.format}, {best.category}, {best.fixes} fix(es)")
        for other in found[1:4]:
            print(f"    also possible: {other.plate} ({other.format}, score {other.score:.3f})")
    return 0


def cmd_send_test(args) -> int:
    from carwash_lpr.events import PlateEvent
    from carwash_lpr.outbox import Outbox
    from carwash_lpr.sender import WebhookSender

    cfg = load_config(args.config)
    if not cfg.webhook.url:
        print("webhook.url is not configured", file=sys.stderr)
        return 2
    bay = _pick_bay(cfg, args.bay)
    found = plates.interpret(args.plate)
    event = PlateEvent(
        event_type=args.type,
        bay_id=bay.id,
        bay_name=bay.name,
        trigger="test",
        plate=found.plate if found else args.plate,
        plate_display=found.display if found else args.plate,
        plate_format=found.format if found else None,
        country=found.country if found else None,
        confidence=1.0,
        votes=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        outbox = Outbox(Path(tmp) / "test.db")
        payload = event.payload(cfg.device_id, cfg.site_id)
        outbox.add(event, payload, deliver=True)
        sender = WebhookSender(cfg.webhook, outbox)
        url, body, _ = sender.build_request(outbox.due(time.time())[0])
        print(f"{cfg.webhook.method} {url}\n{body.decode()}\n")
        sender.deliver_due()
        row = outbox.recent(1)[0]
        outbox.close()
    if row["status"] == "delivered":
        print(f"delivered: HTTP {row['response_code']}\n{row['response_body']}")
        return 0
    print(f"NOT delivered: {row['last_error']}\n{row['response_body'] or ''}", file=sys.stderr)
    return 1


def cmd_snapshot(args) -> int:
    import cv2

    from carwash_lpr.camera import create_source
    from carwash_lpr.recognizer import draw_overlay, roi_pixels

    cfg = load_config(args.config)
    bay = _pick_bay(cfg, args.bay)
    source = create_source(bay.camera)
    source.open()
    try:
        deadline = time.monotonic() + args.warmup
        frame = source.read()
        while time.monotonic() < deadline:  # let exposure and focus settle
            frame = source.read()
    finally:
        source.close()
    roi = roi_pixels(bay.roi, frame.shape[1], frame.shape[0])
    if args.annotate:
        from carwash_lpr.recognizer import PlateRecognizer

        reads = PlateRecognizer(cfg.recognizer, _models_dir(cfg)).recognize(frame, roi)
        labels = []
        for read in reads:
            found = plates.interpret(read.text)
            labels.append(found.display if found else read.text)
            print(f"{labels[-1]} confidence={read.confidence:.2f} width={read.width}px")
        frame = draw_overlay(frame, roi, reads, labels)
    cv2.imwrite(args.output, frame)
    print(f"saved {args.output} ({frame.shape[1]}x{frame.shape[0]})")
    return 0


def cmd_demo_images(args) -> int:
    from carwash_lpr.demo import write_demo_images

    paths = write_demo_images(args.folder, args.plates)
    print(f"wrote {len(paths)} images to {args.folder}")
    print("use them with a bay camera of type 'images' and path set to that folder")
    return 0


def cmd_benchmark(args) -> int:
    from carwash_lpr import benchmark
    from carwash_lpr.camera import CameraError
    from carwash_lpr.recognizer import PlateRecognizer

    cfg = load_config(args.config) if args.config else None
    if args.cameras and cfg is None:
        raise ConfigError("--cameras needs the configuration file (-c)")
    if args.seconds <= 0 or (args.bays is not None and not 1 <= args.bays <= 32):
        raise ConfigError("--seconds must be positive and --bays between 1 and 32")
    rec_cfg = cfg.recognizer if cfg else RecognizerConfig()
    min_reads = cfg.voting.min_reads if cfg else VotingConfig().min_reads
    header = f"Plate reader: {rec_cfg.detector_model} + {rec_cfg.ocr_model}, {rec_cfg.threads} threads"
    print("loading the plate reader...", file=sys.stderr)
    recognizer = PlateRecognizer(rec_cfg, _models_dir(cfg))
    if args.cameras:
        cameras = "1 camera" if len(cfg.bays) == 1 else f"{len(cfg.bays)} cameras"
        print(f"opening {cameras}, then measuring for {args.seconds:.0f} s...", file=sys.stderr)
        try:
            report = benchmark.run_with_cameras(cfg, recognizer, args.seconds)
        except CameraError as exc:
            print(f"camera problem: {exc}", file=sys.stderr)
            return 1
    else:
        bays = args.bays or (len(cfg.bays) if cfg else 4)
        camera = cfg.bays[0].camera if cfg else None
        width = args.width or (camera.width if camera else 1920)
        height = args.height or (camera.height if camera else 1080)
        if width < 640 or height < 360:
            raise ConfigError("frames must be at least 640x360")
        rois = [bay.roi for bay in cfg.bays] if cfg else [[0.0, 0.0, 1.0, 1.0]]
        print(f"simulating {benchmark.bays_text(bays)} for {args.seconds:.0f} s...", file=sys.stderr)
        report = benchmark.run_synthetic(recognizer, bays, (width, height), rois, args.seconds)
    print(benchmark.format_report(report, min_reads, header))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carwash-lpr", description="Moldovan licence plate recognition for car wash bays"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    config_path = default_config()
    env_file = default_env_file()

    def with_config(p: argparse.ArgumentParser, required: bool = True) -> argparse.ArgumentParser:
        # Optional configs still default to the installed one, so that on the Pi every
        # command uses the service's settings and models.
        installed = os.path.exists(config_path)
        p.add_argument(
            "-c", "--config", default=config_path if required or installed else None,
            help=f"configuration file (default: {config_path}{'' if required else ', if it exists'})",
        )
        p.add_argument(
            "--env-file", default=env_file,
            help=f"secrets referenced as ${{NAME}} in the configuration (default: {env_file})",
        )
        return p

    p = with_config(sub.add_parser("run", help="run the recognition service"))
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging (every OCR read)")
    p.set_defaults(func=cmd_run)

    p = with_config(sub.add_parser("check-config", help="validate the configuration and print it"))
    p.set_defaults(func=cmd_check_config)

    p = with_config(sub.add_parser("download-models", help="download the detector and OCR models"), required=False)
    p.set_defaults(func=cmd_download_models)

    p = with_config(sub.add_parser("recognize", help="read plates in image files"), required=False)
    p.add_argument("images", nargs="+")
    p.add_argument("--annotate", metavar="DIR", help="write copies of the images with the plates marked")
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("plate", help="show how a plate text is interpreted")
    p.add_argument("text", nargs="+")
    p.set_defaults(func=cmd_plate)

    p = with_config(sub.add_parser("send-test", help="send one test event to the car wash system"))
    p.add_argument("--bay", help="bay id (default: the first bay)")
    p.add_argument("--plate", default="ABC123")
    p.add_argument("--type", default="test", choices=["test", "plate_recognized", "plate_unrecognized", "vehicle_left"])
    p.set_defaults(func=cmd_send_test)

    p = with_config(sub.add_parser("snapshot", help="save one camera image (stop the service first)"))
    p.add_argument("--bay", help="bay id (default: the first bay)")
    p.add_argument("-o", "--output", default="snapshot.jpg")
    p.add_argument("--warmup", type=float, default=2.0, help="seconds to let exposure/focus settle")
    p.add_argument("--annotate", action="store_true", help="run recognition and mark ROI and plates")
    p.set_defaults(func=cmd_snapshot)

    p = with_config(
        sub.add_parser("benchmark", help="measure how many plate reads per second each bay gets on this Pi"),
        required=False,
    )
    p.add_argument("--bays", type=int, help="number of simulated bays (default: as configured, or 4)")
    p.add_argument("--seconds", type=float, default=20.0, help="how long all bays run at once (default: 20)")
    p.add_argument(
        "--cameras", action="store_true",
        help="use the configured cameras, so video decoding is included (stop the service first)",
    )
    p.add_argument("--width", type=int, help="simulated frame width (default: first bay's camera, or 1920)")
    p.add_argument("--height", type=int, help="simulated frame height (default: first bay's camera, or 1080)")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("demo-images", help="write synthetic camera frames for a dry run without a camera")
    p.add_argument("folder")
    p.add_argument("--plates", nargs="+", default=["KCA 123", "BL AB 123", "ION 7"])
    p.set_defaults(func=cmd_demo_images)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        setup_logging("WARNING")
    if getattr(args, "config", None) and getattr(args, "env_file", None):
        load_env_file(args.env_file)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
