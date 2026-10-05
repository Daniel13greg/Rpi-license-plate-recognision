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
from carwash_lpr.config import CATEGORIES, Config, ConfigError, PlatesConfig, RecognizerConfig, load_config, redacted

DEFAULT_CONFIG = os.environ.get("CARWASH_LPR_CONFIG", "/etc/carwash-lpr/config.yaml")
log = logging.getLogger("carwash_lpr")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carwash-lpr", description="Moldovan licence plate recognition for car wash bays"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser, required: bool = True) -> argparse.ArgumentParser:
        p.add_argument(
            "-c", "--config", default=DEFAULT_CONFIG if required else None,
            help=f"configuration file (default: {DEFAULT_CONFIG})" if required else "configuration file",
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

    p = sub.add_parser("demo-images", help="write synthetic camera frames for a dry run without a camera")
    p.add_argument("folder")
    p.add_argument("--plates", nargs="+", default=["KCA 123", "BL AB 123", "ION 7"])
    p.set_defaults(func=cmd_demo_images)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        setup_logging("WARNING")
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
