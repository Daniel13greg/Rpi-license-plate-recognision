"""Configuration: a YAML file mapped onto dataclasses with validation.

String values may reference environment variables as ``${NAME}`` or ``${NAME:-default}``,
which keeps secrets out of the YAML file (systemd loads them from an EnvironmentFile).
Relative paths are resolved against the directory of the configuration file.
"""

from __future__ import annotations

import dataclasses
import os
import re
import socket
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Union

import yaml

from carwash_lpr.plates import CATEGORIES

PlateCategory = Literal["moldova", "moldova_special", "foreign", "unknown"]
EventType = Literal["plate_recognized", "plate_unrecognized", "vehicle_left"]

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_BAY_ID = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_SECRET_FIELDS = {"bearer_token", "hmac_secret", "token"}


class ConfigError(ValueError):
    pass


@dataclass
class RecognizerConfig:
    detector_model: str = "yolo-v9-t-512-license-plate-end2end"
    detector_model_path: str | None = None  # custom YOLOv9 ONNX plate detector
    detector_confidence: float = 0.4
    ocr_model: str = "cct-xs-v2-global-model"
    ocr_model_path: str | None = None  # custom fast-plate-ocr ONNX model ...
    ocr_config_path: str | None = None  # ... and its plate config YAML
    threads: int = 4
    min_plate_width: int = 60  # pixels; smaller plates are too far away to read reliably
    edge_margin: int = 4  # pixels; plates touching the frame/ROI edge are probably cut off
    crop_padding: float = 0.04  # grow each detected box by this fraction before OCR

    def validate(self, path: str) -> None:
        _check_range(f"{path}.detector_confidence", self.detector_confidence, 0.0, 1.0)
        _check_range(f"{path}.threads", self.threads, 1, 64)
        _check_range(f"{path}.min_plate_width", self.min_plate_width, 0, 10_000)
        _check_range(f"{path}.edge_margin", self.edge_margin, 0, 1_000)
        _check_range(f"{path}.crop_padding", self.crop_padding, 0.0, 0.5)
        if bool(self.ocr_model_path) != bool(self.ocr_config_path):
            raise ConfigError(f"{path}: set both ocr_model_path and ocr_config_path for a custom OCR model")


@dataclass
class PlatesConfig:
    # Which kinds of plates may produce events (see carwash_lpr.plates.FORMATS).
    accept: list[PlateCategory] = field(default_factory=lambda: ["moldova", "moldova_special", "foreign"])
    min_read_confidence: float = 0.6  # a single frame's read below this is ignored

    def validate(self, path: str) -> None:
        if not self.accept:
            raise ConfigError(f"{path}.accept: list at least one of {', '.join(CATEGORIES)}")
        _check_range(f"{path}.min_read_confidence", self.min_read_confidence, 0.0, 1.0)


@dataclass
class VotingConfig:
    window_seconds: float = 3.0  # reads older than this are forgotten
    min_reads: int = 2  # frames that must agree before a plate is reported
    min_agreement: float = 0.6  # share of the window's votes the winning plate needs
    min_confidence: float = 0.8  # average OCR confidence the winning plate needs

    def validate(self, path: str) -> None:
        _check_range(f"{path}.window_seconds", self.window_seconds, 0.1, 600)
        _check_range(f"{path}.min_reads", self.min_reads, 1, 100)
        _check_range(f"{path}.min_agreement", self.min_agreement, 0.01, 1.0)
        _check_range(f"{path}.min_confidence", self.min_confidence, 0.0, 1.0)


@dataclass
class CameraConfig:
    type: Literal["picamera2", "rpicam", "opencv", "images"] = "picamera2"
    index: int = 0  # CSI camera number (picamera2/rpicam) or /dev/videoN (opencv without url)
    url: str | None = None  # opencv: rtsp://..., http://..., /dev/video0 or a video file
    path: str | None = None  # images: folder with test pictures
    width: int = 1920
    height: int = 1080
    fps: float = 10.0
    hflip: bool = False
    vflip: bool = False
    af_mode: Literal["default", "manual", "auto", "continuous"] = "default"
    lens_position: float | None = None  # dioptres (1 / metres) for manual focus
    exposure_time_us: int | None = None  # short exposures avoid motion blur
    analogue_gain: float | None = None
    controls: dict[str, Any] = field(default_factory=dict)  # extra libcamera controls
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    extra_args: list[str] = field(default_factory=list)  # extra rpicam-vid arguments
    loop: bool = True  # images / video files: start again at the end

    def validate(self, path: str) -> None:
        _check_range(f"{path}.width", self.width, 64, 10_000)
        _check_range(f"{path}.height", self.height, 64, 10_000)
        _check_range(f"{path}.fps", self.fps, 0.1, 240)
        _check_range(f"{path}.index", self.index, 0, 64)
        if self.type == "images" and not self.path:
            raise ConfigError(f"{path}.path: required for camera type 'images'")
        if self.lens_position is not None and self.af_mode not in ("default", "manual"):
            raise ConfigError(f"{path}.lens_position: only used with af_mode 'manual'")


@dataclass
class MotionConfig:
    enabled: bool = True  # only run recognition while something moves
    threshold: float = 0.005  # fraction of changed pixels that counts as motion
    hold_seconds: float = 5.0  # keep recognising this long after the last motion
    idle_interval_seconds: float = 2.0  # still check a frame this often without motion (0 = never)

    def validate(self, path: str) -> None:
        _check_range(f"{path}.threshold", self.threshold, 0.0, 1.0)
        _check_range(f"{path}.hold_seconds", self.hold_seconds, 0.0, 3600)
        _check_range(f"{path}.idle_interval_seconds", self.idle_interval_seconds, 0.0, 3600)


@dataclass
class PresenceConfig:
    """Continuous mode: decides when a car has arrived or left from the plate alone."""

    absence_timeout_seconds: float = 30.0  # plate unseen this long: the car has left
    repeat_cooldown_seconds: float = 300.0  # same plate again within this time: no new event
    report_departures: bool = False  # send vehicle_left (best effort without a sensor)

    def validate(self, path: str) -> None:
        _check_range(f"{path}.absence_timeout_seconds", self.absence_timeout_seconds, 1.0, 86_400)
        _check_range(f"{path}.repeat_cooldown_seconds", self.repeat_cooldown_seconds, 0.0, 86_400)


@dataclass
class TriggerConfig:
    """Trigger mode: a presence sensor (GPIO) or an HTTP call starts each read."""

    gpio_pin: int | None = None  # BCM number of the presence sensor input
    active_high: bool = True  # sensor output is high while a car is present
    pull_up: bool | None = None  # internal pull resistor: true=up, false=down, null=none
    activate_delay_seconds: float = 0.2  # sensor must be active this long (debounce)
    release_delay_seconds: float = 3.0  # sensor must be inactive this long before the car has left
    window_seconds: float = 10.0  # give up reading after this long
    pre_trigger_seconds: float = 2.0  # also use reads from just before the trigger
    report_unrecognized: bool = True  # send plate_unrecognized when no plate could be read
    report_departures: bool = True  # send vehicle_left when the sensor releases

    def validate(self, path: str) -> None:
        if self.gpio_pin is not None:
            _check_range(f"{path}.gpio_pin", self.gpio_pin, 0, 27)
        _check_range(f"{path}.activate_delay_seconds", self.activate_delay_seconds, 0.0, 60)
        _check_range(f"{path}.release_delay_seconds", self.release_delay_seconds, 0.0, 600)
        _check_range(f"{path}.window_seconds", self.window_seconds, 0.5, 600)
        _check_range(f"{path}.pre_trigger_seconds", self.pre_trigger_seconds, 0.0, 60)


@dataclass
class BayConfig:
    id: str
    name: str = ""
    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # x1, y1, x2, y2 (0..1)
    process_fps: float = 4.0  # recognition attempts per second
    mode: Literal["continuous", "trigger"] = "continuous"
    motion: MotionConfig = field(default_factory=MotionConfig)
    presence: PresenceConfig = field(default_factory=PresenceConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)

    def validate(self, path: str) -> None:
        if not _BAY_ID.match(self.id):
            raise ConfigError(f"{path}.id: use 1-32 letters, digits, '-' or '_' (got {self.id!r})")
        if len(self.roi) != 4:
            raise ConfigError(f"{path}.roi: expected [x1, y1, x2, y2]")
        x1, y1, x2, y2 = self.roi
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ConfigError(f"{path}.roi: need 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 (fractions of the frame)")
        _check_range(f"{path}.process_fps", self.process_fps, 0.1, 60)
        if not self.name:
            self.name = f"Bay {self.id}"


@dataclass
class WebhookConfig:
    url: str = ""  # empty: events are only kept locally (and served by the local API)
    method: Literal["POST", "PUT"] = "POST"
    timeout_seconds: float = 5.0
    bearer_token: str = ""
    hmac_secret: str = ""  # signs each request (X-LPR-Signature)
    headers: dict[str, str] = field(default_factory=dict)
    verify_tls: bool = True
    ca_bundle: str | None = None  # custom CA for a self-signed server certificate
    max_event_age_seconds: float = 120.0  # undelivered events older than this are dropped
    include_images: bool = False  # add base64 JPEGs of the plate and the frame
    payload_template: dict[str, Any] | None = None  # reshape the JSON for an existing API
    event_types: list[EventType] = field(
        default_factory=lambda: ["plate_recognized", "plate_unrecognized", "vehicle_left"]
    )

    def validate(self, path: str) -> None:
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ConfigError(f"{path}.url: must start with http:// or https://")
        _check_range(f"{path}.timeout_seconds", self.timeout_seconds, 0.5, 120)
        _check_range(f"{path}.max_event_age_seconds", self.max_event_age_seconds, 1.0, 7 * 86_400)


@dataclass
class ApiConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    token: str = ""  # required as "Authorization: Bearer <token>" (or ?token=) when set

    def validate(self, path: str) -> None:
        _check_range(f"{path}.port", self.port, 1, 65_535)


@dataclass
class StorageConfig:
    save_snapshots: bool = True  # keep a JPEG of the frame and the plate for every event
    snapshot_retention_days: float = 7.0
    max_snapshot_mb: int = 2000
    event_retention_days: float = 30.0

    def validate(self, path: str) -> None:
        _check_range(f"{path}.snapshot_retention_days", self.snapshot_retention_days, 0.01, 3650)
        _check_range(f"{path}.max_snapshot_mb", self.max_snapshot_mb, 10, 10_000_000)
        _check_range(f"{path}.event_retention_days", self.event_retention_days, 0.01, 3650)


@dataclass
class Config:
    device_id: str = field(default_factory=socket.gethostname)
    site_id: str = ""
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: str = "/var/lib/carwash-lpr"
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    plates: PlatesConfig = field(default_factory=PlatesConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    bays: list[BayConfig] = field(default_factory=list)

    def validate(self, path: str) -> None:
        if not self.bays:
            raise ConfigError("bays: configure at least one bay")
        seen = set()
        for bay in self.bays:
            if bay.id in seen:
                raise ConfigError(f"bays: duplicate bay id {bay.id!r}")
            seen.add(bay.id)
        cameras = [b.camera.index for b in self.bays if b.camera.type in ("picamera2", "rpicam")]
        if len(cameras) != len(set(cameras)):
            raise ConfigError("bays: two bays use the same Raspberry Pi camera")

    def bay(self, bay_id: str) -> BayConfig:
        for bay in self.bays:
            if bay.id == bay_id:
                return bay
        raise KeyError(bay_id)


def _check_range(path: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ConfigError(f"{path}: {value} is outside the allowed range {low}..{high}")


def expand_env(value: Any, environ: Mapping[str, str]) -> Any:
    """Replace ${NAME} / ${NAME:-default} in all strings of a parsed YAML document."""
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set (use ${{{name}:-}} if it is optional)")

        return _ENV_REF.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(v, environ) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v, environ) for k, v in value.items()}
    return value


def _convert(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        options = [a for a in args if a is not type(None)]
        errors = []
        for option in options:
            try:
                return _convert(option, value, path)
            except ConfigError as exc:
                errors.append(exc)
        raise errors[0]
    if origin is Literal:
        if value not in args:
            choices = ", ".join(repr(a) for a in args)
            raise ConfigError(f"{path}: {value!r} is not one of {choices}")
        return value
    if dataclasses.is_dataclass(tp):
        return _build(tp, value, path)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list")
        (item_type,) = args or (Any,)
        return [_convert(item_type, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        value_type = args[1] if args else Any
        return {str(k): _convert(value_type, v, f"{path}.{k}") for k, v in value.items()}
    if tp is Any:
        return value
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "yes", "on", "1"):
            return True
        if isinstance(value, str) and value.strip().lower() in ("false", "no", "off", "0"):
            return False
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
    elif tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    name = getattr(tp, "__name__", str(tp))
    raise ConfigError(f"{path}: expected {name}, got {value!r}")


def _build(cls: type, data: Any, path: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        where = f"{path}: " if path else ""
        raise ConfigError(f"{where}unknown option(s): {', '.join(map(str, unknown))}")
    kwargs = {}
    for name, f in fields.items():
        sub = f"{path}.{name}" if path else name
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], sub)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub}: required")
    obj = cls(**kwargs)
    validate = getattr(obj, "validate", None)
    if validate is not None:
        validate(path)
    return obj


def _resolve(base: Path, value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


def config_from_dict(data: Any, base_dir: Path | None = None, environ: Mapping[str, str] | None = None) -> Config:
    data = expand_env(data, os.environ if environ is None else environ)
    cfg = _build(Config, data, "")
    base = base_dir or Path.cwd()
    cfg.data_dir = _resolve(base, cfg.data_dir)
    rec = cfg.recognizer
    rec.detector_model_path = _resolve(base, rec.detector_model_path)
    rec.ocr_model_path = _resolve(base, rec.ocr_model_path)
    rec.ocr_config_path = _resolve(base, rec.ocr_config_path)
    cfg.webhook.ca_bundle = _resolve(base, cfg.webhook.ca_bundle)
    for bay in cfg.bays:
        bay.camera.path = _resolve(base, bay.camera.path)
    return cfg


def load_config(path: str | os.PathLike, environ: Mapping[str, str] | None = None) -> Config:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return config_from_dict(data, path.resolve().parent, environ)


def redacted(cfg: Config) -> dict:
    """The effective configuration as a dict, with secrets masked (for printing)."""

    def mask(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: ("***" if k in _SECRET_FIELDS and v else mask(v)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [mask(v) for v in obj]
        return obj

    return mask(dataclasses.asdict(cfg))
