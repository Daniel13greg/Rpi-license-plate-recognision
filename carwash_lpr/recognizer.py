"""Plate detection (YOLOv9, open-image-models) and OCR (fast-plate-ocr) on ONNX Runtime.

Both networks are small enough for a Raspberry Pi 4/5 CPU. The OCR model is trained on
plates from 65+ countries, Moldova included, and also predicts the plate's country.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from carwash_lpr.config import ConfigError, RecognizerConfig

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels


@dataclass
class PlateRead:
    """One plate found in one frame."""

    text: str  # raw OCR text
    confidence: float  # mean probability of the characters and of the end of the plate
    min_char_confidence: float
    box: Box
    detection_confidence: float
    region: str | None = None  # country predicted by the OCR model
    region_confidence: float | None = None
    crop: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def area(self) -> int:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class Recognizer(Protocol):
    def recognize(self, frame: np.ndarray, roi: Box | None = None) -> list[PlateRead]: ...


def roi_pixels(roi: Sequence[float], width: int, height: int) -> Box:
    """Convert a [x1, y1, x2, y2] fraction-of-frame ROI to pixels."""
    x1, y1, x2, y2 = roi
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        max(int(round(x2 * width)), int(round(x1 * width)) + 1),
        max(int(round(y2 * height)), int(round(y1 * height)) + 1),
    )


def ensure_models(cfg: RecognizerConfig, models_dir: Path) -> tuple[Path, Path, Path]:
    """Paths of the detector, OCR model and OCR config, downloading them if needed."""
    from fast_plate_ocr.inference.hub import AVAILABLE_ONNX_MODELS
    from fast_plate_ocr.inference.hub import download_model as download_ocr
    from open_image_models.detection.core.hub import DETECTION_MODELS
    from open_image_models.detection.core.hub import download_model as download_detector

    if cfg.detector_model_path:
        detector = Path(cfg.detector_model_path)
        if not detector.is_file():
            raise ConfigError(f"recognizer.detector_model_path: {detector} not found")
    elif cfg.detector_model not in DETECTION_MODELS or "license-plate" not in cfg.detector_model:
        names = ", ".join(n for n in DETECTION_MODELS if "license-plate" in n)
        raise ConfigError(f"recognizer.detector_model: unknown model {cfg.detector_model!r}; use one of {names}")
    else:
        detector = download_detector(cfg.detector_model, save_directory=models_dir / cfg.detector_model)
    if cfg.ocr_model_path and cfg.ocr_config_path:
        ocr_model, ocr_config = Path(cfg.ocr_model_path), Path(cfg.ocr_config_path)
        for p in (ocr_model, ocr_config):
            if not p.is_file():
                raise ConfigError(f"recognizer: OCR file {p} not found")
    elif cfg.ocr_model not in AVAILABLE_ONNX_MODELS:
        names = ", ".join(AVAILABLE_ONNX_MODELS)
        raise ConfigError(f"recognizer.ocr_model: unknown model {cfg.ocr_model!r}; use one of {names}")
    else:
        ocr_model, ocr_config = download_ocr(cfg.ocr_model, save_directory=models_dir / cfg.ocr_model)
    return Path(detector), Path(ocr_model), Path(ocr_config)


def _session_options(threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Busy-waiting worker threads would keep the Pi's CPU hot between frames.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return options


class PlateRecognizer:
    """Detector + OCR. Thread safe: bays share one instance and take turns."""

    def __init__(self, cfg: RecognizerConfig, models_dir: Path):
        from fast_plate_ocr import LicensePlateRecognizer
        from open_image_models import create_detector

        self.cfg = cfg
        detector_path, ocr_model, ocr_config = ensure_models(cfg, models_dir)
        providers = ["CPUExecutionProvider"]
        self.detector = create_detector(
            detector_path,
            backend="yolo_v9",
            class_labels=("License Plate",),
            conf_thresh=cfg.detector_confidence,
            providers=providers,
            sess_options=_session_options(cfg.threads),
        )
        self.ocr = LicensePlateRecognizer(
            onnx_model_path=ocr_model,
            plate_config_path=ocr_config,
            providers=providers,
            sess_options=_session_options(cfg.threads),
        )
        self.color_mode = self.ocr.config.image_color_mode
        self.pad_char = self.ocr.config.pad_char or ""
        self._lock = threading.Lock()
        log.info("recognizer ready: detector %s, OCR %s", detector_path.name, ocr_model.name)

    def recognize(self, frame: np.ndarray, roi: Box | None = None) -> list[PlateRead]:
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = roi or (0, 0, width, height)
        view = np.ascontiguousarray(frame[y0:y1, x0:x1])
        vh, vw = view.shape[:2]
        margin = self.cfg.edge_margin
        reads = []
        with self._lock:
            for det in self.detector.predict(view):
                b = det.bounding_box
                bx1, by1, bx2, by2 = max(b.x1, 0), max(b.y1, 0), min(b.x2, vw), min(b.y2, vh)
                bw, bh = bx2 - bx1, by2 - by1
                if bw < max(self.cfg.min_plate_width, 8) or bh < 4:
                    continue
                # A box touching the edge of the view is probably a partly visible plate,
                # which would read as a shorter (wrong) number.
                if margin and (bx1 < margin or by1 < margin or bx2 > vw - margin or by2 > vh - margin):
                    continue
                pad_x = int(bw * self.cfg.crop_padding)
                pad_y = int(bh * self.cfg.crop_padding)
                crop = view[max(by1 - pad_y, 0) : min(by2 + pad_y, vh), max(bx1 - pad_x, 0) : min(bx2 + pad_x, vw)]
                read = self._read_text(crop)
                if read is None:
                    continue
                text, confidence, min_conf, region, region_conf = read
                reads.append(
                    PlateRead(
                        text=text,
                        confidence=confidence,
                        min_char_confidence=min_conf,
                        box=(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                        detection_confidence=float(det.confidence),
                        region=region,
                        region_confidence=region_conf,
                        crop=crop.copy(),
                    )
                )
        return reads

    def _read_text(self, crop: np.ndarray):
        if self.color_mode == "grayscale":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif self.color_mode == "rgb":
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            image = crop
        prediction = self.ocr.run_one(image, return_confidence=True)
        text = prediction.plate
        if prediction.char_probs is None:
            return None
        probs = [float(p) for p in np.asarray(prediction.char_probs).ravel()]
        # The model fills a fixed number of slots; past the text they hold padding. The first
        # padding slot's probability says how sure the model is that the plate ends there.
        used = probs[: len(text) + 1] if len(text) < len(probs) else probs[: len(text)]
        if self.pad_char:
            text = text.replace(self.pad_char, "")
        if not text or not used:
            return None
        region_conf = float(prediction.region_prob) if prediction.region_prob is not None else None
        return text, sum(used) / len(used), min(used), prediction.region, region_conf


def draw_overlay(frame: np.ndarray, roi: Box | None, reads: Sequence[PlateRead], labels: Sequence[str] = ()) -> np.ndarray:
    """Copy of the frame with the ROI and plate boxes drawn, for camera set-up."""
    image = frame.copy()
    scale = max(1.0, image.shape[1] / 1280)
    thickness = max(2, int(2 * scale))
    if roi is not None:
        cv2.rectangle(image, roi[:2], roi[2:], (255, 160, 0), thickness)
    for i, read in enumerate(reads):
        x1, y1, x2, y2 = read.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        label = labels[i] if i < len(labels) else read.text
        text = f"{label} {read.confidence:.0%} {read.width}px"
        org = (x1, max(y1 - int(8 * scale), int(20 * scale)))
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image
