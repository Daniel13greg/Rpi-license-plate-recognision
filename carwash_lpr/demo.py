"""Synthetic Moldovan plates for tests and for a dry run without a camera.

The images are crude (OpenCV's built-in font, a box for a car) but the plate layout,
the flag band with "MD" and the proportions follow the real 520 x 112 mm plates.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

BLUE = (153, 51, 0)  # BGR
FLAG = ((174, 70, 0), (0, 210, 255), (47, 9, 204))  # blue, yellow, red


def render_plate(text: str, green: bool = False, width: int = 520) -> np.ndarray:
    """A plate image; ``green`` draws the characters of the electric vehicle plates."""
    height = round(width * 112 / 520)
    s = width / 520
    img = np.full((height, width, 3), 245, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), max(2, round(4 * s)))
    cv2.rectangle(img, (round(4 * s), round(4 * s)), (round(60 * s), height - round(5 * s)), BLUE, -1)
    for i, colour in enumerate(FLAG):
        x = round((12 + i * 14) * s)
        cv2.rectangle(img, (x, round(14 * s)), (x + round(14 * s), round(42 * s)), colour, -1)
    cv2.putText(img, "MD", (round(10 * s), round(85 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (255, 255, 255), max(1, round(2 * s)), cv2.LINE_AA)
    font, thickness = cv2.FONT_HERSHEY_DUPLEX, max(2, round(9 * s))
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, thickness)
    scale = min((width - 90 * s) / tw, 82 * s / th)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = round(70 * s + ((width - 80 * s) - tw) / 2)
    colour = (60, 140, 0) if green else (15, 15, 15)
    cv2.putText(img, text, (x, (height + th) // 2), font, scale, colour, thickness, cv2.LINE_AA)
    return img


def render_scene(
    plate_text: str | None,
    size: tuple[int, int] = (1280, 720),
    plate_width: int = 260,
    offset: tuple[int, int] = (0, 0),
    green: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """A bay with a car whose front plate reads ``plate_text`` (None: empty bay)."""
    width, height = size
    rng = np.random.default_rng(seed)
    ramp = np.linspace(60, 140, height, dtype=np.float32)[:, None, None]
    scene = np.broadcast_to(ramp, (height, width, 3)).astype(np.uint8).copy()
    if plate_text is not None:
        cx, cy = width // 2 + offset[0], int(height * 0.62) + offset[1]
        cv2.rectangle(scene, (cx - 420, cy - 260), (cx + 420, cy + 200), (40, 40, 140), -1)
        cv2.rectangle(scene, (cx - 300, cy - 240), (cx + 300, cy - 110), (30, 30, 30), -1)
        for dx in (-330, 330):
            cv2.circle(scene, (cx + dx, cy - 40), 45, (200, 200, 200), -1)
        plate = render_plate(plate_text, green=green, width=plate_width)
        ph, pw = plate.shape[:2]
        x, y = cx - pw // 2, cy + 60
        scene[y : y + ph, x : x + pw] = plate
    noise = rng.normal(0, 3, scene.shape)
    return np.clip(scene + noise, 0, 255).astype(np.uint8)


def write_demo_images(folder: str | Path, plates: list[str], frames_per_car: int = 12, empty_frames: int = 12) -> list[Path]:
    """A sequence of frames: empty bay, a car arriving, standing, leaving; for each plate."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    n = 0
    for car, text in enumerate(plates):
        for i in range(empty_frames):
            paths.append(folder / f"{n:05d}.jpg")
            cv2.imwrite(str(paths[-1]), render_scene(None, seed=n))
            n += 1
        for i in range(frames_per_car):
            # the car drives in: the plate grows and moves up a little
            progress = min(1.0, (i + 1) / max(1, frames_per_car // 2))
            plate_width = int(180 + 100 * progress)
            paths.append(folder / f"{n:05d}.jpg")
            cv2.imwrite(str(paths[-1]), render_scene(text, plate_width=plate_width, offset=(0, int(40 * (1 - progress))), seed=n))
            n += 1
    return paths
