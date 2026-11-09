from __future__ import annotations

import cv2
import numpy as np


ENDING_COLOR = (0, 0, 255)
BIFURCATION_COLOR = (255, 0, 0)


def _draw_header(image: np.ndarray, total: int, rig: int, bif: int) -> np.ndarray:
    header_height = 42
    canvas = cv2.copyMakeBorder(
        image,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    cv2.putText(
        canvas,
        f"Total: {total}  RIG: {rig}  BIF: {bif}",
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_points(
    image: np.ndarray,
    minutiae: list[dict],
    scale: float = 1.0,
) -> np.ndarray:
    for minutia in minutiae:
        x = round(float(minutia["x"]) * scale)
        y = round(float(minutia["y"]) * scale)
        color = ENDING_COLOR if minutia["type"] == "RIG" else BIFURCATION_COLOR
        if minutia["type"] == "RIG":
            cv2.circle(image, (x, y), max(4, round(7 * scale)), color, 2, cv2.LINE_AA)
        else:
            half = max(4, round(6 * scale))
            cv2.rectangle(image, (x - half, y - half), (x + half, y + half), color, 2)

        angle = float(minutia["direction"])
        length = max(12, round(24 * scale))
        end = (
            round(x + length * np.cos(angle)),
            round(y - length * np.sin(angle)),
        )
        cv2.line(image, (x, y), end, color, 2, cv2.LINE_AA)
    return image


def annotate_binary(
    image: np.ndarray,
    mask: np.ndarray,
    minutiae: list[dict],
) -> np.ndarray:
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ridges = (image < 128) & (mask > 0)
    display[ridges] = (185, 185, 185)
    _draw_points(display, minutiae)
    rig = sum(item["type"] == "RIG" for item in minutiae)
    bif = sum(item["type"] == "BIF" for item in minutiae)
    return _draw_header(display, len(minutiae), rig, bif)


def annotate_photo(
    image: np.ndarray,
    minutiae: list[dict],
    max_side: int = 1600,
) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    display = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    _draw_points(display, minutiae, scale)
    rig = sum(item["type"] == "RIG" for item in minutiae)
    bif = sum(item["type"] == "BIF" for item in minutiae)
    return _draw_header(display, len(minutiae), rig, bif)


def save_annotated(
    image: np.ndarray,
    mask: np.ndarray,
    original_photo: np.ndarray,
    minutiae: list[dict],
    output_bw: str,
    output_photo: str,
) -> None:
    binary = annotate_binary(image, mask, minutiae)
    photo = annotate_photo(original_photo, minutiae)
    if not cv2.imwrite(output_bw, binary, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not save annotated binary image: {output_bw}")
    if not cv2.imwrite(output_photo, photo, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not save annotated photo: {output_photo}")
