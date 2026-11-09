from __future__ import annotations

import cv2
import numpy as np


def estimate_ridge_period(image: np.ndarray, max_side: int = 1600) -> dict[str, float | int | None]:
    """Estimate ridge spacing and FFT peak contrast in processed-image pixels."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")

    height, width = image.shape
    scale = min(1.0, max_side / max(height, width))
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    dark = (small < 128).astype(np.uint8)
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        np.ones((25, 25), dtype=np.uint8),
    )
    finger = cv2.erode(
        dark,
        np.ones((31, 31), dtype=np.uint8),
    )

    result = _estimate_period_in_blocks(small, finger, 64)
    if result["period_small_px"] is not None and result["period_small_px"] > 20:
        result = _estimate_period_in_blocks(small, finger, 128)

    period_small = result["period_small_px"]
    result["scale_factor"] = float(scale)
    result["ridge_period_px"] = (
        float(period_small) / scale if period_small is not None else None
    )
    return result


def _estimate_period_in_blocks(
    image: np.ndarray,
    finger: np.ndarray,
    block_size: int,
) -> dict[str, float | int | None]:
    height, width = image.shape
    if height < block_size or width < block_size:
        return {
            "block_size": block_size,
            "valid_blocks": 0,
            "period_small_px": None,
            "fft_peak_strength": None,
        }

    frequencies = np.fft.fftshift(np.fft.fftfreq(block_size) * block_size)
    fx, fy = np.meshgrid(frequencies, frequencies)
    radius = np.hypot(fx, fy)
    eligible = (radius >= 2) & (radius <= block_size / 3)
    window = np.outer(np.hanning(block_size), np.hanning(block_size))
    periods: list[float] = []
    strengths: list[float] = []
    step = block_size

    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            if not np.all(finger[y : y + block_size, x : x + block_size]):
                continue
            block = image[y : y + block_size, x : x + block_size].astype(np.float32)
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2((block - block.mean()) * window))) ** 2
            values = spectrum[eligible]
            if values.size == 0:
                continue
            peak_index = np.argmax(np.where(eligible, spectrum, -np.inf))
            peak_y, peak_x = np.unravel_index(peak_index, spectrum.shape)
            peak_radius = radius[peak_y, peak_x]
            if peak_radius <= 0:
                continue
            periods.append(block_size / peak_radius)
            strengths.append(float(spectrum[peak_y, peak_x] / (np.median(values) + 1e-12)))

    return {
        "block_size": block_size,
        "valid_blocks": len(periods),
        "period_small_px": float(np.median(periods)) if periods else None,
        "fft_peak_strength": float(np.median(strengths)) if strengths else None,
    }
