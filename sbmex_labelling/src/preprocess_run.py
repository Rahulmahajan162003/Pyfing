from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dpi import estimate_ridge_period


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PUBLIC_EQUIVALENT = "public equivalent of our team's model (may differ)"


def _load_preprocessor(workspace_root: Path):
    root = workspace_root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import preprocess_fingerprint

    return preprocess_fingerprint


def _save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def _make_center_crops(
    photo: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    stem: str,
    size: int = 600,
) -> None:
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"]:
        center_x = round(moments["m10"] / moments["m00"])
        center_y = round(moments["m01"] / moments["m00"])
    else:
        center_x, center_y = photo.shape[1] // 2, photo.shape[0] // 2

    x0 = max(0, min(photo.shape[1] - size, center_x - size // 2))
    y0 = max(0, min(photo.shape[0] - size, center_y - size // 2))
    x1, y1 = x0 + size, y0 + size
    photo_crop = photo[y0:y1, x0:x1]
    processed_crop = processed[y0:min(y1, processed.shape[0]), x0:x1]
    photo_crop = _pad_crop(photo_crop, size, size, 255)
    processed_crop = _pad_crop(processed_crop, size, size, 255)
    photo_crop = cv2.cvtColor(photo_crop, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_GRAY2RGB)
    panel = np.concatenate((photo_crop, processed_rgb), axis=1)
    _save_image(output_dir / f"{stem}_crop.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    _save_image(
        output_dir / f"{stem}_photo_crop.png",
        cv2.cvtColor(photo_crop, cv2.COLOR_RGB2BGR),
    )
    _save_image(
        output_dir / f"{stem}_processed_crop.png",
        cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR),
    )


def _pad_crop(crop: np.ndarray, height: int, width: int, value: int) -> np.ndarray:
    missing_height = height - crop.shape[0]
    missing_width = width - crop.shape[1]
    if missing_height < 0 or missing_width < 0:
        return crop[:height, :width]
    return cv2.copyMakeBorder(
        crop,
        0,
        missing_height,
        0,
        missing_width,
        cv2.BORDER_CONSTANT,
        value=value,
    )


def _process_one(
    path: Path,
    preprocessor: Any,
    segmentation_model: Any,
    zero_dce_model: Any,
    workspace_root: Path,
) -> dict[str, Any]:
    photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if photo is None:
        raise RuntimeError(f"Could not read input image: {path}")

    started = time.perf_counter()
    processed, mask, info = preprocessor.preprocess(
        photo,
        segmentation_model,
        zero_dce_model,
        slap_normalize=False,
    )
    elapsed = time.perf_counter() - started

    if processed.shape[1] != photo.shape[1]:
        raise RuntimeError(
            f"Processed width {processed.shape[1]} does not match photo width {photo.shape[1]}"
        )
    if processed.shape[0] > photo.shape[0]:
        raise RuntimeError(
            f"Processed height {processed.shape[0]} exceeds photo height {photo.shape[0]}"
        )

    stem = path.stem
    processed_path = workspace_root / "work" / "processed" / f"{stem}.png"
    mask_path = workspace_root / "work" / "masks" / f"{stem}.png"
    info_path = workspace_root / "work" / "info" / f"{stem}.json"
    debug_dir = workspace_root / "work" / "debug"
    _save_image(processed_path, processed)
    _save_image(mask_path, (mask * 255).astype(np.uint8))
    _make_center_crops(photo, processed, mask, debug_dir, stem)

    mask_for_processed = mask[: processed.shape[0], :]
    inside = mask_for_processed != 0
    black_fraction = (
        float(np.count_nonzero((processed < 128) & inside) / np.count_nonzero(inside))
        if np.any(inside)
        else None
    )
    ridge = estimate_ridge_period(processed)
    result = {
        **info,
        "image_id": stem,
        "source_photo": str(path),
        "original_size": [int(photo.shape[1]), int(photo.shape[0])],
        "processed_size": [int(processed.shape[1]), int(processed.shape[0])],
        "seconds": round(elapsed, 3),
        "slap_normalize": False,
        "model_provenance": PUBLIC_EQUIVALENT,
        "u2net_model": PUBLIC_EQUIVALENT,
        "zero_dce_model": PUBLIC_EQUIVALENT,
        "black_pixel_fraction_inside_mask": black_fraction,
        "ridge_period": ridge,
        "processed_file": str(processed_path.relative_to(workspace_root)),
        "mask_file": str(mask_path.relative_to(workspace_root)),
        "center_crop_file": str((debug_dir / f"{stem}_crop.png").relative_to(workspace_root)),
    }
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def preprocess_images(
    input_dir: Path,
    workspace_root: Path,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    input_dir = input_dir.resolve()
    paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: str(path).lower(),
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No supported input images found in {input_dir}")

    log_path = workspace_root / "logs" / "preprocess_log.txt"
    failure_path = workspace_root / "logs" / "preprocess_failures.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logger = logging.getLogger("sbmex_preprocess")
    logger.info("Input=%s selected_images=%d", input_dir, len(paths))

    preprocessor = _load_preprocessor(workspace_root)
    assets_dir = workspace_root.parent / "assets" / "models"
    segmentation_model = preprocessor.TFLiteModel(
        assets_dir / "u2net_320x320_float32.tflite"
    )
    zero_dce_model = preprocessor.TFLiteModel(
        assets_dir / "zero_dce_model_256.tflite"
    )

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, path in enumerate(paths, start=1):
        stem = path.stem
        processed_path = workspace_root / "work" / "processed" / f"{stem}.png"
        mask_path = workspace_root / "work" / "masks" / f"{stem}.png"
        info_path = workspace_root / "work" / "info" / f"{stem}.json"
        if processed_path.is_file() and mask_path.is_file() and info_path.is_file():
            try:
                result = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Reprocessing %s because saved info is unreadable: %s", path, exc)
            else:
                print(f"[{index}/{len(paths)}] SKIP {path.name} (already processed)")
                results.append(result)
                continue
        try:
            result = _process_one(
                path,
                preprocessor,
                segmentation_model,
                zero_dce_model,
                workspace_root,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed %s", path)
            print(f"[{index}/{len(paths)}] FAIL {path.name}: {message}")
            failures.append((str(path), message))
            continue
        results.append(result)
        ridge = result["ridge_period"]
        period = ridge["ridge_period_px"]
        strength = ridge["fft_peak_strength"]
        print(
            f"[{index}/{len(paths)}] OK {path.name}: "
            f"{result['processed_size'][0]}x{result['processed_size'][1]}, "
            f"enhancement={result['enhancement']}, "
            f"ridge_period={period if period is not None else 'unavailable'}, "
            f"fft_peak_strength={strength if strength is not None else 'unavailable'}, "
            f"black_fraction={result['black_pixel_fraction_inside_mask']}, "
            f"{result['seconds']}s"
        )
        logger.info("Processed %s: %s", path, result)

    with failure_path.open("w", encoding="utf-8", newline="") as output:
        output.write("image,error\n")
        for image, error in failures:
            escaped = error.replace('"', '""')
            output.write(f'"{image}","{escaped}"\n')
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess fingerprint photos for SBMEX")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Process the first N sorted images; default is the Phase C first 10",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    workspace_root = Path(__file__).resolve().parents[1]
    results = preprocess_images(args.input, workspace_root, args.limit)
    print(f"Completed: {len(results)} images succeeded or were already processed.")
    if not results:
        raise RuntimeError("Preprocessing completed with no successful images.")


if __name__ == "__main__":
    main()
