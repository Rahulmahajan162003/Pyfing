from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import keras
import numpy as np
import pyfing as pf
from pyfing import simple_api as pf_simple_api
from pyfing.minutiae_from_enhanced_image import SbmexParameters

from .draw import save_annotated


PYFING_COMMIT = "df4c0f0d19fd1d8befd9fc0615b71d48d784df03"
PUBLIC_EQUIVALENT = "public equivalent of our team's model (may differ)"
MINUTIA_FIELDS = (
    "id",
    "x",
    "y",
    "direction",
    "direction_deg",
    "type",
    "confidence",
    "source",
    "verification_status",
)


def _write_minutiae_csv(path: Path, image_id: str, minutiae: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("image_id", *MINUTIA_FIELDS))
        writer.writeheader()
        for item in minutiae:
            writer.writerow({"image_id": image_id, **item})


def _convert_minutiae(minutiae: list[Any], scale: float) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, minutia in enumerate(minutiae, start=1):
        direction = (float(minutia.direction) + math.pi) % math.tau - math.pi
        if minutia.type == "E":
            minutia_type = "RIG"
        elif minutia.type == "B":
            minutia_type = "BIF"
        else:
            raise ValueError(f"Unexpected SBMEX minutia type: {minutia.type!r}")
        converted.append(
            {
                "id": index,
                "x": float(minutia.x) / scale,
                "y": float(minutia.y) / scale,
                "direction": direction,
                "direction_deg": math.degrees(direction) % 360.0,
                "type": minutia_type,
                "confidence": float(minutia.quality),
                "source": "PYFING_SBMEX",
                "verification_status": "auto_unverified",
            }
        )
    return converted


def _snffe_period(
    image500: np.ndarray,
    segmentation_mask: np.ndarray,
) -> tuple[float | None, str | None]:
    try:
        orientation = pf.orientation_field_estimation(
            image500,
            segmentation_mask=segmentation_mask,
            dpi=500,
            method="SNFOE",
        )
        period_map = pf.frequency_estimation(
            image500,
            orientation,
            segmentation_mask=segmentation_mask,
            dpi=500,
            method="SNFFE",
        )
        valid = (
            (segmentation_mask > 0)
            & np.isfinite(period_map)
            & (period_map > 0)
        )
        if not np.any(valid):
            return None, "SNFFE returned no finite positive periods inside SUFS mask"
        return float(np.median(period_map[valid])), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _release_models(*model_names: str) -> None:
    for model_name in model_names:
        setattr(pf_simple_api, f"_{model_name}_alg", None)
    keras.backend.clear_session()
    gc.collect()


def _load_photo(path: str) -> np.ndarray:
    photo = cv2.imread(path, cv2.IMREAD_COLOR)
    if photo is None:
        raise RuntimeError(f"Could not read source photo: {path}")
    return photo


def _label_one(
    info_path: Path,
    workspace_root: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    image_id = info["image_id"]
    stem = Path(image_id).stem
    processed_path = workspace_root / "work" / "processed" / f"{stem}.png"
    processed = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
    if processed is None:
        raise RuntimeError(f"Could not read processed image: {processed_path}")
    photo = _load_photo(info["source_photo"])

    ridge_period = info["ridge_period"]["ridge_period_px"]
    if ridge_period is None or not math.isfinite(float(ridge_period)) or ridge_period <= 0:
        raise RuntimeError(f"Invalid measured ridge period: {ridge_period!r}")
    scale = 9.5 / float(ridge_period)
    width = max(1, round(processed.shape[1] * scale))
    height = max(1, round(processed.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(processed, (width, height), interpolation=interpolation)
    image500 = np.where(resized < 128, 0, 255).astype(np.uint8)

    debug_dir = workspace_root / "work" / "debug" / stem
    debug_paths = {
        "img500.png": debug_dir / "img500.png",
        "seg.png": debug_dir / "seg.png",
        "sbmex_input.png": debug_dir / "sbmex_input.png",
    }
    cached = {
        name: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        for name, path in debug_paths.items()
    }
    if (
        all(image is not None and image.shape == image500.shape for image in cached.values())
        and np.array_equal(cached["img500.png"], image500)
        and np.array_equal(cached["sbmex_input.png"], np.where(image500 < 128, 255, 0))
    ):
        seg = cached["seg.png"]
        sbmex_input = cached["sbmex_input.png"]
        logger.info("%s: reusing verified 500-dpi debug inputs", image_id)
        print(f"{image_id}: reusing verified 500-dpi debug inputs")
    else:
        seg = pf.fingerprint_segmentation(image500, dpi=500, method="SUFS")
        if seg.shape != image500.shape or seg.dtype != np.uint8:
            raise RuntimeError(
                f"SUFS returned invalid mask shape/dtype: {seg.shape}/{seg.dtype}"
            )
        sbmex_input = np.where(image500 < 128, 255, 0).astype(np.uint8)

    inside = seg > 0
    ridge_pixels = image500 == 0
    sbmex_white = sbmex_input == 255
    polarity_match = bool(np.array_equal(ridge_pixels[inside], sbmex_white[inside]))
    print(
        f"{image_id}: polarity check inside SUFS mask: "
        f"{int(np.count_nonzero(ridge_pixels & inside))} black ridges / "
        f"{int(np.count_nonzero(sbmex_white & inside))} "
        f"white SBMEX pixels; exact match={polarity_match}"
    )
    if not polarity_match:
        raise RuntimeError("Black-ridge to white-SBMEX polarity check failed")

    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_images = {
        "img500.png": image500,
        "seg.png": seg,
        "sbmex_input.png": sbmex_input,
    }
    for name, image in debug_images.items():
        if not cv2.imwrite(str(debug_dir / name), image):
            raise OSError(f"Could not save SBMEX debug image: {debug_dir / name}")

    _release_models("sufs_alg")

    raw_minutiae = pf.minutiae_extraction(
        sbmex_input,
        dpi=500,
        method="SBMEX",
        segmentation_mask=seg,
    )
    minutiae = _convert_minutiae(raw_minutiae, scale)
    print(
        f"{image_id}: SBMEX returned {len(raw_minutiae)} minutiae "
        f"(RIG={sum(m['type'] == 'RIG' for m in minutiae)}, "
        f"BIF={sum(m['type'] == 'BIF' for m in minutiae)})"
    )

    output_dir = workspace_root / "output"
    mask_path = workspace_root / "work" / "masks" / f"{stem}.png"
    stored_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if stored_mask is None:
        raise RuntimeError(f"Could not load stored preprocessing mask: {mask_path}")
    annotation_dir_bw = output_dir / "annotated_bw"
    annotation_dir_photo = output_dir / "annotated_photo"
    annotation_dir_bw.mkdir(parents=True, exist_ok=True)
    annotation_dir_photo.mkdir(parents=True, exist_ok=True)
    save_annotated(
        processed,
        stored_mask,
        photo,
        minutiae,
        str(annotation_dir_bw / f"{stem}_sbmex.jpg"),
        str(annotation_dir_photo / f"{stem}_sbmex.jpg"),
    )

    snffe_period, snffe_warning = _snffe_period(image500, seg)
    _release_models("snfoe_alg", "snffe_alg")
    if snffe_warning:
        logger.warning("%s SNFFE cross-check failed: %s", image_id, snffe_warning)
        print(f"{image_id}: SNFFE cross-check unavailable: {snffe_warning}")
    else:
        print(f"{image_id}: SNFFE median period at 500 dpi: {snffe_period:.3f}px")

    label = {
        "image_id": image_id,
        "source_photo": info["source_photo"],
        "image_size": [int(photo.shape[1]), int(photo.shape[0])],
        "model": f"pyfing {pf.__version__} SBMEX",
        "pyfing_commit": PYFING_COMMIT,
        "keras_backend": os.environ.get("KERAS_BACKEND", "tensorflow"),
        "model_provenance": PUBLIC_EQUIVALENT,
        "sbmex_parameters": SbmexParameters().__dict__,
        "segmentation": "pyfing SUFS",
        "preprocessing": {
            "script": "preprocess_fingerprint.py",
            "slap_normalize": False,
            "model_provenance": PUBLIC_EQUIVALENT,
            "info": info,
        },
        "ridge_period_px": float(ridge_period),
        "scale_to_500dpi": float(scale),
        "snffe_period_at_500dpi": snffe_period,
        "snffe_warning": snffe_warning,
        "coordinates": "pixels of the original photo (= processed image), origin top-left",
        "polarity_check": {
            "passed": polarity_match,
            "black_ridges_inside_segmentation": int(
                np.count_nonzero(ridge_pixels & inside)
            ),
            "white_sbmex_pixels_inside_segmentation": int(
                np.count_nonzero(sbmex_white & inside)
            ),
        },
        "minutiae": minutiae,
    }

    return label


def label_preprocessed(
    workspace_root: Path,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")
    info_paths = sorted(
        (workspace_root / "work" / "info").glob("*.json"),
        key=lambda path: path.name.lower(),
    )
    if limit is not None:
        info_paths = info_paths[:limit]
    if not info_paths:
        raise FileNotFoundError("No preprocessing info JSON files were found")

    log_path = workspace_root / "logs" / "label_log.txt"
    failure_path = workspace_root / "logs" / "label_failures.csv"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logger = logging.getLogger("sbmex_label")
    logger.info(
        "Starting label run: selected=%d pyfing=%s commit=%s",
        len(info_paths),
        pf.__version__,
        PYFING_COMMIT,
    )
    print(
        f"pyfing {pf.__version__} commit {PYFING_COMMIT}; "
        f"SBMEX defaults: {SbmexParameters().__dict__}"
    )

    completed: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, info_path in enumerate(info_paths, start=1):
        image_id = info_path.stem
        label_path = workspace_root / "output" / "labels_json" / f"{image_id}.json"
        if label_path.is_file():
            try:
                label = json.loads(label_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Reprocessing %s because saved label is unreadable: %s", image_id, exc)
            else:
                print(f"[{index}/{len(info_paths)}] SKIP {image_id} (already labeled)")
                completed.append(label)
                continue
        started = time.perf_counter()
        try:
            label = _label_one(
                info_path,
                workspace_root,
                logger,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to label %s", image_id)
            print(f"[{index}/{len(info_paths)}] FAIL {image_id}: {message}")
            failures.append((image_id, message))
            continue
        label["labeling_seconds"] = round(time.perf_counter() - started, 3)
        stem = info_path.stem
        json_path = workspace_root / "output" / "labels_json" / f"{stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(label, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _write_minutiae_csv(
            workspace_root / "output" / "labels_csv" / f"{stem}.csv",
            label["image_id"],
            label["minutiae"],
        )
        completed.append(label)
        logger.info("Labeled %s; minutiae=%d", image_id, len(label["minutiae"]))
        print(f"[{index}/{len(info_paths)}] OK {image_id}: {label['labeling_seconds']}s")

    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("image", "error"))
        writer.writerows(failures)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Label preprocessed fingerprints with pyfing SBMEX")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Label the first N preprocessed images; default is the approved first 10",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    workspace_root = Path(__file__).resolve().parents[1]
    labels = label_preprocessed(workspace_root, args.limit)
    print(f"Completed: {len(labels)} labeled or already labeled images.")
    if not labels:
        raise RuntimeError("No images were labeled successfully.")


if __name__ == "__main__":
    main()
