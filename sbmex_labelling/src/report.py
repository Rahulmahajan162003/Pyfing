from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPORT_FIELDS = (
    "image_id",
    "ridge_period_px",
    "fft_peak_strength",
    "black_pixel_fraction_inside_mask",
    "snffe_period_at_500dpi",
    "total_minutiae",
    "RIG",
    "BIF",
    "flags",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _flags(info: dict[str, Any], label: dict[str, Any] | None) -> list[str]:
    flags: list[str] = []
    ridge_period = info["ridge_period"]["ridge_period_px"]
    if ridge_period is not None and ridge_period > 25:
        flags.append("RIDGE_PERIOD_ABOVE_25PX_THRESHOLD_WINDOW")

    if label is None:
        flags.append("LABELING_FAILED")
        return flags

    snffe_period = label.get("snffe_period_at_500dpi")
    if snffe_period is None:
        flags.append("SNFFE_CROSSCHECK_MISSING")
    elif not 8 <= snffe_period <= 11.5:
        flags.append("SNFFE_PERIOD_OUTSIDE_8_11_5PX")
    return flags


def _make_row(
    info: dict[str, Any],
    label: dict[str, Any] | None,
) -> dict[str, Any]:
    minutiae = label["minutiae"] if label is not None else []
    return {
        "image_id": info["image_id"],
        "ridge_period_px": info["ridge_period"]["ridge_period_px"],
        "fft_peak_strength": info["ridge_period"]["fft_peak_strength"],
        "black_pixel_fraction_inside_mask": info["black_pixel_fraction_inside_mask"],
        "snffe_period_at_500dpi": (
            label.get("snffe_period_at_500dpi") if label is not None else ""
        ),
        "total_minutiae": len(minutiae),
        "RIG": sum(item["type"] == "RIG" for item in minutiae),
        "BIF": sum(item["type"] == "BIF" for item in minutiae),
        "flags": ";".join(_flags(info, label)),
    }


def _write_overview(
    workspace_root: Path,
    image_ids: list[str],
    output_path: Path,
) -> None:
    columns = 2
    tile_width = 320
    tile_height = 425
    tiles: list[np.ndarray] = []
    for image_id in image_ids:
        image_path = (
            workspace_root
            / "output"
            / "annotated_bw"
            / f"{Path(image_id).stem}_sbmex.jpg"
        )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile_width, tile_height - 25), interpolation=cv2.INTER_AREA)
        tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
        tile[25:, :] = image
        cv2.putText(
            tile,
            image_id,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        raise RuntimeError("No annotated binary images are available for the overview")
    rows = (len(tiles) + columns - 1) // columns
    overview = np.full((rows * tile_height, columns * tile_width, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_height
        x = (index % columns) * tile_width
        overview[y : y + tile_height, x : x + tile_width] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overview, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Could not save report overview: {output_path}")


def write_report(workspace_root: Path, limit: int = 10) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    info_paths = sorted(
        (workspace_root / "work" / "info").glob("*.json"),
        key=lambda path: path.name.lower(),
    )[:limit]
    if not info_paths:
        raise FileNotFoundError("No preprocessing info JSON files were found")

    rows: list[dict[str, Any]] = []
    image_ids: list[str] = []
    for info_path in info_paths:
        info = _read_json(info_path)
        if info is None:
            raise RuntimeError(f"Could not read preprocessing info: {info_path}")
        label = _read_json(
            workspace_root
            / "output"
            / "labels_json"
            / f"{info_path.stem}.json"
        )
        rows.append(_make_row(info, label))
        if label is not None:
            image_ids.append(info["image_id"])

    report_path = workspace_root / "output" / "report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _write_overview(
        workspace_root,
        image_ids,
        workspace_root / "output" / "overview.jpg",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an SBMEX batch report and overview")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    workspace_root = Path(__file__).resolve().parents[1]
    rows = write_report(workspace_root, args.limit)
    for row in rows:
        print(
            f"{row['image_id']}: total={row['total_minutiae']} "
            f"RIG={row['RIG']} BIF={row['BIF']} "
            f"SNFFE={row['snffe_period_at_500dpi']} flags={row['flags']}"
        )
    print(f"Report: {workspace_root / 'output' / 'report.csv'}")
    print(f"Overview: {workspace_root / 'output' / 'overview.jpg'}")


if __name__ == "__main__":
    main()
