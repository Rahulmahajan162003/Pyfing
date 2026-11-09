import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


LOW_LIGHT_THRESHOLD = 150.0
SEGMENTATION_THRESHOLD = 0.30


class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def run(self, input_array):
        input_array = input_array.astype(self.input["dtype"])
        self.interpreter.set_tensor(self.input["index"], input_array)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])


def segment_finger(bgr, model):
    input_shape = model.input["shape"]
    height, width = int(input_shape[1]), int(input_shape[2])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height))
    tensor = resized.astype(np.float32) / 255.0
    output = model.run(tensor[np.newaxis, ...])

    probability = np.squeeze(output)

    if probability.ndim == 3:
        probability = probability[..., 0]

    probability = cv2.resize(
        probability,
        (bgr.shape[1], bgr.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    binary = (probability >= SEGMENTATION_THRESHOLD).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return binary

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [hull], -1, 1, thickness=cv2.FILLED)

    return clean_mask


def enhance_zero_dce(rgb, model):
    original_height, original_width = rgb.shape[:2]

    input_shape = model.input["shape"]
    height, width = int(input_shape[1]), int(input_shape[2])

    resized = cv2.resize(rgb, (width, height))
    tensor = resized.astype(np.float32) / 255.0
    output = np.squeeze(model.run(tensor[np.newaxis, ...]))

    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)

    return cv2.resize(
        output,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )


def create_central_roi(mask, alpha=0.25):
    kernel = np.ones((7, 7), dtype=np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return cleaned

    largest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest)

    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)

    erosion_size = int(min(mask.shape[:2]) * alpha * 0.5)

    if erosion_size > 1:
        erosion_kernel = np.ones(
            (erosion_size, erosion_size),
            dtype=np.uint8,
        )
        hull_mask = cv2.erode(hull_mask, erosion_kernel)

    return hull_mask


def auto_crop_bottom(image):
    content_rows = np.where(np.any(image < 200, axis=1))[0]

    if len(content_rows) == 0:
        return image.copy()

    last_row = int(content_rows[-1])
    padding = int(image.shape[0] * 0.02)
    end = min(image.shape[0], last_row + padding + 1)

    return image[:end, :].copy()


def normalize_slap_fingerprint(image):
    output_size = 320
    content_size = 288

    height, width = image.shape[:2]
    scale = content_size / max(width, height)

    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    resized = cv2.resize(image, (resized_width, resized_height))

    canvas = np.full(
        (output_size, output_size),
        255,
        dtype=np.uint8,
    )

    x = (output_size - resized_width) // 2
    y = (output_size - resized_height) // 2

    canvas[
        y:y + resized_height,
        x:x + resized_width,
    ] = resized

    return canvas


def preprocess(bgr, segmentation_model, zero_dce_model, slap_normalize=False):
    mask = segment_finger(bgr, segmentation_model)

    if cv2.countNonZero(mask) == 0:
        raise RuntimeError("U2Net did not find a usable finger region")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    foreground = rgb.copy()
    foreground[mask == 0] = 255

    gray_foreground = cv2.cvtColor(
        foreground,
        cv2.COLOR_RGB2GRAY,
    )
    luminance = float(np.mean(gray_foreground))

    if luminance < LOW_LIGHT_THRESHOLD:
        enhanced_rgb = enhance_zero_dce(
            foreground,
            zero_dce_model,
        )
        enhancement = "zero_dce"
    else:
        equalized = cv2.equalizeHist(gray_foreground)
        enhanced_rgb = cv2.cvtColor(
            equalized,
            cv2.COLOR_GRAY2RGB,
        )
        enhancement = "histogram_equalization"

    enhanced_gray = cv2.cvtColor(
        enhanced_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    thresholded = cv2.adaptiveThreshold(
        enhanced_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        1,
    )

    processed = cv2.bitwise_not(thresholded)
    processed[mask == 0] = 255

    central_roi = create_central_roi(mask)
    processed[central_roi == 0] = 255

    processed = auto_crop_bottom(processed)

    if slap_normalize:
        processed = normalize_slap_fingerprint(processed)

    info = {
        "luminance": round(luminance, 2),
        "enhancement": enhancement,
        "segmented_fraction": round(
            float(cv2.countNonZero(mask)) / mask.size,
            4,
        ),
        "output_width": processed.shape[1],
        "output_height": processed.shape[0],
    }

    return processed, mask, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--u2net",
        default="assets/models/u2net_320x320_float32.tflite",
    )
    parser.add_argument(
        "--zero-dce",
        default="assets/models/zero_dce_model_256.tflite",
    )
    parser.add_argument(
        "--slap-normalize",
        action="store_true",
        help="Place output on the 320x320 slap canvas",
    )
    parser.add_argument(
        "--save-mask",
        help="Optional location for the segmentation mask",
    )
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise RuntimeError(f"Could not read input image: {args.input}")

    segmentation_model = TFLiteModel(args.u2net)
    zero_dce_model = TFLiteModel(args.zero_dce)

    processed, mask, info = preprocess(
        image,
        segmentation_model,
        zero_dce_model,
        slap_normalize=args.slap_normalize,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output), processed):
        raise RuntimeError(f"Could not save output: {output}")

    if args.save_mask:
        mask_output = (mask * 255).astype(np.uint8)
        cv2.imwrite(args.save_mask, mask_output)

    print("Preprocessing completed")
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
