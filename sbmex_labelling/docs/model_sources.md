# Public-equivalent preprocessing models

These are public equivalents, not the team's original model files. They may differ
from the team's models. No images were uploaded during model acquisition or testing.

## U²-Net fingerprint segmentation

- **Source code:** [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net),
  commit `ac7e1c817ecab7c7dff5ce6b1abba61cd213ff29`.
- **Weights:** the repository's linked full `u2net.pth` checkpoint from
  [Google Drive](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view).
  The downloaded checkpoint was 176,290,937 bytes; SHA-256
  `10025a17f49cd3208afc342b589890e402ee63123d6f2d289a4a0903695cce58`.
- **License:** Apache-2.0, as stated by the upstream repository.
- **Conversion:** loaded the full `U2NET(3, 1)` architecture and checkpoint in
  evaluation mode, exported only the first output (`d0`) to ONNX at opset 17
  with input `[1, 3, 320, 320]`, then converted it to float32 TFLite with
  `onnx2tf` 1.28.8. The converter's bundled validation image was incompatible
  with NumPy 2, so the conversion API's validation image provider was replaced
  in-process with a local all-zero float32 tensor. This does not change model
  weights or conversion graph.
- **Input contract:** `[1, 320, 320, 3]`, float32 RGB in `[0, 1]`.
- **Output contract:** `[1, 320, 320, 1]`, float32 sigmoid probability map in
  `[0, 1]`; the maximum-probability foreground is the finger.
- **TFLite artifact:** `assets/models/u2net_320x320_float32.tflite`,
  176,010,756 bytes; SHA-256
  `a47964fa61d7632f4190e6904029853fa64c89e2f8405a6a0f9d4845a7b64b60`.
- **Contract test:** three liveness images
  (`IMG_20261106_160109.jpg`, `IMG_20261106_160111.jpg`,
  `IMG_20261106_160112.jpg`). Each output had shape 320x320 after squeeze,
  range 0..1, and a nonempty mask at threshold 0.30 (foreground fractions
  0.194, 0.196, 0.221). TFLite output matched the official PyTorch `d0`
  pixel-for-pixel on all three (IoU 1.0000, mean absolute error 0).
  A mask overlay was inspected and covered the finger.
- **Reproduction commands:** run the following exporter from the upstream U-2-Net
  checkout after placing `u2net.pth` beside the script:

  ```powershell
  @'
  import torch
  from model.u2net import U2NET

  class MainOutput(torch.nn.Module):
      def __init__(self, base):
          super().__init__()
          self.base = base

      def forward(self, image):
          return self.base(image)[0]

  model = U2NET(3, 1)
  model.load_state_dict(
      torch.load("u2net.pth", map_location="cpu", weights_only=True),
      strict=True,
  )
  model.eval()
  torch.onnx.export(
      MainOutput(model).eval(),
      torch.zeros(1, 3, 320, 320),
      "u2net_320.onnx",
      input_names=["input"],
      output_names=["probability"],
      opset_version=17,
      dynamo=False,
  )
  '@ | python -
  ```

  `onnx2tf`'s current CLI path attempted to load an incompatible bundled
  validation `.npy` under NumPy 2. Conversion was completed through its Python
  API with a local synthetic validation tensor:

  ```python
  import numpy as np
  import onnx2tf.onnx2tf as converter

  converter.download_test_image_data = lambda: np.zeros(
      (1, 320, 320, 3), dtype=np.float32
  )
  converter.convert(
      input_onnx_file_path="u2net_320.onnx",
      output_folder_path="u2net-tflite",
      copy_onnx_input_output_names_to_tflite=True,
      verbosity="info",
  )
  ```

  Then copy the resulting float32 artifact into the workspace model folder:

  ```powershell
  Copy-Item u2net-tflite\u2net_320_float32.tflite assets\models\u2net_320x320_float32.tflite
  ```

## Zero-DCE low-light enhancement

- **Example implementation:** [keras-team/keras-io Zero-DCE example](https://github.com/keras-team/keras-io/blob/9554fcf2e76cde813fbe6449d48175af7373dfe4/examples/vision/zero_dce.py),
  commit `9554fcf2e76cde813fbe6449d48175af7373dfe4`.
- **Pretrained model:** [keras-io/low-light-image-enhancement](https://huggingface.co/keras-io/low-light-image-enhancement),
  Hub revision `eb992d12eca55d932541b1c59f1e735718f41179`.
- **License:** Apache-2.0, as declared on the Hugging Face model card.
- **Conversion:** loaded the pretrained SavedModel using TensorFlow 2.20.0.
  The SavedModel serving graph emits 24 curve-parameter channels, not an
  enhanced image. A fixed-shape TensorFlow wrapper splits those into eight
  RGB curves, applies the full Zero-DCE curve sequence to the RGB input, clips
  the enhanced result to `[0, 1]`, and was converted with
  `tf.lite.TFLiteConverter.from_concrete_functions`. Thus the TFLite output
  is the enhanced image, not raw curve parameters.
- **Input/output contract:** input `[1, 256, 256, 3]`, float32 RGB in `[0, 1]`;
  output `[1, 256, 256, 3]`, float32 enhanced RGB in `[0, 1]`.
- **TFLite artifact:** `assets/models/zero_dce_model_256.tflite`,
  328,740 bytes; SHA-256
  `9f859ea345bb3bb80373cf6ed78ca54365b9b734fc23c10f11f623a5fb429d8b`.
- **Contract test:** three liveness images (the same filenames listed above).
  Every output had shape 256x256x3 after squeeze and values in `[0, 1]`.
  Mean grayscale luminance changed from 176.41 to 252.58, 175.95 to 252.53,
  and 173.06 to 252.39. Input/output flattened correlations were 0.8827,
  0.8895, and 0.8784; the same scenes remain recognizable. However, 78.0-81.7%
  of output channel values were at least 0.99: the photos are strongly washed
  out. This is a material quality limitation, not a successful visual-quality
  assessment. The preprocessing script remains unchanged and only uses this
  branch when its measured luminance is below 150.
- **Reproduction command:** with the pinned SavedModel extracted to
  `zero_dce_saved_model`, run a Python conversion script implementing the
  following graph. The loaded SavedModel must remain referenced while tracing
  the wrapper so its variables stay alive.

  ```python
  import tensorflow as tf

  loaded = tf.saved_model.load("zero_dce_saved_model")
  source = loaded.signatures["serving_default"]

  class Enhancement(tf.Module):
      @tf.function(input_signature=[
          tf.TensorSpec([1, 256, 256, 3], tf.float32, name="image")
      ])
      def __call__(self, image):
          params = source(input_3=image)["conv2d_20"]
          enhanced = image
          for curve in tf.split(params, 8, axis=-1):
              enhanced = enhanced + curve * (tf.square(enhanced) - enhanced)
          return {"enhanced": tf.clip_by_value(enhanced, 0.0, 1.0)}

  module = Enhancement()
  concrete = module.__call__.get_concrete_function()
  converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
  with open("zero_dce_model_256.tflite", "wb") as model_file:
      model_file.write(converter.convert())
  ```

## Provenance and scope

Both artifacts are public substitutes and must be identified that way in labels
and reports. U²-Net is the segmentation model only; the low-light model is
enhancement only. Neither model generates minutiae.
