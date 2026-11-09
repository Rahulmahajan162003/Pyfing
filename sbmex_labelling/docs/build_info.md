# pyfing build information

- Python: 3.11.4 (64-bit), project virtual environment: `.venv`
- pyfing source: <https://github.com/raffaele-cappelli/pyfing>
- pyfing Git commit: `df4c0f0d19fd1d8befd9fc0615b71d48d784df03`
- pyfing Git description: `v0.7.3-6-gdf4c0f`
- Installed package version: `0.7.3`
- Keras backend for runtime: `tensorflow` (set `KERAS_BACKEND=tensorflow`)
- Installed runtime versions: TensorFlow 2.21.0, Keras 3.15.1, OpenCV
  contrib-python 5.0.0, NumPy 2.4.6, SciPy 1.17.1, scikit-image 0.26.0,
  Matplotlib 3.11.2, pandas 3.1.1, and pytest 9.2.0.
- OpenCV contrib-python is used instead of the standard OpenCV wheel because
  SBMEX calls `cv.ximgproc.thinning`; the standard wheel lacks `ximgproc`.
- `pip check`: no broken requirements found.
- pyfing pretrained weight files:

  | File | Size (bytes) |
  |---|---:|
  | `pyfing/models/SUFS.weights.h5` | 28,447,464 |
  | `pyfing/models/SNFOE.weights.h5` | 19,790,896 |
  | `pyfing/models/SNFFE.weights.h5` | 19,795,992 |
  | `pyfing/models/SNFEN.weights.h5` | 19,807,920 |
  | `pyfing/models/LEADER.weights.h5` | 4,254,316 |

SBMEX itself is a deterministic algorithm and has no model weights. In this
workflow, SUFS is used only for the fingerprint segmentation mask. SBMEX receives
the prepared binary image and mask directly; pyfing enhancement is not used.

The preprocessing assets are public equivalents of the team's models, not the
team's original models. See [model_sources.md](model_sources.md) for sources,
conversion details, hashes, contract tests, and the Zero-DCE quality caveat.

## Synthetic SBMEX smoke test

Using `KERAS_BACKEND=tensorflow`, pyfing 0.7.3, the default SBMEX parameters,
and a 100x160 black `uint8` image containing a 4-pixel-high white horizontal
ridge from x=30 to x=129 with a full-white segmentation mask:

```text
Minutia(x=32, y=49, direction=0.0, type='E', quality=0.5)
Minutia(x=128, y=49, direction=3.141592653589793, type='E', quality=0.5)
ending_count 2 bifurcation_count 0
```

The two ending directions point along the ridge in opposite directions, as
expected. No SBMEX parameters were changed.
