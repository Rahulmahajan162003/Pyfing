# pyfing 0.7.3 API notes

Source inspected at commit
`df4c0f0d19fd1d8befd9fc0615b71d48d784df03`.

The signatures below are from `pyfing.simple_api`; parameters are not reordered
from the installed API:

```python
pf.fingerprint_segmentation(
    fingerprint: Image, dpi: int = 500, method: str = "SUFS"
) -> Image

pf.frequency_estimation(
    fingerprint: Image,
    orientation_field: np.ndarray,
    segmentation_mask: Image | None = None,
    dpi: int = 500,
    method: str = "SNFFE",
) -> np.ndarray

pf.orientation_field_estimation(
    fingerprint: Image,
    segmentation_mask: Image | None = None,
    dpi: int = 500,
    method: str = "SNFOE",
) -> np.ndarray

pf.minutiae_extraction(
    fingerprint: Image,
    dpi: int = 500,
    method: str = "LEADER",
    segmentation_mask: Image | None = None,
) -> list[Minutia]
```

In this workflow, minutiae are requested with
`pf.minutiae_extraction(image, dpi=500, method="SBMEX", segmentation_mask=seg)`.
The default method in the public API is `LEADER`; the workflow explicitly selects
`SBMEX`. SBMEX expects an enhanced, binarized image. The library's threshold is
strict (`pixel > binarization_threshold` becomes white). Its segmentation mask
must be an 8-bit image with the same dimensions as the input.

## SBMEX class and defaults

The concrete class is `pyfing.Sbmex`, implemented in
`pyfing/minutiae_from_enhanced_image.py`. Its constructor is:

```python
Sbmex(parameters: SbmexParameters | None = None)
```

With no parameter object it instantiates `SbmexParameters()`, whose complete
dataclass defaults are:

```python
SbmexParameters(
    binarization_threshold=0,
    minimum_background_distance=10,
    minimum_tracking_distance=5,
    maximum_tracking_distance=20,
    detection_method="Merge",
    merge_max_distance=6,
    merge_max_direction_difference=np.pi / 6,
    unpaired_minutiae_penalty=0.5,
    valley_minutiae_correction=3,
    minutiae_cluster_sigma=11,
    minutiae_decay_base=0.5,
    minutia_quality_threshold=0.3,
)
```

`detection_method` accepts `"Ridges"`, `"Valleys"`, or `"Merge"` in the
implementation. `"Merge"` runs ridge and valley skeleton passes, pairs close
detections, adjusts unpaired quality, then applies cluster quality decay and
the final quality threshold. The package's test also mentions `"Merge1"` in
intermediate branches, but the final dispatch does not accept it; use only the
documented final choices.

The parameters are left at their pyfing defaults in this workflow.

## Minutia fields and direction

`Minutia` is a `typing.NamedTuple` from `pyfing.definitions`:

```python
Minutia(x: int, y: int, direction: float, type: str, quality: float)
```

- `x`, `y`: pixel coordinates, origin at the top-left.
- `direction`: angle in radians, produced using `atan2(-delta_y, delta_x)`.
  Zero points right; positive angles are counter-clockwise in Cartesian
  coordinates (equivalently, appear clockwise when drawn in image coordinates,
  where y increases downward).
- `type`: `"E"` for ridge ending, `"B"` for bifurcation; the type definition
  also names `"O"` for other, although SBMEX emits endings and bifurcations.
- `quality`: floating-point SBMEX quality after penalties and thresholding.

The simple API returns all detections retained by SBMEX's default quality
threshold. Application code must not add, remove, or alter those minutiae.
