from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required for restoring predictions") from exc


def _read_sitk(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def _get_bounding_boxes(ct_sitk: sitk.Image, pt_sitk: sitk.Image) -> np.ndarray:
    ct_origin = np.array(ct_sitk.GetOrigin())
    pt_origin = np.array(pt_sitk.GetOrigin())

    ct_max = ct_origin + np.array(ct_sitk.GetSize()) * np.array(ct_sitk.GetSpacing())
    pt_max = pt_origin + np.array(pt_sitk.GetSize()) * np.array(pt_sitk.GetSpacing())

    bb_start = np.maximum(ct_origin, pt_origin)
    bb_end = np.minimum(ct_max, pt_max)
    if np.any(bb_end - bb_start <= 0):
        raise ValueError("CT and PET images do not overlap in physical space")
    return np.concatenate([bb_start, bb_end], axis=0)


def _resample_images(
    ct_sitk: sitk.Image,
    pt_sitk: sitk.Image,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[sitk.Image, sitk.Image]:
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkBSpline)

    bb = _get_bounding_boxes(ct_sitk, pt_sitk)
    size = np.round((bb[3:] - bb[:3]) / np.array(spacing)).astype(int)
    if np.any(size <= 0):
        raise ValueError("Invalid resample size computed from overlap bounding box")

    resampler.SetOutputOrigin(bb[:3])
    resampler.SetSize([int(k) for k in size])
    ct_resampled = resampler.Execute(ct_sitk)
    pt_resampled = resampler.Execute(pt_sitk)
    return ct_resampled, pt_resampled


def _resample_ct_only(
    ct_sitk: sitk.Image,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection([1, 0, 0, 0, 1, 0, 0, 0, 1])
    resampler.SetOutputSpacing(spacing)
    resampler.SetInterpolator(sitk.sitkBSpline)

    ct_origin = np.array(ct_sitk.GetOrigin())
    ct_size = np.array(ct_sitk.GetSize())
    ct_spacing = np.array(ct_sitk.GetSpacing())
    ct_max = ct_origin + ct_size * ct_spacing
    size = np.round((ct_max - ct_origin) / np.array(spacing)).astype(int)
    if np.any(size <= 0):
        raise ValueError("Invalid resample size computed from CT extent")

    resampler.SetOutputOrigin(ct_origin)
    resampler.SetSize([int(k) for k in size])
    return resampler.Execute(ct_sitk)


def _register_pet_to_ct(ct_sitk: sitk.Image, pt_sitk: sitk.Image) -> sitk.Image:
    ct = sitk.Cast(ct_sitk, sitk.sitkFloat32)
    pt = sitk.Cast(pt_sitk, sitk.sitkFloat32)

    transform = sitk.CenteredTransformInitializer(
        ct,
        pt,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform)
    final_transform = registration.Execute(ct, pt)
    return sitk.Resample(pt_sitk, ct_sitk, final_transform, sitk.sitkLinear, 0.0, pt_sitk.GetPixelID())


def _paste_crop(
    full_size: tuple[int, int, int],
    roi_start: np.ndarray,
    roi_end: np.ndarray,
    pred_crop: np.ndarray,
) -> np.ndarray:
    full = np.zeros(full_size, dtype=pred_crop.dtype)
    xs, ys, zs = [slice(int(s), int(e)) for s, e in zip(roi_start, roi_end)]
    full[xs, ys, zs] = pred_crop
    return full


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore single-case nnUNet prediction back to raw CT resolution."
    )
    parser.add_argument("--ct", type=Path, required=True, help="CT file path.")
    parser.add_argument("--pt", type=Path, help="PT file path (optional).")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction file path.")
    parser.add_argument("--roi", type=Path, required=True, help="ROI npz path.")
    parser.add_argument("--output", type=Path, required=True, help="Output file path.")
    args = parser.parse_args()

    if not args.ct.is_file():
        raise FileNotFoundError(f"CT not found: {args.ct}")
    if args.pt is not None and not args.pt.is_file():
        raise FileNotFoundError(f"PT not found: {args.pt}")
    if not args.pred.is_file():
        raise FileNotFoundError(f"Prediction not found: {args.pred}")
    if not args.roi.is_file():
        raise FileNotFoundError(f"ROI file not found: {args.roi}")

    ct_raw = _read_sitk(args.ct)
    if args.pt is not None:
        pt_raw = _read_sitk(args.pt)
        try:
            ct_resampled, _ = _resample_images(ct_raw, pt_raw)
        except ValueError:
            pt_registered = _register_pet_to_ct(ct_raw, pt_raw)
            ct_resampled, _ = _resample_images(ct_raw, pt_registered)
    else:
        ct_resampled = _resample_ct_only(ct_raw)

    with np.load(args.roi) as data:
        roi_start = data["roi_start"].astype(int)
        roi_end = data["roi_end"].astype(int)
        crop_applied = bool(data.get("crop_applied", False))

    pred_img = _read_sitk(args.pred)
    pred_np = sitk.GetArrayFromImage(pred_img).astype(np.int16)  # z,y,x
    pred_np = np.transpose(pred_np, (2, 1, 0))  # x,y,z

    if crop_applied:
        full_size = tuple(int(x) for x in ct_resampled.GetSize())
        pred_np = _paste_crop(full_size, roi_start, roi_end, pred_np)

    pred_full = np.transpose(pred_np, (2, 1, 0))  # back to z,y,x
    pred_full_img = sitk.GetImageFromArray(pred_full)
    pred_full_img.CopyInformation(ct_resampled)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_raw)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    pred_raw = resampler.Execute(pred_full_img)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(pred_raw, str(args.output))


if __name__ == "__main__":
    main()
