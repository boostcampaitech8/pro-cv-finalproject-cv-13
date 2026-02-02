from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:
    raise ImportError("SimpleITK is required for preprocessing") from exc

try:
    from skimage.measure import label as cc_label
except ImportError as exc:
    raise ImportError("scikit-image is required for preprocessing") from exc


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


def _get_roi_center(
    pet_xyz: np.ndarray,
    z_top_fraction: float = 0.75,
    z_score_threshold: float = 1.0,
) -> np.ndarray:
    image_shape = np.array(pet_xyz.shape)
    crop_z_start = int(z_top_fraction * image_shape[2])
    top_of_scan = pet_xyz[..., crop_z_start:]

    norm = (top_of_scan - top_of_scan.mean()) / (top_of_scan.std() + 1e-8)
    mask = norm > z_score_threshold
    if not mask.any():
        center_in_top = (np.array(top_of_scan.shape) / 2).astype(int)
    else:
        labeled, num_features = cc_label(mask, return_num=True, connectivity=3)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_label = np.argmax(component_sizes) + 1
            comp_idx = np.argwhere(labeled == largest_label)
        else:
            comp_idx = np.argwhere(mask)
        center_in_top = np.mean(comp_idx, axis=0)

    center_full = center_in_top + np.array([0, 0, crop_z_start])
    return center_full.astype(int)


def _compute_roi_box(
    pt_sitk: sitk.Image,
    crop_box_size: tuple[int, int, int] = (200, 200, 310),
    z_top_fraction: float = 0.75,
    z_score_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    pet_zyx = sitk.GetArrayFromImage(pt_sitk)
    pet_xyz = np.transpose(pet_zyx, (2, 1, 0))

    crop_box_size = np.asarray(crop_box_size, dtype=int)
    center = _get_roi_center(
        pet_xyz,
        z_top_fraction=z_top_fraction,
        z_score_threshold=z_score_threshold,
    )
    img_shape = np.asarray(pet_xyz.shape)
    box_start = np.clip(center - crop_box_size // 2, 0, img_shape)
    box_end = np.clip(box_start + crop_box_size, 0, img_shape)
    box_start = np.maximum(box_end - crop_box_size, 0)
    return box_start, box_end


def _crop_roi_pair(
    ct_sitk: sitk.Image,
    pt_sitk: sitk.Image,
    roi_start: np.ndarray,
    roi_end: np.ndarray,
) -> tuple[sitk.Image, sitk.Image]:
    index = [int(i) for i in roi_start.tolist()]
    size = [int(e - s) for s, e in zip(roi_start.tolist(), roi_end.tolist())]
    ct_crop = sitk.RegionOfInterest(ct_sitk, size=size, index=index)
    pt_crop = sitk.RegionOfInterest(pt_sitk, size=size, index=index)
    return ct_crop, pt_crop


def _infer_case_id_from_ct(ct_path: Path) -> str:
    name = ct_path.name
    if name.endswith("__CT.nii.gz"):
        return name[: -len("__CT.nii.gz")]
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        if stem.endswith("_0000"):
            stem = stem[: -len("_0000")]
        return stem
    return ct_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess single-case Hecktor25 CT/PT into nnUNet inference inputs."
    )
    parser.add_argument("--ct", type=Path, required=True, help="CT file path.")
    parser.add_argument("--pt", type=Path, help="PT file path (optional).")
    parser.add_argument("--case-id", type=str, help="Case id (optional).")
    parser.add_argument("--output-images", type=Path, required=True, help="Output images folder.")
    parser.add_argument("--roi-dir", type=Path, required=True, help="ROI npz folder.")
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Apply fixed ROI crop (200x200x310) after resampling.",
    )
    parser.add_argument(
        "--crop-box-size",
        type=int,
        nargs=3,
        default=(200, 200, 310),
        help="Fixed crop size (x y z). Default: 200 200 310.",
    )
    parser.add_argument(
        "--z-top-fraction",
        type=float,
        default=0.75,
        help="Top portion fraction for PET ROI search. Default: 0.75.",
    )
    parser.add_argument(
        "--z-score-threshold",
        type=float,
        default=1.0,
        help="Z-score threshold for PET ROI mask. Default: 1.0.",
    )
    args = parser.parse_args()

    if not args.ct.is_file():
        raise FileNotFoundError(f"CT not found: {args.ct}")
    ct_raw = _read_sitk(args.ct)
    pt_raw = _read_sitk(args.pt) if args.pt is not None else None

    if pt_raw is not None:
        try:
            ct_resampled, pt_resampled = _resample_images(ct_raw, pt_raw)
        except ValueError:
            pt_registered = _register_pet_to_ct(ct_raw, pt_raw)
            ct_resampled, pt_resampled = _resample_images(ct_raw, pt_registered)

        roi_start, roi_end = _compute_roi_box(
            pt_resampled,
            crop_box_size=tuple(args.crop_box_size),
            z_top_fraction=args.z_top_fraction,
            z_score_threshold=args.z_score_threshold,
        )

        if args.crop:
            ct_save, pt_save = _crop_roi_pair(ct_resampled, pt_resampled, roi_start, roi_end)
            crop_applied = True
        else:
            ct_save, pt_save = ct_resampled, pt_resampled
            crop_applied = False
    else:
        ct_save = _resample_ct_only(ct_raw)
        pt_save = None
        roi_start = np.zeros(3, dtype=int)
        roi_end = np.array(ct_save.GetSize(), dtype=int)
        crop_applied = False

    case_id = args.case_id or _infer_case_id_from_ct(args.ct)

    args.output_images.mkdir(parents=True, exist_ok=True)
    args.roi_dir.mkdir(parents=True, exist_ok=True)

    ct_out = args.output_images / f"{case_id}_0000.nii.gz"
    sitk.WriteImage(ct_save, str(ct_out))
    if pt_save is not None:
        pt_out = args.output_images / f"{case_id}_0001.nii.gz"
        sitk.WriteImage(pt_save, str(pt_out))

    np.savez_compressed(
        args.roi_dir / f"{case_id}_ROI.npz",
        roi_start=roi_start.astype(np.int16),
        roi_end=roi_end.astype(np.int16),
        crop_applied=np.array(bool(crop_applied)),
    )


if __name__ == "__main__":
    main()
