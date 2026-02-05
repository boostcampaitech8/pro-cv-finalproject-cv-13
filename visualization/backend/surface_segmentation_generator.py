"""
DICOM Surface Segmentation Generator

NIfTI 마스크에서 3D mesh를 추출하여 DICOM Surface Segmentation 파일 생성.
marching_cubes로 3D surface mesh 생성 → OHIF 3D 뷰에서 렌더링 가능.
RTSS(2D contour)와 함께 사용하면 2D/3D 동시 시각화 가능.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from skimage import measure
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime

try:
    from colors import STRUCTURE_COLORS
except ImportError:
    from .colors import STRUCTURE_COLORS


# DICOM Surface Segmentation SOP Class UID
SURFACE_SEGMENTATION_STORAGE = "1.2.840.10008.5.1.4.1.1.66.5"


def extract_mesh_from_mask(
    mask_data: np.ndarray,
    affine: np.ndarray,
    decimate_ratio: float = 0.5,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D mesh from binary mask using marching cubes.

    Args:
        mask_data: 3D binary mask (X, Y, Z)
        affine: NIfTI affine matrix for coordinate transformation
        decimate_ratio: Ratio for mesh decimation (0.1-1.0, lower = smaller file)
        smooth: Apply Gaussian smoothing before mesh extraction

    Returns:
        vertices: Nx3 array of vertex coordinates in DICOM space (LPS)
        faces: Mx3 array of triangle indices
        normals: Nx3 array of vertex normals
    """
    if smooth:
        from scipy.ndimage import gaussian_filter
        mask_data = gaussian_filter(mask_data.astype(float), sigma=0.5)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            mask_data,
            level=0.5,
            step_size=1,
            allow_degenerate=False
        )
    except Exception as e:
        print(f"[SurfaceSeg] marching_cubes failed: {e}")
        return np.array([]), np.array([]), np.array([])

    if len(verts) == 0:
        return np.array([]), np.array([]), np.array([])

    # Transform vertices from voxel to world coordinates
    # NIfTI uses RAS, DICOM uses LPS
    ones = np.ones((len(verts), 1))
    verts_h = np.hstack([verts, ones])
    verts_world = (affine @ verts_h.T).T[:, :3]

    # Convert from NIfTI RAS to DICOM LPS (negate x and y)
    verts_lps = verts_world.copy()
    verts_lps[:, 0] = -verts_lps[:, 0]  # R -> L
    verts_lps[:, 1] = -verts_lps[:, 1]  # A -> P
    # S stays the same

    # Mesh decimation to reduce file size
    if decimate_ratio < 1.0 and len(faces) > 100:
        try:
            # Try using trimesh for decimation
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts_lps, faces=faces)
            target_faces = int(len(faces) * decimate_ratio)
            if target_faces > 10:
                mesh_decimated = mesh.simplify_quadric_decimation(target_faces)
                verts_lps = mesh_decimated.vertices
                faces = mesh_decimated.faces
                # Recompute normals after decimation
                normals = mesh_decimated.vertex_normals
                print(f"[SurfaceSeg] Decimated mesh: {len(faces)} faces")
        except ImportError:
            print("[SurfaceSeg] trimesh not available, skipping decimation")
        except Exception as e:
            print(f"[SurfaceSeg] Decimation failed: {e}")

    return verts_lps.astype(np.float32), faces.astype(np.int32), normals.astype(np.float32)


def create_surface_segmentation_dataset(
    reference_ct_dir: str,
    surfaces: Dict[str, Dict[str, Any]],
    study_uid: str = None,
    patient_name: str = "Anonymous",
) -> Dataset:
    """
    Create a DICOM Surface Segmentation dataset.

    Args:
        reference_ct_dir: CT DICOM series directory
        surfaces: {
            "structure_name": {
                "vertices": Nx3 array,
                "faces": Mx3 array,
                "normals": Nx3 array,
                "color": (R, G, B),
            }
        }
        study_uid: StudyInstanceUID (None to get from reference)
        patient_name: Patient name

    Returns:
        pydicom Dataset
    """
    ct_files = sorted(Path(reference_ct_dir).glob("*.dcm"))
    if not ct_files:
        raise ValueError(f"No DICOM files in {reference_ct_dir}")

    ref_ds = pydicom.dcmread(str(ct_files[0]))

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SURFACE_SEGMENTATION_STORAGE
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    # Patient module
    ds.PatientName = patient_name
    ds.PatientID = getattr(ref_ds, 'PatientID', generate_uid()[-16:])
    ds.PatientBirthDate = getattr(ref_ds, 'PatientBirthDate', '')
    ds.PatientSex = getattr(ref_ds, 'PatientSex', '')

    # Study module
    ds.StudyInstanceUID = study_uid or ref_ds.StudyInstanceUID
    ds.StudyDate = getattr(ref_ds, 'StudyDate', datetime.date.today().strftime('%Y%m%d'))
    ds.StudyTime = getattr(ref_ds, 'StudyTime', datetime.datetime.now().strftime('%H%M%S'))
    ds.StudyDescription = getattr(ref_ds, 'StudyDescription', 'Nerve Assessment')
    ds.AccessionNumber = getattr(ref_ds, 'AccessionNumber', '')
    ds.ReferringPhysicianName = ''
    ds.StudyID = getattr(ref_ds, 'StudyID', '1')

    # Series module
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 998
    ds.SeriesDescription = "Surface Segmentation - Nerve Assessment 3D"
    ds.Modality = "SEG"

    # Instance module
    ds.SOPClassUID = SURFACE_SEGMENTATION_STORAGE
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceCreationDate = datetime.date.today().strftime('%Y%m%d')
    ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
    ds.InstanceNumber = 1

    # Frame of Reference
    ds.FrameOfReferenceUID = getattr(ref_ds, 'FrameOfReferenceUID', generate_uid())

    # Content module
    ds.ContentDate = ds.InstanceCreationDate
    ds.ContentTime = ds.InstanceCreationTime
    ds.ContentLabel = "SURFACE_SEGMENTATION"
    ds.ContentDescription = "3D Surface Segmentation for Nerve Assessment"
    ds.ContentCreatorName = "NerveAssessment"

    # Referenced Series Sequence (reference the CT series)
    ref_series_seq = Sequence()
    ref_series_item = Dataset()
    ref_series_item.SeriesInstanceUID = ref_ds.SeriesInstanceUID

    # Referenced Instance Sequence (reference all CT slices)
    ref_instance_seq = Sequence()
    for ct_file in ct_files:
        ct_ds = pydicom.dcmread(str(ct_file), stop_before_pixels=True)
        ref_instance_item = Dataset()
        ref_instance_item.ReferencedSOPClassUID = ct_ds.SOPClassUID
        ref_instance_item.ReferencedSOPInstanceUID = ct_ds.SOPInstanceUID
        ref_instance_seq.append(ref_instance_item)

    ref_series_item.ReferencedInstanceSequence = ref_instance_seq
    ref_series_seq.append(ref_series_item)
    ds.ReferencedSeriesSequence = ref_series_seq

    # Segment Sequence
    segment_seq = Sequence()
    surface_seq = Sequence()

    for seg_num, (name, data) in enumerate(surfaces.items(), start=1):
        vertices = data.get("vertices", np.array([]))
        faces = data.get("faces", np.array([]))
        normals = data.get("normals", np.array([]))
        color = data.get("color", (255, 0, 0))

        if len(vertices) == 0 or len(faces) == 0:
            continue

        segment_item = Dataset()
        segment_item.SegmentNumber = seg_num
        segment_item.SegmentLabel = name
        segment_item.SegmentDescription = f"3D surface mesh for {name}"
        segment_item.SegmentAlgorithmType = "AUTOMATIC"
        segment_item.SegmentAlgorithmName = "NerveAssessment"

        # Recommended Display CIELab Value (convert RGB to CIELab approximation)
        # Simple approximation: scale RGB to 16-bit CIELab range
        segment_item.RecommendedDisplayCIELabValue = [
            int(color[0] * 257),  # L
            int(color[1] * 257),  # a
            int(color[2] * 257),  # b
        ]

        # Segmented Property Category Code Sequence
        prop_cat_seq = Sequence()
        prop_cat_item = Dataset()
        prop_cat_item.CodeValue = "T-D0050"
        prop_cat_item.CodingSchemeDesignator = "SRT"
        prop_cat_item.CodeMeaning = "Tissue"
        prop_cat_seq.append(prop_cat_item)
        segment_item.SegmentedPropertyCategoryCodeSequence = prop_cat_seq

        # Segmented Property Type Code Sequence
        prop_type_seq = Sequence()
        prop_type_item = Dataset()
        prop_type_item.CodeValue = "T-D0050"
        prop_type_item.CodingSchemeDesignator = "SRT"
        prop_type_item.CodeMeaning = name
        prop_type_seq.append(prop_type_item)
        segment_item.SegmentedPropertyTypeCodeSequence = prop_type_seq

        segment_seq.append(segment_item)

        surface_item = Dataset()
        surface_item.SurfaceNumber = seg_num
        surface_item.SurfaceComments = f"Surface mesh for {name}"
        surface_item.SurfaceProcessing = "NO"
        surface_item.RecommendedPresentationType = "SURFACE"
        surface_item.RecommendedPresentationOpacity = 1.0
        surface_item.RecommendedDisplayGrayscaleValue = 65535

        # Recommended Display CIELab Value for surface
        surface_item.RecommendedDisplayCIELabValue = segment_item.RecommendedDisplayCIELabValue

        # Finite Volume - YES means closed surface
        surface_item.FiniteVolume = "YES"
        surface_item.Manifold = "YES"

        # Surface Points Sequence (vertices)
        points_seq = Sequence()
        points_item = Dataset()
        points_item.NumberOfSurfacePoints = len(vertices)
        # Flatten vertices to [x1, y1, z1, x2, y2, z2, ...]
        points_item.PointCoordinatesData = vertices.flatten().astype(np.float32).tobytes()
        points_seq.append(points_item)
        surface_item.SurfacePointsSequence = points_seq

        # Surface Points Normals Sequence (optional but helps rendering)
        if len(normals) == len(vertices):
            normals_seq = Sequence()
            normals_item = Dataset()
            normals_item.NumberOfVectors = len(normals)
            normals_item.VectorDimensionality = 3
            normals_item.VectorCoordinateData = normals.flatten().astype(np.float32).tobytes()
            normals_seq.append(normals_item)
            surface_item.SurfacePointsNormalsSequence = normals_seq

        # Surface Mesh Primitives Sequence (triangles)
        mesh_seq = Sequence()
        mesh_item = Dataset()

        # Triangle Point Index List (1-indexed for DICOM)
        # Each triangle has 3 vertex indices
        triangle_indices = (faces + 1).flatten().astype(np.uint32)  # 1-indexed
        mesh_item.LongTrianglePointIndexList = triangle_indices.tobytes()
        mesh_item.NumberOfTriangles = len(faces)
        mesh_seq.append(mesh_item)
        surface_item.SurfaceMeshPrimitivesSequence = mesh_seq

        # Referenced Segment Number
        ref_seg_seq = Sequence()
        ref_seg_item = Dataset()
        ref_seg_item.ReferencedSegmentNumber = seg_num
        ref_seg_seq.append(ref_seg_item)
        surface_item.ReferencedSurfaceSequence = ref_seg_seq

        surface_seq.append(surface_item)

    ds.SegmentSequence = segment_seq
    ds.SurfaceSequence = surface_seq
    ds.NumberOfSurfaces = len(surface_seq)

    return ds


def nifti_masks_to_surface_segmentation(
    mask_files: Dict[str, str],
    reference_ct_dir: str,
    output_path: str,
    colors: Dict[str, Tuple[int, int, int]] = None,
    study_uid: str = None,
    patient_name: str = "Anonymous",
    decimate_ratio: float = 0.3,
) -> str:
    """
    Convert multiple NIfTI masks to a single DICOM Surface Segmentation file.

    Args:
        mask_files: {"structure_name": "/path/to/mask.nii.gz"}
        reference_ct_dir: CT DICOM series directory
        output_path: Output Surface Segmentation file path
        colors: {"structure_name": (R, G, B)} color map
        study_uid: StudyInstanceUID
        patient_name: Patient name
        decimate_ratio: Mesh decimation ratio (0.1-1.0, lower = smaller file)

    Returns:
        Generated Surface Segmentation file path
    """
    default_colors = STRUCTURE_COLORS
    surfaces = {}

    total_vertices = 0
    total_faces = 0

    for name, mask_path in mask_files.items():
        try:
            nii = nib.load(mask_path)
            mask_data = nii.get_fdata()
            affine = nii.affine

            mask_data = (mask_data > 0).astype(np.float32)

            if not np.any(mask_data > 0):
                print(f"[SurfaceSeg] Skipping empty mask: {name}")
                continue

            vertices, faces, normals = extract_mesh_from_mask(
                mask_data, affine, decimate_ratio=decimate_ratio
            )

            if len(vertices) > 0 and len(faces) > 0:
                color = (255, 0, 0)
                name_lower = name.lower()

                if colors and name in colors:
                    color = colors[name]
                elif colors and name_lower in colors:
                    color = colors[name_lower]
                elif name in default_colors:
                    color = default_colors[name]
                elif name_lower in default_colors:
                    color = default_colors[name_lower]

                surfaces[name] = {
                    "vertices": vertices,
                    "faces": faces,
                    "normals": normals,
                    "color": color,
                }

                total_vertices += len(vertices)
                total_faces += len(faces)
                print(f"[SurfaceSeg] Extracted mesh from {name}: {len(vertices)} vertices, {len(faces)} faces")
            else:
                print(f"[SurfaceSeg] No valid mesh from {name}")

        except Exception as e:
            print(f"[SurfaceSeg] Error processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not surfaces:
        raise ValueError("No valid meshes extracted from any masks")

    print(f"[SurfaceSeg] Total: {len(surfaces)} surfaces, {total_vertices} vertices, {total_faces} faces")

    ds = create_surface_segmentation_dataset(
        reference_ct_dir=reference_ct_dir,
        surfaces=surfaces,
        study_uid=study_uid,
        patient_name=patient_name,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(output_path, write_like_original=False)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"[SurfaceSeg] Created Surface Segmentation file: {output_path} ({file_size_mb:.1f} MB)")

    return output_path


if __name__ == "__main__":
    # Test example
    import sys

    if len(sys.argv) < 4:
        print("Usage: python surface_segmentation_generator.py <mask_dir> <ct_dir> <output.dcm>")
        print("  mask_dir: Directory containing NIfTI masks")
        print("  ct_dir: Directory containing CT DICOM files")
        print("  output.dcm: Output Surface Segmentation file")
        sys.exit(1)

    mask_dir = Path(sys.argv[1])
    ct_dir = sys.argv[2]
    output_path = sys.argv[3]

    mask_files = {}
    for mask_path in mask_dir.glob("*.nii.gz"):
        name = mask_path.stem.replace(".nii", "")
        mask_files[name] = str(mask_path)

    if not mask_files:
        print(f"No NIfTI masks found in {mask_dir}")
        sys.exit(1)

    print(f"Found {len(mask_files)} masks: {list(mask_files.keys())}")

    result = nifti_masks_to_surface_segmentation(
        mask_files=mask_files,
        reference_ct_dir=ct_dir,
        output_path=output_path,
        decimate_ratio=0.3,
    )

    print(f"Created: {result}")
