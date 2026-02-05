"""Shared color definitions for anatomical structures.

Single source of truth for structure colors used across:
- dicom_converter.py (DICOM Series)
- rtss_generator.py (RT Structure Set)
- surface_segmentation_generator.py (3D Surface Meshes)
- nifti_generator.py (NIfTI Labelmap)
- nerve_to_dicom.py (Nerve DICOM contours)
"""

from typing import Dict, Tuple


STRUCTURE_COLORS: Dict[str, Tuple[int, int, int]] = {
    # ===== Nerves =====
    "vagus_nerve": (255, 255, 0),
    "vagus": (255, 255, 0),
    "left_vagus_nerve": (255, 255, 0),
    "right_vagus_nerve": (255, 255, 0),
    "left_vagus": (255, 255, 0),
    "right_vagus": (255, 255, 0),

    "phrenic_nerve": (0, 255, 0),
    "phrenic": (0, 255, 0),
    "left_phrenic_nerve": (0, 255, 0),
    "right_phrenic_nerve": (0, 255, 0),
    "left_phrenic": (0, 255, 0),
    "right_phrenic": (0, 255, 0),

    "recurrent_laryngeal_nerve": (255, 105, 180),
    "rln": (255, 105, 180),
    "rlnl": (255, 105, 180),
    "rlnr": (255, 105, 180),
    "left_rln": (255, 105, 180),
    "right_rln": (255, 105, 180),
    "left_recurrent_laryngeal_nerve": (255, 105, 180),
    "right_recurrent_laryngeal_nerve": (255, 105, 180),

    "ebsln": (127, 255, 212),
    "external_branch_superior_laryngeal_nerve": (127, 255, 212),
    "superior_laryngeal_nerve": (127, 255, 212),
    "left_ebsln": (127, 255, 212),
    "right_ebsln": (127, 255, 212),

    "brachial_plexus": (255, 128, 0),
    "accessory_nerve": (255, 150, 50),
    "hypoglossal_nerve": (255, 190, 80),
    "sympathetic_trunk": (255, 210, 120),

    # ===== Tumors =====
    "tumor": (0, 0, 0),
    "tumour": (0, 0, 0),
    "gtv": (0, 0, 0),
    "tumor_gtv": (0, 0, 0),
    "lesion": (0, 0, 0),
    "mass": (0, 0, 0),

    # ===== Airways =====
    "trachea": (0, 0, 128),
    "bronchus": (0, 0, 160),

    # ===== Digestive =====
    "esophagus": (255, 165, 0),

    # ===== Glands =====
    "thyroid": (128, 0, 128),
    "thyroid_gland": (128, 0, 128),
    "thyroid_l": (128, 0, 128),
    "thyroid_r": (128, 0, 128),
    "left_thyroid": (128, 0, 128),
    "right_thyroid": (128, 0, 128),

    # ===== Blood Vessels =====
    "carotid": (255, 0, 0),
    "carotid_artery": (255, 0, 0),
    "common_carotid_artery": (255, 0, 0),
    "cca": (255, 0, 0),
    "carotid_l": (255, 0, 0),
    "carotid_r": (255, 0, 0),
    "left_common_carotid_artery": (255, 0, 0),
    "right_common_carotid_artery": (255, 0, 0),
    "common_carotid_artery_left": (255, 0, 0),
    "common_carotid_artery_right": (255, 0, 0),
    "carotid_artery_left": (255, 0, 0),
    "carotid_artery_right": (255, 0, 0),

    "jugular": (0, 0, 255),
    "jugular_vein": (0, 0, 255),
    "internal_jugular_vein": (0, 0, 255),
    "ijv": (0, 0, 255),
    "jugular_l": (0, 0, 255),
    "jugular_r": (0, 0, 255),
    "left_internal_jugular_vein": (0, 0, 255),
    "right_internal_jugular_vein": (0, 0, 255),
    "internal_jugular_vein_left": (0, 0, 255),
    "internal_jugular_vein_right": (0, 0, 255),
    "ijv_left": (0, 0, 255),
    "ijv_right": (0, 0, 255),
    "jugular_vein_left": (0, 0, 255),
    "jugular_vein_right": (0, 0, 255),

    "aorta": (255, 99, 71),
    "brachiocephalic_trunk": (255, 127, 80),
    "brachiocephalic_vein_l": (70, 130, 180),
    "brachiocephalic_vein_r": (70, 130, 180),
    "brachiocephalic_vein_left": (70, 130, 180),
    "brachiocephalic_vein_right": (70, 130, 180),
    "subclavian_artery": (200, 0, 0),
    "subclavian_artery_l": (200, 0, 0),
    "subclavian_artery_r": (200, 0, 0),
    "subclavian_artery_left": (200, 0, 0),
    "subclavian_artery_right": (200, 0, 0),
    "subclavian_vein": (0, 0, 200),
    "subclavian_vein_l": (0, 0, 200),
    "subclavian_vein_r": (0, 0, 200),
    "subclavian_vein_left": (0, 0, 200),
    "subclavian_vein_right": (0, 0, 200),

    # ===== Muscles =====
    "anterior_scalene": (139, 69, 19),
    "anterior_scalene_muscle": (139, 69, 19),
    "anterior_scalene_left": (139, 69, 19),
    "anterior_scalene_right": (139, 69, 19),
    "anterior_scalene_muscle_left": (139, 69, 19),
    "anterior_scalene_muscle_right": (139, 69, 19),
    "middle_scalene": (160, 82, 45),
    "posterior_scalene": (180, 100, 60),
    "sternocleidomastoid": (210, 180, 140),
    "sternocleidomastoid_l": (210, 180, 140),
    "sternocleidomastoid_r": (210, 180, 140),
    "sternocleidomastoid_left": (210, 180, 140),
    "sternocleidomastoid_right": (210, 180, 140),

    # ===== Bones =====
    "hyoid": (245, 222, 179),

    # ===== Vertebrae =====
    "spine": (245, 245, 220),
    "spinal_cord": (245, 245, 220),
    "vertebrae": (245, 245, 220),
    "vertebra": (245, 245, 220),
    "c1": (245, 245, 220),
    "c2": (245, 245, 220),
    "c3": (245, 245, 220),
    "c4": (245, 245, 220),
    "c5": (245, 245, 220),
    "c6": (245, 245, 220),
    "c7": (245, 245, 220),
    "t1": (245, 245, 220),
    "cervical_vertebra": (245, 245, 220),
    "thoracic_vertebra": (245, 245, 220),
    "vertebrae_c1": (245, 245, 220),
    "vertebrae_c2": (245, 245, 220),
    "vertebrae_c3": (245, 245, 220),
    "vertebrae_c4": (245, 245, 220),
    "vertebrae_c5": (245, 245, 220),
    "vertebrae_c6": (245, 245, 220),
    "vertebrae_c7": (245, 245, 220),
    "vertebrae_t1": (245, 245, 220),

    # ===== Lungs =====
    "lung_l": (144, 238, 144),
    "lung_r": (144, 238, 144),
    "lung_left": (144, 238, 144),
    "lung_right": (144, 238, 144),
    "heart": (178, 34, 34),

    # ===== Lymph nodes =====
    "lymphnode": (124, 252, 0),
    "lymph_node": (124, 252, 0),

    # Default
    "default": (128, 128, 128),
}


NIFTI_STRUCTURE_COLORS_RGBA: Dict[str, Tuple[int, int, int, int]] = {
    # Nerves (alpha=200)
    "vagus_nerve": (255, 255, 0, 200),
    "vagus": (255, 255, 0, 200),
    "left_vagus_nerve": (255, 255, 0, 200),
    "right_vagus_nerve": (255, 255, 0, 200),
    "left_vagus": (255, 255, 0, 200),
    "right_vagus": (255, 255, 0, 200),
    "nerve_left_vagus": (255, 255, 0, 200),
    "nerve_right_vagus": (255, 255, 0, 200),

    "phrenic_nerve": (0, 255, 0, 200),
    "phrenic": (0, 255, 0, 200),
    "left_phrenic_nerve": (0, 255, 0, 200),
    "right_phrenic_nerve": (0, 255, 0, 200),
    "left_phrenic": (0, 255, 0, 200),
    "right_phrenic": (0, 255, 0, 200),
    "nerve_left_phrenic": (0, 255, 0, 200),
    "nerve_right_phrenic": (0, 255, 0, 200),

    "recurrent_laryngeal_nerve": (255, 105, 180, 200),
    "rln": (255, 105, 180, 200),
    "rlnl": (255, 105, 180, 200),
    "rlnr": (255, 105, 180, 200),
    "left_rln": (255, 105, 180, 200),
    "right_rln": (255, 105, 180, 200),
    "nerve_left_rln": (255, 105, 180, 200),
    "nerve_right_rln": (255, 105, 180, 200),

    "ebsln": (127, 255, 212, 200),
    "left_ebsln": (127, 255, 212, 200),
    "right_ebsln": (127, 255, 212, 200),
    "nerve_left_ebsln": (127, 255, 212, 200),
    "nerve_right_ebsln": (127, 255, 212, 200),

    # Tumors (alpha=200, dark gray for NIfTI display)
    "tumor": (50, 50, 50, 200),
    "tumour": (50, 50, 50, 200),
    "gtv": (50, 50, 50, 200),

    # Airways (alpha=150, bright blue for NIfTI display)
    "trachea": (0, 191, 255, 150),
    "bronchus": (0, 170, 230, 150),

    # Digestive (alpha=150)
    "esophagus": (255, 165, 0, 150),

    # Glands (alpha=150)
    "thyroid": (128, 0, 128, 150),
    "thyroid_gland": (128, 0, 128, 150),

    # Blood Vessels (alpha=150)
    "carotid_artery_left": (255, 0, 0, 150),
    "carotid_artery_right": (255, 0, 0, 150),
    "common_carotid_artery_left": (255, 0, 0, 150),
    "common_carotid_artery_right": (255, 0, 0, 150),
    "jugular_vein_left": (0, 0, 255, 150),
    "jugular_vein_right": (0, 0, 255, 150),
    "internal_jugular_vein_left": (0, 0, 255, 150),
    "internal_jugular_vein_right": (0, 0, 255, 150),
    "ijv_left": (0, 0, 255, 150),
    "ijv_right": (0, 0, 255, 150),
    "aorta": (255, 99, 71, 150),
    "brachiocephalic_trunk": (255, 127, 80, 150),
    "brachiocephalic_vein_left": (70, 130, 180, 150),
    "brachiocephalic_vein_right": (70, 130, 180, 150),
    "subclavian_artery_left": (200, 0, 0, 150),
    "subclavian_artery_right": (200, 0, 0, 150),

    # Muscles (alpha=150)
    "anterior_scalene_muscle_left": (139, 69, 19, 150),
    "anterior_scalene_muscle_right": (139, 69, 19, 150),
    "anterior_scalene_left": (139, 69, 19, 150),
    "anterior_scalene_right": (139, 69, 19, 150),
    "sternocleidomastoid_left": (210, 180, 140, 150),
    "sternocleidomastoid_right": (210, 180, 140, 150),

    # Bones (alpha=100)
    "hyoid": (245, 222, 179, 100),

    # Vertebrae (alpha=100)
    "vertebrae_C1": (245, 245, 220, 100),
    "vertebrae_C2": (245, 245, 220, 100),
    "vertebrae_C3": (245, 245, 220, 100),
    "vertebrae_C4": (245, 245, 220, 100),
    "vertebrae_C5": (245, 245, 220, 100),
    "vertebrae_C6": (245, 245, 220, 100),
    "vertebrae_C7": (245, 245, 220, 100),
    "vertebrae_T1": (245, 245, 220, 100),

    # Lungs (alpha=100)
    "lung_left": (144, 238, 144, 100),
    "lung_right": (144, 238, 144, 100),

    # Heart (alpha=150)
    "heart": (178, 34, 34, 150),
}


NERVE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "vagus_nerve": (255, 255, 0),
    "vagus": (255, 255, 0),
    "vagus_left": (255, 255, 0),
    "vagus_right": (255, 255, 0),
    "left_vagus_nerve": (255, 255, 0),
    "right_vagus_nerve": (255, 255, 0),
    "left_vagus": (255, 255, 0),
    "right_vagus": (255, 255, 0),

    "phrenic_nerve": (0, 255, 0),
    "phrenic": (0, 255, 0),
    "phrenic_left": (0, 255, 0),
    "phrenic_right": (0, 255, 0),
    "left_phrenic_nerve": (0, 255, 0),
    "right_phrenic_nerve": (0, 255, 0),
    "left_phrenic": (0, 255, 0),
    "right_phrenic": (0, 255, 0),

    "recurrent_laryngeal_nerve": (255, 105, 180),
    "rln": (255, 105, 180),
    "rln_left": (255, 105, 180),
    "rln_right": (255, 105, 180),
    "left_rln": (255, 105, 180),
    "right_rln": (255, 105, 180),
    "left_recurrent_laryngeal_nerve": (255, 105, 180),
    "right_recurrent_laryngeal_nerve": (255, 105, 180),

    "ebsln": (127, 255, 212),
    "ebsln_left": (127, 255, 212),
    "ebsln_right": (127, 255, 212),
    "external_branch_superior_laryngeal_nerve": (127, 255, 212),
    "superior_laryngeal_nerve": (127, 255, 212),
    "left_ebsln": (127, 255, 212),
    "right_ebsln": (127, 255, 212),

    "brachial_plexus": (255, 180, 100),
    "accessory_nerve": (255, 150, 50),
    "hypoglossal_nerve": (255, 190, 80),
    "sympathetic_trunk": (255, 210, 120),

    "danger_zone": (255, 0, 0),
    "critical_zone": (255, 0, 255),
}


def get_structure_color(name: str) -> Tuple[int, int, int]:
    """Get RGB color for anatomical structure by name."""
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    return STRUCTURE_COLORS.get(normalized, STRUCTURE_COLORS["default"])


def normalize_structure_name(name: str) -> str:
    """Normalize structure name from file paths."""
    name = name.lower().replace(".nii", "").replace(".gz", "")
    if name.startswith("nerve_"):
        name = name[6:]
    return name


def get_color_for_structure_rgba(name: str) -> Tuple[int, int, int, int]:
    """Get RGBA color for structure (NIfTI labelmap display)."""
    normalized = normalize_structure_name(name)

    colors_lower = {k.lower(): v for k, v in NIFTI_STRUCTURE_COLORS_RGBA.items()}
    if normalized in colors_lower:
        return colors_lower[normalized]

    for key, color in colors_lower.items():
        if key in normalized or normalized in key:
            return color

    return (128, 128, 128, 150)


def get_nerve_color(nerve_name: str) -> Tuple[int, int, int]:
    """Get RGB color for nerve, handling various naming patterns.

    Handles: nerve_vagus_left_core, nerve_rln_right_uncertainty, etc.
    """
    name_lower = nerve_name.lower().replace(" ", "_").replace("-", "_")

    if name_lower in NERVE_COLORS:
        return NERVE_COLORS[name_lower]

    clean_name = name_lower
    if clean_name.startswith("nerve_"):
        clean_name = clean_name[6:]

    for suffix in ["_core", "_uncertainty", "_zone"]:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]

    for side in ["_left", "_right"]:
        if clean_name.endswith(side):
            base_name = clean_name[:-len(side)]
            if base_name in NERVE_COLORS:
                return NERVE_COLORS[base_name]
            side_prefix = side[1:] + "_" + base_name
            if side_prefix in NERVE_COLORS:
                return NERVE_COLORS[side_prefix]

    if clean_name in NERVE_COLORS:
        return NERVE_COLORS[clean_name]

    return (255, 200, 0)
