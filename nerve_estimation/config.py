"""신경 추정 설정 및 상수."""

from typing import Dict, Any

NERVE_CONFIG: Dict[str, Dict[str, Any]] = {
    "vagus": {
        "uncertainty_mm": 5.0,
        "method": "midpoint_posterior",
        "reference": "Inamura et al. 2017",
        "posterior_offset_mm": 2.04,
        "required_structures": ["common_carotid_artery", "internal_jugular_vein"],
        "output_type": "pathway",
    },
    "ebsln": {
        "uncertainty_mm": 6.0,
        "method": "superior_pole_offset",
        "reference": "Estrela et al. 2011 (PMID:21537628)",
        "superior_offset_mm": 7.68,
        "required_structures": ["thyroid_gland"],
        "output_type": "danger_zone",
    },
    "rln": {
        "uncertainty_mm": 5.0,
        "method": "teg_midpoint",
        "reference": "Baseline implementation",
        "required_structures": ["trachea", "esophagus"],
        "output_type": "pathway",
    },
    "phrenic": {
        "uncertainty_mm": 8.0,
        "method": "anterior_surface",
        "reference": "Baseline implementation",
        "required_structures": ["anterior_scalene"],
        "output_type": "pathway",
    },
}

RISK_THRESHOLDS: Dict[str, float] = {
    "high": 5.0,
    "moderate": 10.0,
}

MASK_ALIASES: Dict[str, str] = {
    "jugular_vein_left": "internal_jugular_vein_left",
    "jugular_vein_right": "internal_jugular_vein_right",
    "scalenus_anterior_left": "anterior_scalene_left",
    "scalenus_anterior_right": "anterior_scalene_right",
    "ijv_left": "internal_jugular_vein_left",
    "ijv_right": "internal_jugular_vein_right",
    "cca_left": "common_carotid_artery_left",
    "cca_right": "common_carotid_artery_right",
    "ica_left": "internal_carotid_artery_left",
    "ica_right": "internal_carotid_artery_right",
    "ICA_left": "internal_carotid_artery_left",
    "ICA_right": "internal_carotid_artery_right",
    "IJV_left": "internal_jugular_vein_left",
    "IJV_right": "internal_jugular_vein_right",
    "Anterior_scalene_left": "anterior_scalene_left",
    "Anterior_scalene_right": "anterior_scalene_right",
    # Tumor aliases
    "tumor_gtv": "tumor",
    "GTV": "tumor",
    "gtv": "tumor",
}

STRUCTURE_SUBFOLDER: Dict[str, str] = {
    "common_carotid_artery_left": "normal_structure",
    "common_carotid_artery_right": "normal_structure",
    "trachea": "normal_structure",
    "thyroid_gland": "normal_structure",
    "esophagus": "normal_structure",
    "anterior_scalene_left": "normal_structure",
    "anterior_scalene_right": "normal_structure",
    "internal_jugular_vein_left": "normal_structure",
    "internal_jugular_vein_right": "normal_structure",
    "internal_carotid_artery_left": "normal_structure",
    "internal_carotid_artery_right": "normal_structure",
    "hyoid": "normal_structure",
    "tumor": "tumor",
}

VERTEBRAE = [f"vertebrae_C{i}" for i in range(1, 8)] + ["vertebrae_T1"]
for v in VERTEBRAE:
    STRUCTURE_SUBFOLDER[v] = "normal_structure"

SIDES = ["left", "right"]
