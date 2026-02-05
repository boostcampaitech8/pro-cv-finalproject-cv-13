"""NIfTI 마스크 로딩 및 캐싱."""

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")

from .config import MASK_ALIASES, STRUCTURE_SUBFOLDER


class MaskLoader:
    """NIfTI 마스크 로더."""

    def __init__(
        self,
        segmentation_dir: Optional[str] = None,
        normal_structure_dir: Optional[str] = None,
        tumor_dir: Optional[str] = None,
    ):
        if segmentation_dir is not None:
            base = Path(segmentation_dir)
            self.normal_structure_dir = base / "normal_structure" if normal_structure_dir is None else Path(normal_structure_dir)
            self.tumor_dir = base / "tumor" if tumor_dir is None else Path(tumor_dir)
        else:
            self.normal_structure_dir = Path(normal_structure_dir) if normal_structure_dir else None
            self.tumor_dir = Path(tumor_dir) if tumor_dir else None

        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._affine: Optional[np.ndarray] = None
        self._shape: Optional[Tuple[int, int, int]] = None

    def _get_search_dirs(self, structure_name: str) -> List[Path]:
        dirs = []
        base_name = structure_name.replace("_left", "").replace("_right", "")
        subfolder = STRUCTURE_SUBFOLDER.get(structure_name) or STRUCTURE_SUBFOLDER.get(base_name)

        if subfolder == "normal_structure" and self.normal_structure_dir:
            dirs.append(self.normal_structure_dir)
        elif subfolder == "tumor" and self.tumor_dir:
            dirs.append(self.tumor_dir)
        else:
            if self.normal_structure_dir and self.normal_structure_dir.exists():
                dirs.append(self.normal_structure_dir)
            if self.tumor_dir and self.tumor_dir.exists():
                dirs.append(self.tumor_dir)

        return dirs

    def _resolve_alias(self, name: str) -> List[str]:
        names = [name]
        if name in MASK_ALIASES:
            names.append(MASK_ALIASES[name])
        for alias, canonical in MASK_ALIASES.items():
            if canonical == name and alias not in names:
                names.append(alias)
        return names

    def _find_mask_file(self, structure_name: str) -> Optional[Path]:
        search_dirs = self._get_search_dirs(structure_name)
        possible_names = self._resolve_alias(structure_name)

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for name in possible_names:
                for ext in [".nii.gz", ".nii"]:
                    filepath = search_dir / f"{name}{ext}"
                    if filepath.exists():
                        return filepath
        return None

    def load_mask(self, structure_name: str) -> Optional[np.ndarray]:
        if structure_name in self._cache:
            return self._cache[structure_name][0]

        filepath = self._find_mask_file(structure_name)
        if filepath is None:
            return None

        try:
            nifti = nib.load(str(filepath))
            mask = np.asarray(nifti.get_fdata(), dtype=np.float32)
            affine = nifti.affine
            mask = (mask > 0.5).astype(np.uint8)

            if self._affine is None:
                self._affine = affine.copy()
                self._shape = mask.shape

            self._cache[structure_name] = (mask, affine)
            return mask
        except Exception as e:
            print(f"Warning: Failed to load mask {filepath}: {e}")
            return None

    def get_affine(self, structure_name: Optional[str] = None) -> Optional[np.ndarray]:
        if structure_name is not None:
            if structure_name in self._cache:
                return self._cache[structure_name][1].copy()
            self.load_mask(structure_name)
            if structure_name in self._cache:
                return self._cache[structure_name][1].copy()
        return self._affine.copy() if self._affine is not None else None

    def get_mask_coords(self, structure_name: str) -> Optional[np.ndarray]:
        mask = self.load_mask(structure_name)
        if mask is None:
            return None
        coords = np.argwhere(mask > 0)
        return coords if len(coords) > 0 else None

    def get_available_structures(self) -> List[str]:
        structures = []
        all_dirs = []
        if self.normal_structure_dir and self.normal_structure_dir.exists():
            all_dirs.append(self.normal_structure_dir)
        if self.tumor_dir and self.tumor_dir.exists():
            all_dirs.append(self.tumor_dir)

        for d in all_dirs:
            for f in d.glob("*.nii*"):
                name = f.name.replace(".nii.gz", "").replace(".nii", "")
                canonical = MASK_ALIASES.get(name, name)
                if canonical not in structures:
                    structures.append(canonical)
        return sorted(structures)

    def has_structure(self, structure_name: str) -> bool:
        return self._find_mask_file(structure_name) is not None

    def clear_cache(self):
        self._cache.clear()
