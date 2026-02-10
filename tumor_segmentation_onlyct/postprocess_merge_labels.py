import os
import numpy as np
import nibabel as nib

def merge_labels_in_folder(out_dir: str) -> None:
    for fname in os.listdir(out_dir):
        if not fname.endswith(".nii.gz"):
            continue
        fpath = os.path.join(out_dir, fname)
        img = nib.load(fpath)
        data = img.get_fdata().astype(np.uint8)

        # Label Integration Postprocess
        data = (data > 0).astype(np.uint8)

        out_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(out_img, fpath)
        print(f"merged labels: {fname}")

if __name__ == "__main__":
    merge_labels_in_folder("./tumor_segmentation_onlyct/output_folder")
