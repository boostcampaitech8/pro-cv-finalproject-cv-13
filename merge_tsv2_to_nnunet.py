"""
TSv2 추론 결과 중 structure_list.yaml에 있는 라벨만 nnUNet 결과에 덮어쓰고,
나머지 TSv2 결과는 삭제하는 스크립트.

사용법:
    python merge_tsv2_to_nnunet.py --out_dir <OUT_DIR> --structure_list <YAML> --dataset_json <JSON>
"""

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

# TSv2 출력 파일명 → dataset.json 라벨명 매핑
TSV2_TO_DATASET = {
    "internal_jugular_vein_left": "IJV_left",
    "internal_jugular_vein_right": "IJV_right",
    "anterior_scalene_left": "Anterior_scalene_left",
    "anterior_scalene_right": "Anterior_scalene_right",
    "hyoid": "hyoid",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--structure_list", required=True)
    parser.add_argument("--dataset_json", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    with open(args.structure_list) as f:
        keep_labels = yaml.safe_load(f)["labels"]

    with open(args.dataset_json) as f:
        label_map = json.load(f)["labels"]

    # dataset.json 라벨명 → 정수값
    name_to_val = {name: int(val) for name, val in label_map.items()}

    # 케이스별 처리: OUT_DIR 안의 디렉토리 = TSv2 케이스 출력
    for case_dir in sorted(out_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        case_name = case_dir.name
        # nnUNet은 _0000 채널 접미사를 제거하므로, 매칭을 위해 _0000 제거
        nnunet_name = case_name.rsplit("_", 1)[0] if case_name.endswith("_0000") else case_name
        nnunet_file = out_dir / f"{nnunet_name}.nii.gz"
        if not nnunet_file.exists():
            print(f"[SKIP] nnUNet 결과 없음: {nnunet_file}")
            continue

        print(f"[MERGE] {case_name}")
        nnunet_img = nib.load(str(nnunet_file))
        nnunet_data = np.asarray(nnunet_img.dataobj).copy()

        # 각 TSv2 task 폴더 탐색
        for task_dir in sorted(case_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            for label_file in sorted(task_dir.glob("*.nii.gz")):
                tsv2_name = label_file.stem.replace(".nii", "")
                if tsv2_name not in keep_labels:
                    continue

                dataset_name = TSV2_TO_DATASET.get(tsv2_name, tsv2_name)
                if dataset_name not in name_to_val:
                    print(f"  [WARN] {dataset_name} not in dataset.json, skip")
                    continue

                label_val = name_to_val[dataset_name]
                mask_img = nib.load(str(label_file))
                mask_data = np.asarray(mask_img.dataobj)
                nnunet_data[mask_data > 0] = label_val
                print(f"  [OK] {tsv2_name} -> label {label_val}")

        # 덮어쓴 결과 저장
        out_img = nib.Nifti1Image(nnunet_data, nnunet_img.affine, nnunet_img.header)
        nib.save(out_img, str(nnunet_file))
        print(f"  [SAVE] {nnunet_file}")

        # TSv2 케이스 폴더 삭제
        shutil.rmtree(case_dir)
        print(f"  [DEL] {case_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
