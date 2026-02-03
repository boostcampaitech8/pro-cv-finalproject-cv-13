"""
TSv2 추론 결과 중 structure_list.yaml에 있는 라벨만 nnUNet 결과에 **추가**하는 스크립트.
Dataset003 사용 시 라벨이 겹치지 않으므로 덮어쓰기 대신 병합(merge)만 수행.
CPU 병렬 처리로 속도 향상.

사용법:
    python merge_tsv2_to_nnunet_parallel.py --out_dir <OUT_DIR> --structure_list <YAML> [--n_jobs <N>]
"""

import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np
import yaml


def process_case(args_tuple):
    """단일 케이스 처리 함수 (병렬 처리용)"""
    case_dir, out_dir, label_map = args_tuple

    case_name = case_dir.name
    # nnUNet은 _0000 채널 접미사를 제거하므로, 매칭을 위해 _0000 제거
    nnunet_name = case_name.rsplit("_", 1)[0] if case_name.endswith("_0000") else case_name
    nnunet_file = out_dir / f"{nnunet_name}.nii.gz"

    if not nnunet_file.exists():
        return f"[SKIP] nnUNet 결과 없음: {nnunet_file}"

    result_lines = [f"[MERGE] {case_name}"]

    nnunet_img = nib.load(str(nnunet_file))
    nnunet_data = np.asarray(nnunet_img.dataobj).copy()

    # 각 TSv2 task 폴더 탐색
    for task_dir in sorted(case_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        for label_file in sorted(task_dir.glob("*.nii.gz")):
            tsv2_name = label_file.stem.replace(".nii", "")

            # structure_list.yaml의 키와 매칭
            if tsv2_name not in label_map:
                result_lines.append(f"  [SKIP] {tsv2_name} not in structure_list.yaml")
                continue

            label_val = label_map[tsv2_name]
            mask_img = nib.load(str(label_file))
            mask_data = np.asarray(mask_img.dataobj)

            # Dataset003 사용 시 라벨이 안 겹치므로, 배경(0)인 곳에만 TSV2 결과 추가
            # 덮어쓰기 방지: nnUNet 결과가 있는 곳은 건드리지 않음
            mask_to_add = (mask_data > 0) & (nnunet_data == 0)
            nnunet_data[mask_to_add] = label_val
            result_lines.append(f"  [OK] {tsv2_name} -> label {label_val}")

    # 병합된 결과 저장
    out_img = nib.Nifti1Image(nnunet_data, nnunet_img.affine, nnunet_img.header)
    nib.save(out_img, str(nnunet_file))
    result_lines.append(f"  [SAVE] {nnunet_file}")

    # TSv2 케이스 폴더 삭제
    shutil.rmtree(case_dir)
    result_lines.append(f"  [DEL] {case_dir}")

    return "\n".join(result_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--structure_list", required=True)
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="병렬 처리 프로세스 수 (기본값: CPU 코어 수)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # structure_list.yaml에서 라벨명과 번호 읽기
    with open(args.structure_list) as f:
        label_map = yaml.safe_load(f)["labels"]

    # 라벨 맵 확인
    print(f"structure_list.yaml에서 읽은 라벨: {label_map}")

    # 처리할 케이스 디렉토리 목록
    case_dirs = [d for d in sorted(out_dir.iterdir()) if d.is_dir()]

    if not case_dirs:
        print("처리할 케이스가 없습니다.")
        return

    # 병렬 처리할 작업 준비
    n_jobs = args.n_jobs if args.n_jobs else cpu_count()
    print(f"총 {len(case_dirs)}개 케이스를 {n_jobs}개 프로세스로 병렬 처리합니다.")

    tasks = [(case_dir, out_dir, label_map) for case_dir in case_dirs]

    # 병렬 처리
    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_case, tasks)

    # 결과 출력
    for result in results:
        print(result)

    # JSON 파일 삭제
    for json_file in out_dir.glob("*.json"):
        json_file.unlink()
        print(f"[DEL JSON] {json_file}")

    print("Done.")


if __name__ == "__main__":
    main()
