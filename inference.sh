## 1. 가상 환경 접속
VENV_DIR="/data/ephemeral/home/testvenv" # 경로 설정 필요
source $VENV_DIR/bin/activate

## 2. 입출력 파일 설정
IN_DIR="/data/ephemeral/home/normal_structure_segmentation/input_folder" # 경로 설정 필요
OUT_DIR="/data/ephemeral/home/normal_structure_segmentation/output_folder" # 경로 설정 필요


## 3. Totalsegmentator V2

# TASKS=("total")
TASKS=("headneck_bones_vessels" "headneck_muscles") # head&neck

# 실행 바이너리/디바이스
TS_BIN="${TS_BIN:-TotalSegmentator}"
FALLBACK_TS_BIN="${VENV_DIR}/bin/TotalSegmentator"
DEVICE="${DEVICE:-gpu}"

if ! command -v "${TS_BIN}" >/dev/null 2>&1; then
  if [[ -x "${FALLBACK_TS_BIN}" ]]; then
    TS_BIN="${FALLBACK_TS_BIN}"
  fi
fi


echo "TotalSegmentator inference"
echo "Tasks: ${TASKS[*]}"
echo "Input: ${IN_DIR}"
echo "Output: ${OUT_DIR}"
echo "Device: ${DEVICE}"
echo "Binary: ${TS_BIN}"

mkdir -p "${OUT_DIR}"

for nifti in "${IN_DIR}"/*.nii.gz; do
  [[ -e "${nifti}" ]] || { echo "No .nii.gz in ${IN_DIR}"; exit 0; }
  fname="$(basename "${nifti}")"
  case_name="${fname%.nii.gz}"
  case_out="${OUT_DIR}/${case_name}"
  mkdir -p "${case_out}"

  for task in "${TASKS[@]}"; do
    task_out="${case_out}/${task}"
    mkdir -p "${task_out}"
    echo "[RUN] ${case_name} ${task}"
    "${TS_BIN}" -i "${nifti}" -o "${task_out}" -ta "${task}" --device "${DEVICE}" \
      --higher_order_resampling
  done
done

echo "Done."

## 4. nnUNet

# nnunet 환경 설정
export nnUNet_results="./normal_structure_model"

# 모델 선정 및 추론
nnUNetv2_predict -i $IN_DIR -o $OUT_DIR -d 1 -c 3d_fullres -f 0

## 5. TSv2 결과를 nnUNet 결과에 병합
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/merge_tsv2_to_nnunet.py" \
  --out_dir "${OUT_DIR}" \
  --structure_list "${SCRIPT_DIR}/structure_list.yaml" \
  --dataset_json "${OUT_DIR}/dataset.json"