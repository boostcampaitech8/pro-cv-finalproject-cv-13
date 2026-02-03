#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash inference.sh [INPUT_DIR] [OUTPUT_DIR] [--only-ct]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IN_DIR="${1:-${SCRIPT_DIR}/input_folder}"
OUT_DIR="${2:-${SCRIPT_DIR}/output_folder}"
ONLY_CT="false"

for arg in "$@"; do
  if [ "${arg}" = "--only-ct" ]; then
    ONLY_CT="true"
  fi
done

    
## 3. 가용 CPU 코어 수
NUM_CORES=$(nproc)

## 4. Totalsegmentator V2

# TASKS=("total")
TASKS=("headneck_bones_vessels" "headneck_muscles") # head&neck

# 실행 바이너리/디바이스
TS_BIN="${TS_BIN:-TotalSegmentator}"
DEVICE="${DEVICE:-gpu}"

# if ! command -v "${TS_BIN}" >/dev/null 2>&1; then
#   if [[ -x "${FALLBACK_TS_BIN}" ]]; then
#     TS_BIN="${FALLBACK_TS_BIN}"
#   fi
# done

mkdir -p "${IN_DIR}" "${OUT_DIR}"

# Optional venv activation for non-docker environments
# if [ -n "${VENV_DIR:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
#   # shellcheck disable=SC1090
#   source "${VENV_DIR}/bin/activate"
# fi

shopt -s nullglob
all_files=("${IN_DIR}"/*.nii.gz)
if [ ${#all_files[@]} -eq 0 ]; then
  echo "No .nii.gz in ${IN_DIR}"
  exit 0
fi

selected_files=()
if [ "${ONLY_CT}" = "true" ]; then
  ct_named_files=("${IN_DIR}"/*_0000.nii.gz)
  if [ ${#ct_named_files[@]} -gt 0 ]; then
    selected_files=("${ct_named_files[@]}")
  else
    echo "No CT files matching *_0000.nii.gz in ${IN_DIR}"
    exit 0
  fi
else
  selected_files=("${all_files[@]}")
fi

if [ ${#selected_files[@]} -eq 0 ]; then
  echo "No CT-like .nii.gz files found in ${IN_DIR}"
  exit 0
fi

prepared_inputs=()
for f in "${selected_files[@]}"; do
  base="$(basename "${f}")"
  if [[ "${base}" == *_0000.nii.gz ]]; then
    prepared_inputs+=("${f}")
    continue
  fi
  # new_path="${IN_DIR}/${base%.nii.gz}_0000.nii.gz"
  # mv "${f}" "${new_path}"
  # prepared_inputs+=("${new_path}")
  prepared_inputs+=("${f}")
done

TASKS=("headneck_bones_vessels" "headneck_muscles")
TS_BIN="${TS_BIN:-TotalSegmentator}"
DEVICE="${DEVICE:-gpu}"

echo "TotalSegmentator inference"
echo "Tasks: ${TASKS[*]}"
echo "Input: ${IN_DIR}"
echo "Output: ${OUT_DIR}"
echo "Device: ${DEVICE}"

for nifti in "${prepared_inputs[@]}"; do
  fname="$(basename "${nifti}")"
  case_name="${fname%.nii.gz}"
  case_out="${OUT_DIR}/${case_name}"
  mkdir -p "${case_out}"

  for task in "${TASKS[@]}"; do
    task_out="${case_out}/${task}"
    mkdir -p "${task_out}"
    "${TS_BIN}" -i "${nifti}" -o "${task_out}" -ta "${task}" --device "${DEVICE}" --higher_order_resampling
  done
done

export nnUNet_results="${SCRIPT_DIR}/normal_structure_model"

NNUNET_INPUT_DIR="${IN_DIR}"
if [ "${ONLY_CT}" = "true" ]; then
  NNUNET_INPUT_DIR="$(mktemp -d)"
  for nifti in "${prepared_inputs[@]}"; do
    cp -f "${nifti}" "${NNUNET_INPUT_DIR}/"
  done
fi

## 5. nnUNet
nnUNetv2_predict -i "${NNUNET_INPUT_DIR}" -o "${OUT_DIR}" -d 3 -c 3d_fullres -f 0 -tr nnUNetTrainer_100epochs

python "${SCRIPT_DIR}/merge_tsv2_to_nnunet.py" \
  --out_dir "${OUT_DIR}" \
  --structure_list "${SCRIPT_DIR}/structure_list.yaml" \
  --n_jobs ${NUM_CORES}