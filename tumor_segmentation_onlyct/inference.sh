## 1. 가상 환경 접속
VENV_DIR="/data/ephemeral/home/testvenv" # 경로 설정 필요
source $VENV_DIR/bin/activate

## 2. 입출력 파일 설정 및 파일 이름 설정
IN_DIR="./tumor_segmentation_onlyct/input_folder"
OUT_DIR="./tumor_segmentation_onlyct/output_folder"


cd $IN_DIR
for f in *.nii.gz; do
    # 이미 _0000.nii.gz 형태면 스킵
    if [[ "$f" == *"_0000.nii.gz" ]]; then
        echo "이미 _0000 있음: $f (스킵)"
        continue
    fi
    
    # _0000 추가
    new_name="${f%.nii.gz}_0000.nii.gz"
    mv "$f" "$new_name"
    echo "변경: $f → $new_name"
done
cd ..
cd ..

## 3. nnUNet

# nnunet 환경 설정
export nnUNet_results="./tumor_segmentation_onlyct/tumor_segmentation_onlyct_model"

# 모델 선정 및 추론
nnUNetv2_predict -i $IN_DIR -o $OUT_DIR -d 901 -c 3d_cascade_fullres -f 0 -tr nnUNetTrainer_100epochs

