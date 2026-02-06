# tumor_segmentation_onlyct

CT 단일 모달리티(nii.gz) 입력을 nnU-Net v2 모델로 추론하는 스크립트 모음입니다.  
이 저장소는 **학습 없이 추론만 수행**하도록 구성되어 있습니다.

## 구성
- `requirements.sh`: 가상환경 생성 및 의존성 설치
- `inference.sh`: 입력 파일 준비 및 nnU-Net v2 추론 실행
- `nnUNet/`: nnU-Net v2 소스(Editable install)
- `tumor_segmentation_onlyct_model/`: 사전 학습된 모델 가중치

## 설치
```bash
bash /data/ephemeral/home/T8215/tumor_segmentation_onlyct/requirements.sh
```

설치 스크립트가 하는 일
- Python 3.10 venv 생성 (`/data/ephemeral/home/testvenv`)
- 임시 디렉토리 환경변수 설정
- `gcc` 설치
- `torch==2.8.0`(CUDA 12.8) 및 `torchvision`, `torchaudio` 설치
- `nnUNet` 소스 editable 설치

## 입력/출력 폴더 준비
`inference.sh`는 아래 경로를 사용합니다.
- 입력: `./tumor_segmentation_onlyct/input_folder`
- 출력: `./tumor_segmentation_onlyct/output_folder`

필요하면 직접 생성하세요.
```bash
mkdir -p /data/ephemeral/home/T8215/tumor_segmentation_onlyct/input_folder
mkdir -p /data/ephemeral/home/T8215/tumor_segmentation_onlyct/output_folder
```

## 입력 파일 규칙
nnU-Net v2는 입력 파일명이 `_0000.nii.gz`로 끝나야 합니다.  
`inference.sh`가 자동으로 접미사를 붙여줍니다.

예)
- `CHUM-001.nii.gz` → `CHUM-001_0000.nii.gz`

## 추론 실행
**반드시 `/data/ephemeral/home/T8215`에서 실행**하세요. (상대경로를 사용함)
```bash
cd /data/ephemeral/home/T8215
bash /data/ephemeral/home/T8215/tumor_segmentation_onlyct/inference.sh
```

## 모델 정보
- Dataset ID: `901` (`Dataset901_HECKTOR25`)
- Trainer: `nnUNetTrainer_100epochs`
- Plans: `nnUNetResEncUNetLPlans`
- Configuration: `3d_fullres`
- Checkpoint: `tumor_segmentation_onlyct_model/Dataset901_HECKTOR25/nnUNetTrainer_100epochs__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth`

## 주의사항
- 스크립트는 `nnUNet_results`를 `tumor_segmentation_onlyct_model`로 설정합니다.
- `nnUNet_raw`, `nnUNet_preprocessed` 경고는 추론만 할 때는 무시해도 됩니다.
- `3d_cascade_fullres`를 쓰면 `3d_lowres` 결과가 필요합니다. 현재 스크립트는 `3d_fullres`만 사용합니다.

## 라이선스/인용
nnU-Net v2를 사용하므로 아래 논문을 인용해야 합니다.
- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203-211.
