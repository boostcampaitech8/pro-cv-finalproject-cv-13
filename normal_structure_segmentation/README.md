# Normal Structure Segmentation

Neck CT에서 정상 해부학적 구조물을 자동으로 분할하는 추론 파이프라인입니다.
**nnU-Net**과 **TotalSegmentator V2(TSv2)** 두 모델의 결과를 결합하여 최종 세그멘테이션을 생성합니다.

## 프로젝트 구조

```
normal_structure_segmentation/     # 프로젝트 루트 폴더
├── requirements.sh                # 환경 설정 및 패키지 설치
├── inference.sh                   # 추론 파이프라인 실행 스크립트
├── merge_tsv2_to_nnunet.py        # TSv2 결과를 nnUNet 결과에 병합
├── structure_list.yaml            # TSv2에서 가져올 구조물 목록
├── input_folder/                  # 입력 CT 이미지 (.nii.gz)
├── output_folder/                 # 출력 세그멘테이션 결과
├── normal_structure_model/        # 학습된 nnUNet 모델 가중치
└── nnUNet/                        # nnU-Net 프레임워크 (v2.6.3)
```

**주의**
1. input_folder와 output_folder를 생성해야 합니다.(github에는 빈 폴더가 push되지 않습니다.)
2. input_folder에 .nii.gz 형식의 입력할 CT 이미지를 넣어야 합니다.

### nnU-Net 가중치 경로

```
normal_structure_model/
└── Dataset003_HeadNeckCT/
    └── nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/
        ├── fold_0/
        │   └── checkpoint_final.pth    # 학습된 모델 가중치
        ├── plans.json                  # 모델 구성 (네트워크 구조, 패치 크기 등)
        └── dataset.json                # 라벨 정의 (18 클래스 매핑)
```

`inference.sh`에서 `nnUNet_results` 환경변수를 `./normal_structure_model`로 설정하여 해당 가중치를 참조합니다.

**주의**
1. fold_0 폴더를 생성 후 해당 폴더에 가중치를 집어 넣어야 합니다.

## 세그멘테이션 대상 (18 구조물)

| 카테고리 | 구조물 |
|---------|--------|
| 혈관 | IJV (좌/우), 총경동맥 (좌/우) |
| 근육 | 전사각근 (좌/우) |
| 골격 | 설골, 경추 C1–C7, 흉추 T1 |
| 장기 | 식도, 갑상선, 기관 |

이 중 **5개 구조물**(설골, IJV 좌/우, 전사각근 좌/우)은 TSv2 결과로 덮어씁니다.

## 환경 설정

```bash
bash ./normal_stucture_segmentation/requirements.sh
```

- Python 3.10 가상환경 생성
- PyTorch 2.8.0 (CUDA 12.8) 설치
- nnU-Net, TotalSegmentator 설치

**주의**
1. 원격 서버를 이용하지 않는 경우 경로 설정이 필요한 부분이 있습니다.(1. 가상 환경 생성 ~ 3. 임시 디렉토리 설정)

## 추론 실행

```bash
bash ./normal_stucture_segmentation/inference.sh
```

**주의**
1. 원격 서버를 이용하지 않는 경우 경로 설정이 필요한 부분이 있습니다.(1. 가상 환경 활성화)

### 파이프라인 단계

1. **TotalSegmentator V2** — `headneck_bones_vessels`, `headneck_muscles` 태스크 추론 (GPU)
2. **nnU-Net** — 3D full resolution 모델로 전체 구조물 추론 (`Dataset003_HeadNeckCT`, fold 0)
3. **병합** — `structure_list.yaml`에 정의된 5개 구조물을 TSv2 결과로 nnUNet 결과에 합치기

### 입출력

- **입력**: `input_folder/` 내 `.nii.gz` CT 이미지
- **출력**: `output_folder/` 내 `.nii.gz` (18-class 라벨맵)

## 모델 정보

- **nnU-Net**: 3D Full Resolution PlainConvUNet, 패치 크기 128³, 학습 데이터 48건
- **TotalSegmentator V2**: 사전학습된 head & neck 모델
- **복셀 간격**: 1.5mm × 1.5mm × 1.5mm

## 2026-02-3 변경 사항

1. nnUNet 모델 예측 라벨 18 -> 13개 
* TSV2가 예측하는 라벨(5개)는 예측하지 않도록 변경
* 변경 파일
   1) normal_structure_model 하위 파일 및 폴더 전부


3. CPU 병렬 처리
* TSV2 라벨과 nnUNet 라벨이 겹치지 않아서 라벨을 합치는 것으로 변경
* 변경 파일
   1) inference.sh
   2) structure_list.yaml
   3) merge_tsv2_to_nnunet.py
