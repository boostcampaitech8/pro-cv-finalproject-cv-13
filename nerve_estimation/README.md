# Nerve Estimation Pipeline

CT 분할 마스크를 기반으로 경부 신경의 위치를 추정하고, 종양과의 근접도를 평가하는 파이프라인

## 개요

경부 수술 시 손상 위험이 있는 4개 신경의 위치를 해부학적 구조물 간의 관계를 이용해 추정

## 의존성

- Python >= 3.8
- NumPy
- SciPy
- NiBabel

## 빠른 시작

```python
from nerve_estimation import run_nerve_estimation

# 신경 추정 및 위험도 계산 실행
results = run_nerve_estimation(
    segmentation_dir="/path/to/segmentation",
    output_dir="/path/to/output"
)

# 결과 확인
for nerve in results["nerves"]:
    print(f"{nerve['nerve']} ({nerve['side']}): {nerve['type']}")

for risk in results["risks"]:
    print(f"{risk['nerve']}: {risk['risk_level']} ({risk['min_distance_mm']:.1f}mm)")
```

## 입력 데이터 구조

```
segmentation_dir/
├── normal_structure/
│   ├── common_carotid_artery_left.nii.gz
│   ├── common_carotid_artery_right.nii.gz
│   ├── internal_jugular_vein_left.nii.gz
│   ├── internal_jugular_vein_right.nii.gz
│   ├── thyroid_gland.nii.gz
│   ├── trachea.nii.gz
│   ├── esophagus.nii.gz
│   ├── anterior_scalene_left.nii.gz
│   └── anterior_scalene_right.nii.gz
└── tumor/
    └── tumor.nii.gz
```

## 아키텍처

```
nerve_estimation/
├── __init__.py          # 패키지 진입점
├── config.py            # 전역 설정
├── pipeline.py          # 메인 파이프라인 클래스
├── mask_loader.py       # NIfTI 마스크 로딩 및 캐싱
├── landmarks.py         # 해부학적 랜드마크 추출
├── utils.py             # 좌표 변환 유틸리티
├── risk.py              # 위험도 계산 (EDT 기반)
├── export.py            # NIfTI/JSON 내보내기
└── estimators/
    ├── base.py          # 추상 기본 클래스
    ├── vagus.py         # 미주신경 추정기
    ├── ebsln.py         # 상후두신경 외지 추정기
    ├── rln.py           # 반회후두신경 추정기
    └── phrenic.py       # 횡격막신경 추정기
```

## 위험도 평가

### 알고리즘

1. **EDT (Euclidean Distance Transform)** 계산
   - 종양 외부 모든 복셀에서 종양 표면까지의 거리 맵 생성

2. **최소 거리 계산**
   - 신경 경로의 각 점에서 거리 맵 조회

3. **실효 거리 계산**
   ```
   effective_distance = min_distance - uncertainty
   ```

4. **위험도 판정**
   | 조건 | 위험도 |
   |------|--------|
   | 겹침 또는 실효거리 < 5mm | HIGH |
   | 실효거리 < 10mm | MODERATE |
   | 그 외 | LOW |

## 출력

### JSON 결과

```json
{
  "metadata": {
    "timestamp": "2025-02-03T11:30:00",
    "spacing_mm": [0.5, 0.5, 1.0]
  },
  "nerves": [
    {
      "nerve": "vagus",
      "side": "left",
      "type": "pathway",
      "uncertainty_mm": 5.0,
      "method": "midpoint_posterior",
      "reference": "Inamura et al. 2017",
      "pathway_voxels": [[120, 85, 50], ...],
      "pathway_mm": [[60.0, 42.5, 50.0], ...]
    }
  ],
  "risks": [
    {
      "nerve": "vagus",
      "side": "left",
      "risk_level": "MODERATE",
      "min_distance_mm": 12.5,
      "uncertainty_mm": 5.0,
      "effective_distance_mm": 7.5
    }
  ]
}
```

## 설정

### 신경 파라미터 (`config.py`)

```python
NERVE_CONFIG = {
    "vagus": {
        "uncertainty_mm": 5.0,
        "method": "midpoint_posterior",
        "posterior_offset_mm": 2.04,
        "required_structures": ["common_carotid_artery", "internal_jugular_vein"],
    },
    ...
}
```

### 위험도 임계값

```python
RISK_THRESHOLDS = {
    "high": 5.0,      # mm
    "moderate": 10.0, # mm
}
```

## 좌표계

NIfTI 표준 RAS+ 좌표계를 사용