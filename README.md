# Inference 

두 모델의 추론 파이프라인입니다. (normal_structure_segmentation 및 tumor_segmentation branch를 합친 branch)
- `tumor_segmentation`: nnUNetv2 종양 분할 + Dice 계산
- `normal_structure_segmentation`: nnUNet + TotalSegmentator 결과 병합
- `nerve_estimation`: 신경 추정 로직

## Docker Compose 실행
`docker-compose.yml`이 있는 폴더에서 실행하세요.

```bash
docker compose up --build
```

서비스별 실행:

```bash
docker compose up --build tumor-seg
docker compose up --build normal-seg
docker compose up --build nerve-estimation
```

모든 서비스 실행:
```bash
docker compose up
```

중지:

```bash
docker compose down
```

## 참고
- 입력/출력 경로 마운트는 `docker-compose.yml`에서 관리합니다.
- 코드 변경 후에는 `--build` 옵션으로 이미지를 다시 빌드하세요.
- `normal-seg`를 `--only-ct`로 실행할 때 CT 파일명은 `*_0000.nii.gz` 형식이어야 합니다.
- 모든 결과들은 `outputs` 폴더에 저장됩니다. 
  - `normal` : 정상 구조물
  - `tumor` : 병변
  - `nerve` : 신경 추정 결과