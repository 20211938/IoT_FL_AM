# 데이터셋 정보 (Dataset Information)

## 개요 (Overview)

이 문서는 FLAM 프로젝트에서 사용하는 결함 분류 데이터셋에 대한 정보를 제공합니다.

## 데이터셋 통계 (Dataset Statistics)

### 전체 통계
- **전체 파일 수**: 9,665개
- **결함 유형 수**: 6개
- **총 샘플 수**: 11,719개 (중복 포함)

### 결함 유형별 분포 (Defect Type Distribution)

| 순위 | 결함 유형 | 샘플 수 | 비율 |
|------|----------|--------|------|
| 1 | Super Elevation | 4,819개 | 41.1% |
| 2 | Recoater Streaking | 3,746개 | 31.9% |
| 3 | Fail | 1,141개 | 9.7% |
| 4 | Laser capture timing error | 1,089개 | 9.3% |
| 5 | Reocater Streaking | 615개 | 5.2% |
| 6 | Recoater capture timing error | 309개 | 2.6% |

## 학습 가능한 클래스 (Trainable Classes)

현재 데이터셋에는 **30개 이상의 샘플을 가진 6개의 클래스**가 있습니다:

1. **Super Elevation** (4,819개)
   - 가장 많은 샘플을 가진 클래스
   - 전체 데이터의 약 41% 차지

2. **Recoater Streaking** (3,746개)
   - 두 번째로 많은 샘플
   - 전체 데이터의 약 32% 차지

3. **Fail** (1,141개)
   - 일반적인 실패 케이스
   - 전체 데이터의 약 10% 차지

4. **Laser capture timing error** (1,089개)
   - 레이저 캡처 타이밍 오류
   - 전체 데이터의 약 9% 차지

5. **Reocater Streaking** (615개)
   - Recoater 관련 스트리킹 결함
   - 전체 데이터의 약 5% 차지

6. **Recoater capture timing error** (309개)
   - Recoater 캡처 타이밍 오류
   - 전체 데이터의 약 3% 차지

## 데이터셋 정리 상태 (Dataset Cleanup Status)

### 정리 기준
- **최소 비율**: 1.00% (전체 데이터의 1% 미만인 클래스 제거)
- **의미 없는 이름 제거**: 숫자만, D1/D2 패턴, 2자 이하 이름

### 정리 결과
- **검사한 파일 수**: 9,665개
- **발견된 결함 유형 수**: 6개
- **삭제된 클래스**: 없음 (모든 클래스가 기준을 만족)

**결론**: 현재 데이터셋은 정리 기준을 모두 만족하며, 모든 클래스가 학습에 적합한 상태입니다.

## 클래스 불균형 분석 (Class Imbalance Analysis)

### 불균형 비율
- **최대 샘플 수**: 4,819개 (Super Elevation)
- **최소 샘플 수**: 309개 (Recoater capture timing error)
- **불균형 비율**: 약 15.6:1

### 권장 사항
- 클래스 불균형이 있지만, 모든 클래스가 30개 이상의 샘플을 가지고 있어 학습 가능합니다.
- 필요시 데이터 증강(Data Augmentation) 또는 클래스 가중치(Class Weighting)를 고려할 수 있습니다.

## 데이터 구조 (Data Structure)

### 파일 형식
- **이미지 파일**: `.jpg` 형식
- **메타데이터 파일**: `.jpg.json` 형식 (각 이미지에 대응)

### 디렉토리 구조
```
data/labeled_layers/
├── {database_name}/
│   ├── {image_file}.jpg
│   ├── {image_file}.jpg.json
│   └── ...
└── ...
```

### 메타데이터 구조
메타데이터 JSON 파일에는 다음 정보가 포함됩니다:
- `DepositionImageModel.TagBoxes`: Deposition 이미지의 결함 태그
- `ScanningImageModel.TagBoxes`: Scanning 이미지의 결함 태그
- 각 태그는 `Name`과 `Comment` 필드를 포함하며, `Comment`가 우선적으로 사용됩니다.

## 사용 도구 (Tools)

### 데이터 다운로드
```bash
# MongoDB에서 데이터 다운로드 (최대 10,000개)
python utils/Dataset/download_labeled_layers.py --metadata
```

### 데이터셋 정리
```bash
# 소수 클래스 및 의미 없는 이름 제거
python utils/Dataset/cleanup_dataset.py --data-dir data/labeled_layers
```

### 데이터셋 분석
```bash
# 결함 유형 분석
python utils/Dataset/analyze_defect_types.py --data-dir data/labeled_layers
```

## 학습 권장 사항 (Training Recommendations)

### 모델 학습
현재 데이터셋은 다음 조건을 만족합니다:
- ✅ 모든 클래스가 30개 이상의 샘플 보유
- ✅ 의미 있는 클래스 이름 사용
- ✅ 적절한 데이터 분포

### 권장 하이퍼파라미터
- **최소 샘플 수**: 30개 이상
- **클래스 수**: 6개
- **데이터 증강**: 클래스 불균형 완화를 위해 권장
- **클래스 가중치**: 불균형 보정을 위해 고려

## 업데이트 이력 (Update History)

- **최종 업데이트**: 2024년
- **데이터셋 크기**: 9,665개 파일
- **정리 상태**: 완료 (삭제할 클래스 없음)

## 참고 사항 (Notes)

1. **데이터 중복**: 일부 이미지가 여러 결함 유형을 동시에 가질 수 있어, 총 샘플 수(11,719개)가 파일 수(9,665개)보다 많습니다.

2. **클래스 명명**: 
   - `Recoater Streaking`과 `Reocater Streaking`은 서로 다른 클래스입니다 (오타가 아닙니다).
   - `Laser capture timing error`와 `Recoater capture timing error`도 서로 다른 클래스입니다.

3. **데이터 품질**: 
   - 모든 클래스가 정리 기준을 만족하여 학습에 적합한 상태입니다.
   - 추가 정리가 필요하지 않습니다.

