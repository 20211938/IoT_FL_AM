# FLAM 프로젝트 코드 분석 (전체 Python 파일 포함)

## 🎯 이 프로젝트가 하는 일

**3D 프린팅 과정에서 결함을 찾는 AI 모델**을 만드는 프로젝트입니다.
- 여러 공장(클라이언트)의 데이터를 모으지 않고도 함께 학습하는 **연합 학습(Federated Learning)** 방식 사용
- 각 픽셀을 3가지로 분류: **파우더(0) / 부품(1) / 결함(2)**
- MongoDB에서 레이블된 레이어 이미지를 다운로드하여 학습 데이터로 활용

---

## 📁 파일 구조 및 역할

```
FLAM/
├── Federated_learning.ipynb          # 메인 노트북 (실행 스크립트)
├── utils/
│   ├── image_processing.py           # 이미지 전처리 및 타일링
│   ├── dataset_functions.py          # 데이터셋 생성 및 관리
│   ├── unet.py                       # U-Net 모델 정의
│   ├── federated_averaging.py        # FedAvg 연합 학습 알고리즘
│   ├── visualization.py              # 결과 시각화 및 평가
│   ├── defect_type_classifier.py     # 결함 유형 분류 모델 학습
│   └── test_defect_type_classifier.py # 결함 분류 모델 테스트
├── util_dataset/
│   ├── download_labeled_layers.py    # MongoDB에서 레이블된 이미지 다운로드
│   ├── cleanup_dataset.py            # 데이터셋 정리 및 소수 클래스 제거
│   └── analyze_defect_types.py       # 결함 유형 분석
└── data/                             # 이미지 데이터 저장소
    └── labeled_layers/               # 다운로드된 레이블된 이미지
```

---

## 📝 각 파일 상세 분석

### 1. `utils/image_processing.py` - 이미지 전처리 및 타일링

**역할**: 큰 이미지를 작은 타일로 나누고, 타일을 다시 합치는 기능 제공

### 2. `utils/dataset_functions.py` - 데이터셋 생성 및 관리

**역할**: 클라이언트별 데이터셋 생성 및 데이터 결합

### 3. `utils/unet.py` - U-Net 모델 정의

**역할**: U-Net 아키텍처 모델 생성 및 컴파일

### 4. `utils/federated_averaging.py` - 연합 학습 알고리즘

**역할**: FedAvg 알고리즘 구현 - 각 클라이언트에서 로컬 학습 후 서버에서 가중치 평균

#### 알고리즘 동작 과정

```
1. 초기화
   - 각 클라이언트의 데이터 개수 계산
   - 가중 평균을 위한 비율 계산 (proportionsDict)
   - 서버 가중치 초기화

2. 각 서버 라운드마다:
   
   a) 클라이언트 업데이트 (로컬 학습)
      - 각 클라이언트에 대해:
        1. 글로벌 모델 복제
        2. 서버 가중치로 초기화
        3. 로컬 데이터로 LOCAL_EPOCHS만큼 학습
        4. 학습된 가중치 저장
        5. 손실/정확도 기록
   
   b) 서버 업데이트 (가중치 평균)
      - 각 클라이언트의 가중치를 데이터 비율로 가중 평균
      - 공식: w_global = Σ(n_k / N) * w_k
        - n_k: 클라이언트 k의 데이터 개수
        - N: 전체 데이터 개수
        - w_k: 클라이언트 k의 가중치
   
   c) 글로벌 모델 업데이트
      - 평균된 가중치를 글로벌 모델에 적용
   
   d) 테스트 평가
      - 테스트셋으로 성능 평가
      - 테스트 손실/정확도 기록

3. 반환
   - 학습된 모델 및 모든 기록 반환
```

#### 가중 평균 예시

```python
# 클라이언트별 데이터 개수
client1: 1000개 → 비율 0.5
client2: 500개  → 비율 0.25
client3: 500개  → 비율 0.25

# 가중 평균
w_global = 0.5 * w_client1 + 0.25 * w_client2 + 0.25 * w_client3
```

---

### 5. `utils/visualization.py` - 결과 시각화 및 평가

**역할**: 학습된 모델의 예측 결과 시각화 및 성능 평가

### 6. `util_dataset/download_labeled_layers.py` - MongoDB 이미지 다운로드

**역할**: MongoDB GridFS에서 레이블된 레이어 이미지를 다운로드

**주요 기능**:
- MongoDB 연결 및 데이터베이스 목록 조회
- `IsLabeled=True`인 레이어만 필터링
- GridFS에서 이미지 파일 다운로드
- 메타데이터 JSON 파일 저장 (선택사항)
- 전체 다운로드 제한 설정 (기본값: 10,000개)

### 7. `util_dataset/cleanup_dataset.py` - 데이터셋 정리

**역할**: 다운로드된 데이터셋에서 소수 클래스 및 의미 없는 이름 제거

**주요 기능**:
- 결함 유형별 통계 수집
- 비율 기반 필터링 (기본값: 1% 미만 제거)
- 의미 없는 이름 감지 및 제거 (숫자만, D1/D2 패턴, 2자 이하)
- DRY RUN 모드 지원

### 8. `utils/defect_type_classifier.py` - 결함 유형 분류 모델

**역할**: 특정 결함 유형을 분류하는 CNN 모델 학습

**주요 기능**:
- 메타데이터에서 결함 유형 추출
- CNN 기반 다중 클래스 분류 모델
- 학습/검증 데이터 분할
- 체크포인트 저장 및 최적 모델 선택

---

## 🚀 전체 실행 과정 (기본 설정)

프로젝트를 처음부터 실행하는 전체 과정입니다. 모든 명령어는 기본 설정을 사용합니다.

### 1단계: 데이터 다운로드

MongoDB에서 레이블된 이미지를 다운로드합니다 (최대 10,000개).

```bash
python util_dataset/download_labeled_layers.py --metadata
```

**결과**: `data/labeled_layers/` 디렉토리에 이미지 파일과 메타데이터 JSON 파일이 저장됩니다.

### 2단계: 데이터셋 정리

다운로드된 데이터에서 소수 클래스(1% 미만) 및 의미 없는 이름을 제거합니다.

```bash
python util_dataset/cleanup_dataset.py --data-dir data/labeled_layers
```

**결과**: 학습에 적합한 데이터만 남게 됩니다.

### 3단계: 결함 분류 모델 학습

정리된 데이터로 결함 유형 분류 모델을 학습합니다.

```bash
python utils/defect_type_classifier.py --data-dir data/labeled_layers --metadata
```

**결과**: `checkpoints/` 디렉토리에 최적 모델이 저장됩니다.

### 4단계: 모델 테스트

학습된 모델의 성능을 평가합니다.

```bash
python utils/test_defect_type_classifier.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/labeled_layers \
    --metadata
```

**결과**: 정확도, 혼동 행렬 등 평가 결과가 출력됩니다.

---

## 🔍 U-Net을 이용한 픽셀 단위 결함 탐지 (요약)

### 개요

U-Net 모델을 사용하여 이미지의 각 픽셀을 3가지 클래스로 분류합니다:
- **파우더 (0)**: 파우더 영역
- **부품 (1)**: 정상 부품 영역
- **결함 (2)**: 결함 영역

### 주요 특징

- **연합 학습 (Federated Learning)**: 여러 공장의 데이터를 모으지 않고도 함께 학습
- **픽셀 단위 분할**: 이미지의 각 픽셀을 개별적으로 분류
- **타일링**: 큰 이미지를 128×128 타일로 분할하여 처리
- **8개 클라이언트**: 7개 공장에서 학습, 1개 공장으로 테스트

### 실행 방법

```python
# Jupyter Notebook에서 실행
# Federated_learning.ipynb 파일 참조
```

### 하이퍼파라미터

```python
SERVER_ROUNDS = 2                 # 서버 라운드 수
LOCAL_EPOCHS = 5                  # 클라이언트당 로컬 에포크
LOCAL_BATCH_SIZE = 32             # 배치 크기
LOCAL_LEARNING_RATE = 8e-05       # 로컬 학습률
tileSize = 128                    # 타일 크기
```

### 알고리즘 동작

1. 각 클라이언트에서 로컬 학습 수행
2. 서버에서 가중치를 데이터 비율로 가중 평균
3. 글로벌 모델 업데이트
4. 반복 (SERVER_ROUNDS만큼)

---

## 🎓 결함 분류 모델 상세 설명

### 개요

결함 분류 모델은 이미지 전체를 입력받아 **어떤 종류의 결함인지**를 분류하는 CNN 기반 다중 클래스 분류 모델입니다. U-Net과 달리 픽셀 단위가 아닌 이미지 단위로 결함 유형을 판단합니다.

### 데이터셋 구조

#### 디렉토리 구조

```
data/labeled_layers/
├── {database_name}/              # 데이터베이스별 디렉토리
│   ├── {db_name}_layer{num}_{id}.jpg      # 이미지 파일
│   ├── {db_name}_layer{num}_{id}.jpg.json # 메타데이터 파일
│   └── ...
└── ...
```

#### 파일 형식

- **이미지 파일**: `.jpg` 형식의 레이어 이미지
- **메타데이터 파일**: `.jpg.json` 형식의 JSON 파일
  - 각 이미지에 대응하는 메타데이터
  - 결함 유형 정보 포함

#### 메타데이터 구조

메타데이터 JSON 파일에는 다음 정보가 포함됩니다:

```json
{
  "DepositionImageModel": {
    "TagBoxes": [
      {
        "Name": "D1",
        "Comment": "Super Elevation"
      }
    ]
  },
  "ScanningImageModel": {
    "TagBoxes": [
      {
        "Name": "D2",
        "Comment": "Recoater Streaking"
      }
    ]
  }
}
```

- `DepositionImageModel.TagBoxes`: Deposition 이미지의 결함 태그
- `ScanningImageModel.TagBoxes`: Scanning 이미지의 결함 태그
- 각 태그의 `Comment` 필드가 결함 유형으로 사용됩니다 (없으면 `Name` 사용)

### 결함 종류 (Defect Types)

현재 데이터셋에는 **6가지 결함 유형**이 있습니다:

| 순위 | 결함 유형 | 샘플 수 | 비율 | 설명 |
|------|----------|--------|------|------|
| 1 | **Super Elevation** | 4,819개 | 41.1% | 표면이 정상보다 높게 솟아오른 결함 |
| 2 | **Recoater Streaking** | 3,746개 | 31.9% | Recoater로 인한 스트리킹 결함 |
| 3 | **Fail** | 1,141개 | 9.7% | 일반적인 실패 케이스 |
| 4 | **Laser capture timing error** | 1,089개 | 9.3% | 레이저 캡처 타이밍 오류 |
| 5 | **Reocater Streaking** | 615개 | 5.2% | Recoater 관련 스트리킹 결함 (Recoater와 다른 유형) |
| 6 | **Recoater capture timing error** | 309개 | 2.6% | Recoater 캡처 타이밍 오류 |

**참고사항**:
- `Recoater Streaking`과 `Reocater Streaking`은 서로 다른 클래스입니다 (오타가 아닙니다)
- 일부 이미지는 여러 결함 유형을 동시에 가질 수 있습니다
- 총 샘플 수(11,719개)가 파일 수(9,665개)보다 많은 이유는 하나의 이미지가 여러 결함 유형을 포함할 수 있기 때문입니다

### 데이터셋 통계

- **전체 파일 수**: 9,665개
- **결함 유형 수**: 6개
- **총 샘플 수**: 11,719개 (중복 포함)
- **클래스 불균형 비율**: 약 15.6:1 (Super Elevation vs Recoater capture timing error)

### 모델 구조

```
입력: 이미지 (224×224×3)
  ↓
CNN 레이어
  ├─ Conv2D + BatchNorm + ReLU
  ├─ MaxPooling
  ├─ Conv2D + BatchNorm + ReLU
  ├─ MaxPooling
  └─ ...
  ↓
Fully Connected 레이어
  ↓
Softmax
  ↓
출력: 결함 유형 클래스 확률 (6개 클래스)
```

### 학습 과정

1. **메타데이터 파싱**: JSON 파일에서 결함 유형 추출
2. **데이터 필터링**: 최소 샘플 수 미만인 클래스 제거 (기본값: 10개)
3. **데이터 분할**: 학습 80% / 검증 20%
4. **모델 초기화**: CNN 모델 생성
5. **학습**: Adam 옵티마이저로 학습
6. **검증**: 각 에포크마다 검증 정확도 확인
7. **체크포인트 저장**: 최고 성능 모델 저장

### 기본 하이퍼파라미터

```python
epochs = 20              # 학습 에포크 수
batch_size = 16         # 배치 크기
learning_rate = 0.001   # 학습률
min_count = 10          # 최소 샘플 수 (이보다 적으면 클래스 제거)
```

### 실행 명령어

```bash
# 기본 사용
python utils/defect_type_classifier.py --data-dir data/labeled_layers --metadata

# 모델 테스트
python utils/test_defect_type_classifier.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/labeled_layers \
    --metadata
```

### 평가 지표

- **전체 정확도**: 전체 예측 중 정확한 예측 비율
- **클래스별 정확도**: 각 결함 유형별 정확도
- **혼동 행렬**: 클래스 간 오분류 패턴 분석

---

## 🔄 전체 워크플로우

### A. 결함 분류 모델 학습 워크플로우 (새로운 방식)

```
1. 데이터 다운로드
   └─ download_labeled_layers.py: MongoDB에서 레이블된 이미지 다운로드 (최대 10,000개)
   
2. 데이터셋 정리
   └─ cleanup_dataset.py: 소수 클래스 및 의미 없는 이름 제거
      ├─ 비율 기반 필터링 (1% 미만)
      ├─ 의미 없는 이름 제거 (숫자만, D1/D2 패턴)
      └─ 정리된 데이터셋 생성
   
3. 결함 유형 분석 (선택사항)
   └─ analyze_defect_types.py: 데이터셋 통계 및 분포 분석
   
4. 결함 분류 모델 학습
   └─ defect_type_classifier.py: CNN 기반 다중 클래스 분류 모델 학습
      ├─ 메타데이터에서 결함 유형 추출
      ├─ 학습/검증 데이터 분할
      ├─ CNN 모델 학습
      └─ 최적 모델 체크포인트 저장
   
5. 모델 테스트
   └─ test_defect_type_classifier.py: 학습된 모델 평가
      ├─ 정확도 계산
      ├─ 혼동 행렬 생성
      └─ 오분류 분석
```

### B. 픽셀 단위 결함 탐지 워크플로우 (기존 방식)

```
1. 데이터 준비
   └─ download_labeled_layers.py: MongoDB에서 레이블된 이미지 다운로드
   
2. 이미지 전처리
   └─ image_processing.py: 이미지를 128×128 타일로 분할
   
3. 데이터셋 생성
   └─ dataset_functions.py: 클라이언트별 데이터셋 생성
   
4. 모델 초기화
   └─ unet.py: U-Net 모델 생성 및 컴파일
   
5. 연합 학습
   └─ federated_averaging.py: FedAvg 알고리즘 실행
      ├─ 각 클라이언트에서 로컬 학습
      ├─ 서버에서 가중치 평균
      └─ 반복
   
6. 결과 평가 및 시각화
   └─ visualization.py: 예측 결과 시각화 및 성능 평가
```

---

## 🔑 핵심 개념 정리

| 개념 | 설명 | 관련 파일 |
|------|------|----------|
| **타일링** | 큰 이미지를 128×128 조각으로 나누기 | `image_processing.py` |
| **클라이언트** | 각 공장 (client1~client8) | `dataset_functions.py` |
| **연합 학습** | 각 공장에서 따로 학습 후 서버에서 합치기 | `federated_averaging.py` |
| **U-Net** | 이미지 분할용 모델 (인코더-디코더 + Skip Connection) | `unet.py` |
| **3클래스** | 파우더(0), 부품(1), 결함(2) | 모든 파일 |
| **GridFS** | MongoDB의 대용량 파일 저장 시스템 | `download_labeled_layers.py` |
| **MeanIoU** | 평균 Intersection over Union (정확도 지표) | `visualization.py` |

---

## 📊 주요 하이퍼파라미터

```python
# 모델 설정
tileSize = 128                    # 타일 크기
learning_rate = 0.0008            # 모델 초기 학습률

# 연합 학습 설정
SERVER_ROUNDS = 2                 # 서버 라운드 수
LOCAL_EPOCHS = 5                  # 클라이언트당 로컬 에포크
LOCAL_BATCH_SIZE = 32             # 배치 크기
LOCAL_LEARNING_RATE = 8e-05       # 로컬 학습률

```

---

## 💻 실행 가이드

### 결함 분류 모델 학습 (권장)

```bash
# 1. 데이터 다운로드
python util_dataset/download_labeled_layers.py --metadata

# 2. 데이터셋 정리
python util_dataset/cleanup_dataset.py --data-dir data/labeled_layers --min-count 30

# 3. 결함 유형 분석 (선택사항)
python util_dataset/analyze_defect_types.py --data-dir data/labeled_layers

# 4. 결함 분류 모델 학습
python utils/defect_type_classifier.py \
    --data-dir data/labeled_layers \
    --metadata \
    --epochs 20 \
    --batch-size 16 \
    --min-count 30

# 5. 모델 테스트
python utils/test_defect_type_classifier.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/labeled_layers \
    --metadata
```

### 픽셀 단위 결함 탐지 (연합 학습)

```python
# 1. 데이터 준비 (선택사항 - MongoDB에서 다운로드)
# python util_dataset/download_labeled_layers.py --output data/labeled_layers

# 2. 데이터셋 생성
datasetImageDict, datasetMaskDict = create_dataset(
    clientIdentifierDict, 
    imagePath0, 
    imagePath1, 
    npyPath, 
    tileSize=128
)

# 3. 학습/테스트 나누기
trainClients = ['client1', 'client2', ..., 'client7']  # 7개 공장
testClients = ['client8']                              # 1개 공장 (테스트용)

# 4. 모델 초기화
model = initialize_unet()

# 5. 연합 학습 시작
model, serverWeights, lossDict, testLoss, accuracyDict, testAccuracy = \
    federated_averaging(
        model,
        SERVER_ROUNDS=2,
        LOCAL_EPOCHS=5,
        LOCAL_BATCH_SIZE=32,
        LOCAL_LEARNING_RATE=8e-05,
        clientIDs=trainClients,
        imageDict=datasetImageDict,
        segMaskDict=datasetMaskDict,
        testImages=testImages,
        testMasks=testMasks
    )

# 6. 결과 시각화
visualize_results_testset(
    model,
    datasetImageDict,
    datasetMaskDict,
    testClients,
    clientIdentifierDict
)

# 7. 모델 비교 (선택사항)
compare_results_testset(
    cl_model,  # 중앙화 학습 모델
    fl_model,  # 연합 학습 모델
    datasetImageDict,
    datasetMaskDict,
    testClients,
    clientIdentifierDict
)
```

---

## ❓ 왜 연합 학습을 쓰나요?

### 일반 학습 (Centralized Learning)
- **방식**: 모든 데이터를 한 곳에 모아서 학습
- **문제점**:
  - 데이터 프라이버시 이슈
  - 네트워크 대역폭 소모
  - 중앙 서버 부하

### 연합 학습 (Federated Learning)
- **방식**: 각 공장의 데이터는 그대로 두고, 모델 가중치만 공유
- **장점**:
  - 데이터 프라이버시 보호
  - 네트워크 부하 감소 (가중치만 전송)
  - 분산 처리 가능
- **단점**:
  - 통신 오버헤드
  - 클라이언트 간 데이터 불균형 가능

---

## 📚 추가 정보

- **논문**: Federated learning-based semantic segmentation for pixel-wise defect detection in additive manufacturing
- **데이터**: Laser Powder Bed Fusion (L-PBF) 이미지
- **목적**: 3D 프린팅 과정의 결함 자동 탐지
- **데이터 소스**: MongoDB (keties.iptime.org:50002)
- **데이터베이스 구조**:
  - 각 실험마다 별도 DB (예: `20210909_2131_D160`)
  - `LayersModelDB`: 레이어 메타데이터 (IsLabeled 필드 포함)
  - `{db_name}_vision`: GridFS로 저장된 비전 이미지

---

## 🎯 요약

### 결함 분류 모델 (새로운 방식)
1. **MongoDB에서 레이블된 이미지 다운로드** (`util_dataset/download_labeled_layers.py`)
2. **데이터셋 정리** - 소수 클래스 및 의미 없는 이름 제거 (`util_dataset/cleanup_dataset.py`)
3. **결함 유형 분석** - 데이터셋 통계 확인 (`util_dataset/analyze_defect_types.py`)
4. **CNN 모델 학습** - 결함 유형 분류 모델 학습 (`utils/defect_type_classifier.py`)
5. **모델 테스트** - 학습된 모델 평가 (`utils/test_defect_type_classifier.py`)

### 픽셀 단위 결함 탐지 (기존 방식)
1. **MongoDB에서 레이블된 이미지 다운로드** (`util_dataset/download_labeled_layers.py`)
2. **이미지를 작은 조각으로 나눔** (128×128 타일) (`image_processing.py`)
3. **각 공장별로 데이터 정리** (8개 클라이언트) (`dataset_functions.py`)
4. **U-Net 모델 생성** (3클래스 분류) (`unet.py`)
5. **각 공장에서 학습 → 서버에서 평균 → 반복** (연합 학습) (`federated_averaging.py`)
6. **결과 확인 및 시각화** (`visualization.py`)

---

## 🔧 의존성 패키지

```txt
# 기본 패키지
numpy          # 수치 연산
Pillow         # 이미지 처리
matplotlib     # 시각화
tqdm           # 진행률 표시

# 딥러닝 프레임워크
tensorflow     # U-Net 모델 학습 (연합 학습)
torch          # PyTorch (결함 분류 모델)
torchvision    # PyTorch 비전 유틸리티

# 데이터베이스
pymongo        # MongoDB 클라이언트

# 기타
requests       # HTTP 요청 (필요시)
scikit-learn   # 머신러닝 유틸리티 (테스트용)
```

---

## 📝 파일별 함수 목록

### `image_processing.py`
- `split_image()`: 이미지를 타일로 분할
- `unsplit_image()`: 타일을 원본 이미지로 복원
- `unsplit_image_mask()`: 타일 마스크를 원본 마스크로 복원
- `preprocess_image()`: 이미지 전처리 및 타일링

### `dataset_functions.py`
- `create_dataset()`: 클라이언트별 데이터셋 생성
- `unwrap_client_data()`: 여러 클라이언트 데이터 결합

### `unet.py`
- `initialize_unet()`: U-Net 모델 생성 및 컴파일

### `federated_averaging.py`
- `federated_averaging()`: FedAvg 연합 학습 알고리즘 실행

### `visualization.py`
- `visualize_results_testset()`: 테스트셋 결과 시각화
- `compare_results_testset()`: CL vs FL 모델 비교 시각화

### `util_dataset/download_labeled_layers.py`
- `parse_args()`: 명령줄 인자 파싱
- `build_client()`: MongoDB 클라이언트 생성
- `resolve_databases()`: 처리할 DB 목록 결정
- `ensure_collections()`: 컬렉션 확인 및 GridFS 생성
- `truthy_filter()`: IsLabeled 필터 생성
- `doc_to_filename()`: 문서를 파일명으로 변환
- `write_bytes()`: 바이트 데이터 저장
- `write_metadata()`: 메타데이터 JSON 저장
- `download_for_db()`: DB별 이미지 다운로드
- `main()`: 메인 실행 함수

### `util_dataset/cleanup_dataset.py`
- `extract_defect_types_from_metadata()`: JSON 메타데이터에서 결함 유형 추출
- `is_meaningless_name()`: 의미 없는 이름 확인
- `cleanup_dataset()`: 데이터셋 정리 및 삭제 메인 함수
- `main()`: CLI 인터페이스

### `util_dataset/analyze_defect_types.py`
- `extract_defect_types_from_metadata()`: JSON 메타데이터에서 결함 유형 추출
- `analyze_defect_dataset()`: 데이터셋의 결함 종류 분석

### `utils/defect_type_classifier.py`
- `extract_defect_types_from_metadata()`: 메타데이터에서 결함 유형 추출
- `analyze_defect_types()`: 결함 유형 분석 및 매핑 생성
- `DefectTypeDataset`: PyTorch 데이터셋 클래스
- `DefectTypeClassifier`: CNN 분류 모델
- `train_defect_classifier()`: 모델 학습 함수
- `main()`: 메인 실행 함수

### `utils/test_defect_type_classifier.py`
- `test_model()`: 모델 테스트 및 예측
- `analyze_results()`: 결과 분석 및 통계
- `visualize_results()`: 혼동 행렬 시각화
