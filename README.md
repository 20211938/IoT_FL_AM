<<<<<<< HEAD
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
│   └── download_labeled_layers.py   # MongoDB에서 레이블된 이미지 다운로드
└── data/                             # 이미지 데이터 저장소
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

### 6. `utils/download_labeled_layers.py` - MongoDB 이미지 다운로드

## 🔄 전체 워크플로우

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

# MongoDB 설정
MONGODB_HOST = "keties.iptime.org"
MONGODB_PORT = 50002
MONGODB_USER = "KETI_readAnyDB"
MONGODB_PASSWORD = "madcoder"
MONGODB_AUTH_DB = "admin"
```

---

## 💻 노트북 실행 순서

```python
# 1. 데이터 준비 (선택사항 - MongoDB에서 다운로드)
# python utils/download_labeled_layers.py --output data/labeled_layers

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

1. **MongoDB에서 레이블된 이미지 다운로드** (`download_labeled_layers.py`)
2. **이미지를 작은 조각으로 나눔** (128×128 타일) (`image_processing.py`)
3. **각 공장별로 데이터 정리** (8개 클라이언트) (`dataset_functions.py`)
4. **U-Net 모델 생성** (3클래스 분류) (`unet.py`)
5. **각 공장에서 학습 → 서버에서 평균 → 반복** (연합 학습) (`federated_averaging.py`)
6. **결과 확인 및 시각화** (`visualization.py`)

---

## 🔧 의존성 패키지

```txt
numpy          # 수치 연산
Pillow         # 이미지 처리
matplotlib     # 시각화
tensorflow     # 딥러닝 프레임워크
pymongo        # MongoDB 클라이언트
requests       # HTTP 요청 (필요시)
tqdm           # 진행률 표시
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

### `download_labeled_layers.py`
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
=======
# IoT_FL_AM
>>>>>>> 3cc1f3221d00f8beb508979c2349d4cb2d5ff233
