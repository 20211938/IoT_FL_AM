# Federated Learning for Additive Manufacturing (FLAM)

이 저장소는 연합 학습 기반의 시맨틱 세그멘테이션을 사용한 적층 제조 결함 검출 프로젝트입니다.

## 프로젝트 개요

본 프로젝트는 여러 클라이언트에 분산된 데이터를 사용하여 연합 학습(Federated Learning) 방식으로 U-Net 모델을 학습시켜 적층 제조 공정의 픽셀 단위 결함을 검출합니다.

## 환경 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. GPU 확인 (선택사항)

GPU를 사용하려면 CUDA와 cuDNN이 설치되어 있어야 합니다. Jupyter Notebook의 첫 번째 셀을 실행하면 GPU 사용 가능 여부가 자동으로 확인됩니다.

## 데이터 준비 및 통합

### 1. 데이터 구조

원본 데이터는 `data` 폴더에 다음과 같은 구조로 저장되어 있습니다:

```
data/
├── 0/              # Post Spreading Images (.jpg)
├── 1/              # Post Fusion Images (.jpg)
└── annotations/    # Segmentation Masks (.npy)
```

또는 여러 데이터셋이 있는 경우:

```
data/
├── dataset1/
│   ├── 0/
│   ├── 1/
│   └── annotations/
├── dataset2/
│   ├── 0/
│   ├── 1/
│   └── annotations/
└── ...
```

### 2. 데이터 통합

여러 데이터셋을 하나의 `merged_data` 폴더로 통합합니다:

```bash
python new_utils/merge_data_files.py
```

이 스크립트는:
- `data` 폴더의 모든 이미지 파일을 찾아서
- `merged_data` 폴더에 `000001`, `000002`, ... 형식으로 번호를 매겨 저장합니다
- `0`, `1`, `annotations` 폴더 구조를 유지합니다

**출력 예시:**
```
소스 폴더: D:\iot\FLAM\data
출력 폴더: D:\iot\FLAM\merged_data
데이터 구조: data/0/, data/1/, data/annotations/ 직접 구조 감지
  693개의 이미지 파일 발견

총 693개의 파일 그룹 발견
파일 복사 시작...
  진행 중: 100개 파일 처리 완료
  진행 중: 200개 파일 처리 완료
  ...

완료! 총 693개의 파일 그룹을 merged_data 폴더에 복사했습니다.
```

### 3. 대용량 이미지 제거 (선택사항)

500KB 이상의 대용량 이미지를 제거하려면:

```bash
python new_utils/remove_large_images.py
```

## 연합 학습 실행

### 1. 노트북 실행

`Federated_learning_merged_data.ipynb` 노트북을 실행합니다.

### 2. 주요 단계

#### Step 1: GPU 확인 및 설정
첫 번째 셀을 실행하여 GPU 사용 가능 여부를 확인합니다.

#### Step 2: 데이터 경로 설정
```python
imagePath0 = 'merged_data/0/'      # Post Spreading Images
imagePath1 = 'merged_data/1/'       # Post Fusion Images
npyPath = 'merged_data/annotations/' # Annotations
```

#### Step 3: 클라이언트 분배
- 총 파일 개수를 설정합니다 (예: `total_files = 693`)
- 8개 클라이언트에 균등하게 분배합니다
- 각 클라이언트는 연속된 파일 번호 범위를 할당받습니다

#### Step 4: 학습/테스트 클라이언트 설정
```python
trainClients = ['client1', 'client2', 'client3', 'client4', 
                'client5', 'client7', 'client8']
testClients = ['client6']
```

#### Step 5: 하이퍼파라미터 설정
```python
SERVER_ROUNDS = 2          # 서버 라운드 수
LOCAL_EPOCHS = 5           # 로컬 에포크 수
LOCAL_BATCH_SIZE = 32      # 배치 크기
LOCAL_LEARNING_RATE = 8e-05 # 학습률
```

#### Step 6: 모델 학습
연합 학습을 시작합니다. 각 클라이언트는 로컬에서 학습한 후 가중치를 서버로 전송하고, 서버는 가중 평균을 계산하여 전역 모델을 업데이트합니다.

### 3. 학습 결과

학습이 완료되면:
- 모델이 `saved_models/` 폴더에 자동으로 저장됩니다
- 파일명 형식: `FL_{SERVER_ROUNDS}_{LOCAL_EPOCHS}_{BATCH_SIZE}_{LR}_HoldoutPart{test_set}.h5`
- 테스트 성능(Loss, Accuracy)이 출력됩니다

## 유틸리티 스크립트

### `new_utils/merge_data_files.py`
여러 데이터셋을 하나로 통합하는 스크립트

### `new_utils/remove_large_images.py`
대용량 이미지 파일을 제거하는 스크립트 (기본값: 500KB 이상)

### `new_utils/convert_tif_to_jpg.py`
TIF 이미지를 JPG로 변환하는 스크립트

### `new_utils/data_augmentation.py`
데이터 증강을 수행하는 스크립트

## 주요 함수 (utils 폴더)

- `dataset_functions.py`: 데이터셋 생성 및 로딩 함수
- `image_processing.py`: 이미지 전처리 및 타일링 함수
- `unet.py`: U-Net 모델 초기화 함수
- `federated_averaging.py`: 연합 학습 알고리즘 구현
- `visualization.py`: 결과 시각화 함수

## 모델 로드

학습된 모델을 로드하려면:

```python
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/FL_2_5_32_8e05_HoldoutPart06.h5')
```

## 참고사항

- **GPU 사용 권장**: 학습 시간을 단축하려면 GPU를 사용하는 것을 권장합니다
- **메모리 관리**: 대용량 데이터셋의 경우 lazy loading 방식을 사용하여 메모리 사용량을 최적화합니다
- **데이터 불균형**: 각 클라이언트의 타일 개수가 다를 수 있으며, 이는 이미지 크기 차이 때문입니다. 연합 학습에서는 가중 평균을 사용하여 자동으로 처리됩니다

## 문제 해결

### GPU가 감지되지 않는 경우
- CUDA와 cuDNN이 올바르게 설치되어 있는지 확인하세요
- TensorFlow GPU 버전이 설치되어 있는지 확인하세요: `pip install tensorflow-gpu`

### 메모리 부족 오류
- 배치 크기를 줄이세요 (`LOCAL_BATCH_SIZE`)
- `remove_large_images.py`를 실행하여 대용량 이미지를 제거하세요

## 라이선스 및 인용

원본 데이터는 Oak Ridge National Laboratory에서 수집 및 컴파일되었으며, [여기](https://www.osti.gov/dataexplorer/biblio/dataset/1779073)에서 사용 가능합니다.

이 데이터를 사용하는 경우 적절히 인용해주세요:
- Dataset: doi:10.13139/ORNLNCCS/1779073
- Related work: [링크](https://www.sciencedirect.com/science/article/pii/S2214860420308253)
