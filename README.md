# FLAM 프로젝트: 3D 프린팅 결함 탐지 및 분류 시스템

## 🎯 이 프로젝트가 하는 일

**3D 프린팅 과정에서 결함을 탐지하고 분류하는 2단계 AI 시스템**을 구축하는 프로젝트입니다.

### 전체 시스템 구조

```
입력 이미지
    ↓
[1단계: U-Net 모델]
    → 픽셀 단위로 결함 영역 탐지
    → 출력: 결함이 있는 영역 (마스크)
    ↓
[2단계: CNN 분류 모델]
    → 탐지된 결함 영역을 입력받아
    → 출력: 결함 유형 분류 (Super Elevation, Fail, Recoater Streaking 등)
```

## 🚀 전체 시스템 학습 및 동작 흐름

### 시스템 전체 흐름

이 프로젝트는 **2단계로 구성된 결함 탐지 및 분류 시스템**입니다:

```
[학습 단계]
1. U-Net 모델 학습 → 결함 위치 탐지 학습
2. CNN 분류 모델 학습 → 결함 유형 분류 학습



## 🔍 1단계: U-Net을 이용한 픽셀 단위 결함 탐지

### U-Net 모델이 하는 일

U-Net은 **이미지의 각 픽셀을 분석하여 결함이 있는 위치를 찾아내는 모델**입니다. 이미지 전체를 입력받아 픽셀 단위로 세 가지 영역을 구분합니다:
- **파우더 (0)**: 파우더 영역
- **부품 (1)**: 정상 부품 영역  
- **결함 (2)**: 결함이 있는 영역

### U-Net의 동작 원리

U-Net은 **인코더-디코더 구조**를 가진 딥러닝 모델입니다:

1. **인코더 (Encoder)**: 이미지를 분석하여 특징을 추출
   - 이미지를 점점 작은 크기로 압축하면서 중요한 특징을 학습
   - 여러 층의 합성곱(Convolution) 레이어를 통해 이미지의 패턴을 이해

2. **디코더 (Decoder)**: 추출된 특징을 바탕으로 원본 크기로 복원
   - 압축된 특징을 다시 확대하여 원본 이미지 크기로 복원
   - 각 픽셀에 대해 "파우더/부품/결함" 중 하나를 예측

3. **Skip Connection**: 인코더의 중간 결과를 디코더에 직접 전달
   - 이미지의 세부 정보를 보존하여 정확한 픽셀 단위 분류 가능
   - U자 형태의 구조로 인해 "U-Net"이라는 이름이 붙음

### 출력 결과

U-Net의 출력은 **원본 이미지와 동일한 크기의 마스크**입니다:
- 각 픽셀마다 0(파우더), 1(부품), 2(결함) 중 하나의 값이 할당됨
- 결함 영역(값이 2인 픽셀)을 시각화하면 결함의 위치와 형태를 확인할 수 있음

### 데이터셋 및 학습 방식

- **데이터 요구사항**: 픽셀 단위로 마스킹된 전처리 데이터(npy 파일) 필요
- **현재 상황**: 전처리된 데이터가 충분하지 않아 **공개된 작은 데이터셋만 사용**
- **학습 방식**: 연합 학습(Federated Learning)을 통해 여러 공장의 데이터를 활용하되, 실제 데이터는 공유하지 않고 모델 가중치만 공유

### U-Net의 역할

U-Net은 **"결함이 어디에 있는가?"**를 답하는 모델입니다. 결함의 위치를 정확히 찾아내지만, **"어떤 종류의 결함인가?"**는 알려주지 않습니다.

---

## 🎓 2단계: CNN 기반 결함 유형 분류 모델

### 다중 레이블 분류

이 모델은 **다중 레이블 분류 모델**입니다. 일반적인 분류 모델과 달리, **한 이미지에 여러 결함 유형을 동시에 예측**할 수 있습니다.

#### 다중 레이블 분류 동작 방식

1. **입력**: 결함이 포함된 이미지
2. **처리**: ResNet34 모델로 이미지 특징 추출 및 분석
3. **출력**: 각 결함 유형별 확률 값
   - Super Elevation: 0.96
   - Fail: 0.94
   - Recoater Streaking: 0.12
   - Laser capture timing error: 0.05
   - ...
4. **최종 예측**: Threshold(0.5)를 기준으로 활성화
   - 0.5 이상이면 해당 결함 유형이 있다고 판단
   - 결과: [Super Elevation: 있음, Fail: 있음, Recoater Streaking: 없음, ...]

#### 실제 예시

```
입력 이미지: layer0003.jpg
메타데이터에 포함된 결함:
  - TagBox 1: "Fail"
  - TagBox 2: "Super Elevation"
  - TagBox 3: "Recoater Streaking"

실제 레이블 (멀티-핫 벡터):
[0, 1, 1, 1, 0, 0]  # 3개 결함 모두 활성화

모델 예측:
확률: [0.11, 0.96, 0.94, 0.92, 0.18, 0.31]
Threshold(0.5): [0, 1, 1, 1, 0, 0]  # 정확한 예측 ✓
```

### 모델의 동작 방식

1. **입력**: U-Net이 탐지한 결함 영역 (또는 결함이 포함된 이미지)
2. **처리**: ResNet 기반 CNN 모델로 이미지의 특징을 분석
3. **출력**: 결함 유형 분류 결과
   - Super Elevation (표면 높이 상승)
   - Recoater Streaking (Recoater 스트리킹)
   - Fail (일반 실패)
   - Laser capture timing error (레이저 캡처 타이밍 오류)
   - 등등...



#### 주요 특징

- **다중 레이블 분류**: 한 이미지에 여러 결함 유형을 동시에 예측
- **DepositionImageModel 전용**: 현재 데이터셋은 DepositionImageModel에만 결함이 있음
- **멀티-핫 인코딩**: 각 이미지의 모든 결함 유형을 벡터로 표현
- **BCEWithLogitsLoss**: 다중 레이블 분류에 적합한 손실 함수 사용


### 결함 종류 (Defect Types)

현재 데이터셋에는 **4가지 결함 유형**이 있습니다 (IsLabeled=true인 경우):

| 순위 | 결함 유형 | 샘플 수 | 비율 | 설명 |
|------|----------|--------|------|------|
| 1 | **Recoater Streaking** | 3,455개 | 36.88% | Recoater로 인한 스트리킹 결함 |
| 2 | **Super Elevation** | 3,381개 | 36.09% | 표면이 정상보다 높게 솟아오른 결함 |
| 3 | **Spatter** | 1,316개 | 14.05% | 스패터 결함 |
| 4 | **Humping** | 1,215개 | 12.97% | 험핑 결함 |

**참고사항**:
- **다중 레이블 분류**: 한 이미지에 여러 결함 유형이 동시에 존재할 수 있으며, 모델은 모든 결함을 동시에 예측합니다
- 총 샘플 수가 파일 수보다 많은 이유는 하나의 이미지가 여러 결함 유형을 포함할 수 있기 때문입니다
- 현재 데이터셋은 **DepositionImageModel에만 결함이 있으며**, ScanningImageModel에는 결함이 없습니다
- **클래스 불균형**: 약 2.8:1 (Recoater Streaking vs Humping) - 이전 데이터셋(15.6:1)보다 불균형이 완화됨

### 데이터셋 통계

- **결함 유형 수**: 4개 (Recoater Streaking, Super Elevation, Spatter, Humping)
- **클래스 불균형 비율**: 약 2.8:1 (가장 많은 클래스 vs 가장 적은 클래스)
- **클래스 가중치**: 코드에서 자동으로 계산되어 소수 클래스에 더 높은 가중치 부여
- **데이터 분할**: 70% 학습 / 15% 검증 / 15% 테스트
  - **랜덤 셔플**: 데이터를 랜덤하게 섞은 후 분할하여 불균형 방지
  - 학습 데이터: 모델 학습용
  - 검증 데이터: 모델 선택 및 조기 종료용
  - 테스트 데이터: 최종 평가용 (학습 중 절대 사용하지 않음)
  - **레이블 분포 확인**: 각 분할의 단일/다중 레이블 비율 자동 확인 및 경고 (5% 이상 차이 시 경고)

### 모델 구조

CNN 분류 모델은 **ResNet34 기반 Transfer Learning 모델**입니다:

1. **입력**: 결함이 포함된 이미지 (224×224 크기)
2. **특징 추출**: ResNet34 백본을 통해 이미지의 특징 추출
   - ImageNet에서 사전 학습된 가중치 사용
   - 이미지의 패턴과 특징을 이해
3. **분류**: 추출된 특징을 바탕으로 결함 유형 분류
   - 각 결함 유형에 대해 독립적으로 확률 계산
   - 여러 결함 유형이 동시에 존재할 수 있음을 반영
4. **출력**: 각 결함 유형별 확률 값
   - 예: [Super Elevation: 0.96, Fail: 0.94, Recoater Streaking: 0.92, ...]
   - Threshold(0.5)를 기준으로 최종 예측 결정


### 기본 하이퍼파라미터

```python
epochs = 300                    # 학습 에포크 수
batch_size = 32                 # 배치 크기
learning_rate = 0.0001          # 학습률
min_count = 10                  # 최소 샘플 수 (이보다 적으면 클래스 제거)
image_size = 224                # 이미지 크기
weight_decay = 1e-4             # Weight Decay (정규화)
label_smoothing = 0.1           # Label Smoothing (단일 레이블용, 다중 레이블에서는 미사용)
scheduler_type = "cosine"       # 학습률 스케줄러 (cosine, cosine_warmup, plateau, step)
early_stopping_threshold = 98.0 # 조기 종료 기준 (검증 정확도 %)
use_data_augmentation = True     # 데이터 증강 사용 여부
model_name = "resnet34"         # 모델 아키텍처 (resnet18, resnet34, resnet50)
pretrained = True               # 사전 학습된 가중치 사용 여부
```

#### 손실 함수

- **BCEWithLogitsLoss + 클래스 가중치**: 다중 레이블 분류에 적합
  - 각 클래스를 독립적인 이진 분류 문제로 처리
  - Sigmoid + Binary Cross Entropy 결합
  - 여러 결함이 동시에 존재하는 상황에 최적화
  - **클래스 불균형 처리**: 각 클래스의 샘플 수를 기반으로 자동 가중치 계산
    - 소수 클래스(Spatter, Humping)에 더 높은 가중치 부여
    - 학습 시 클래스별 가중치가 출력되어 확인 가능

### 실행 명령어

```bash
# 기본 사용
python utils/CNN/defect_type_classifier.py --data-dir data/

# 모델 테스트
python utils/CNN/test_defect_type_classifier.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data
```

### 평가 지표

- **완전 일치 정확도**: 모든 레이블이 정확히 일치하는 샘플의 비율
  - 예측된 모든 결함 유형이 실제와 정확히 일치해야 정확한 것으로 간주
  - Threshold: 0.5 (각 클래스별 확률이 0.5 이상이면 활성화)
- **부분 정확도**: 예측된 레이블 중 실제 레이블과 일치하는 비율
  - 일부 레이블만 맞아도 점수를 부여하여 더 관대한 평가
- **F1 스코어**: Precision과 Recall을 결합한 지표
  - Precision: 예측된 레이블 중 실제로 맞는 비율
  - Recall: 실제 레이블 중 올바르게 예측된 비율
  - 검증 및 테스트 단계에서 상세 지표 출력
- **조기 종료 기준**: 검증 정확도 98% 도달 시 학습 중단
- **데이터 분할 전략**:
  - **랜덤 셔플**: 데이터를 랜덤하게 섞은 후 분할하여 불균형 방지
  - **학습 데이터 (70%)**: 모델 학습에 사용
  - **검증 데이터 (15%)**: 학습 중 모델 성능 모니터링, 최적 모델 선택, 조기 종료 판단에 사용
  - **테스트 데이터 (15%)**: 학습 완료 후 최종 성능 평가에만 사용 (학습 중 절대 사용하지 않음)
  - **레이블 분포 확인**: 각 분할의 단일/다중 레이블 비율을 자동으로 확인하고, 5% 이상 차이가 나면 경고 메시지 출력

---

## 🔄 전체 시스템 워크플로우

### 실제 발표 시나리오

```
[입력] 3D 프린팅 레이어 이미지
    ↓
[1단계: U-Net 모델]
    → 픽셀 단위 분석
    → 결함 영역 탐지 (마스크 생성)
    → 출력: "여기에 결함이 있습니다"
    ↓
[2단계: CNN 분류 모델]
    → 탐지된 결함 영역 입력
    → 결함 유형 분석
    → 출력: "이 결함은 Super Elevation입니다"
```

### U-Net 모델 학습 워크플로우

```
1. 데이터 준비
   └─ 전처리된 npy 파일 (픽셀 단위 마스킹 데이터)
   └─ 현재: 공개된 작은 데이터셋만 사용
   
2. 이미지 전처리
   └─ 큰 이미지를 128×128 타일로 분할
   
3. 연합 학습
   └─ 여러 공장(클라이언트)에서 로컬 학습
   └─ 서버에서 가중치 평균
   └─ 반복하여 모델 개선
   
4. 결과 평가
   └─ 픽셀 단위 정확도 측정
   └─ 결함 탐지 성능 평가
```

### CNN 결함 분류 모델 학습 워크플로우

```
1. 데이터 다운로드
   └─ MongoDB에서 레이블된 이미지 다운로드
   └─ DepositionImageModel의 TagBoxes에서 결함 유형 추출
   
2. 데이터셋 정리
   └─ 소수 클래스 및 의미 없는 이름 제거
   └─ 학습에 적합한 데이터만 선별
   
3. 데이터 분할 (70:15:15)
   └─ 랜덤 셔플 후 분할: 데이터를 랜덤하게 섞은 후 분할하여 불균형 방지
   └─ 학습 데이터: 70% (모델 학습용)
   └─ 검증 데이터: 15% (모델 선택 및 조기 종료용)
   └─ 테스트 데이터: 15% (최종 평가용, 학습 중 절대 사용 안 함)
   └─ 레이블 분포 확인: 각 분할의 단일/다중 레이블 비율 자동 확인 및 경고
   
4. 결함 분류 모델 학습
   └─ ResNet 기반 다중 레이블 분류 모델 학습
   └─ 한 이미지에 여러 결함 유형이 있을 수 있음을 반영
   └─ 검증 데이터로 모델 성능 모니터링 및 조기 종료
   
5. 최종 평가
   └─ 테스트 데이터로 최종 성능 평가
   └─ 다중 레이블 정확도 측정
```

### 두 모델의 관계

- **U-Net**: 결함의 위치를 찾는 모델 (어디에?)
- **CNN 분류 모델**: 결함의 종류를 분류하는 모델 (무엇인가?)
- **연결 방식**: U-Net의 출력(결함 영역)을 CNN 분류 모델의 입력으로 사용
- **독립성**: 두 모델은 서로 다른 데이터셋으로 학습되며, U-Net 모델은 수정하지 않고 별도 분류 모델을 추가

---

---

## 🔧 환경 설정

### 가상 환경 활성화 (Windows PowerShell)

PowerShell에서 가상 환경을 활성화할 때 실행 정책 오류가 발생할 수 있습니다. 다음 방법 중 하나를 사용하세요:

#### 방법 1: 현재 세션에만 실행 정책 변경 (권장)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv311\Scripts\Activate.ps1
```

또는 한 줄로 실행:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\venv311\Scripts\Activate.ps1
```

#### 방법 2: activate.bat 사용

```powershell
cmd /c "venv311\Scripts\activate.bat && powershell"
```

#### 방법 3: Python 직접 실행 (가장 간단)

가상 환경을 활성화하지 않고 직접 Python을 실행:
```powershell
.\venv311\Scripts\python.exe utils/CNN/defect_type_classifier.py --data-dir data/labeled_layers
```

#### 방법 4: 영구적으로 실행 정책 변경 (관리자 권한 필요)

관리자 권한으로 PowerShell을 열고:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

이후 `.\venv311\Scripts\Activate.ps1`만 실행하면 됩니다.

### 가상 환경 활성화 확인

가상 환경이 활성화되면 프롬프트 앞에 `(venv311)`이 표시됩니다:
```powershell
(venv311) PS D:\iot\FLAM>
```

Python 버전 확인:
```powershell
python --version
```

---

## 💻 실행 가이드

### 결함 분류 모델 학습 (권장)

```bash
# 1. 데이터 다운로드
python utils/Dataset/download_labeled_layers.py --metadata

# 2. 데이터셋 정리
python utils/Dataset/cleanup_dataset.py --data-dir data/labeled_layers --min-count 30

# 3. 결함 유형 분석 (선택사항)
python utils/Dataset/analyze_defect_types.py --data-dir data/labeled_layers

# 4. 결함 분류 모델 학습

python utils/CNN/defect_type_classifier.py --data-dir data/labeled_layers


python utils/CNN/defect_type_classifier.py \
    --data-dir data/labeled_layers \
    --epochs 20 \
    --batch-size 16 \
    --min-count 30

# 5. 모델 테스트
python utils/CNN/test_defect_type_classifier.py \
    --checkpoint checkpoints/best_model.pth \
    --data-dir data/labeled_layers
```

### U-Net 모델 학습 (연합 학습)

U-Net 모델은 전처리된 npy 파일이 필요하며, 현재는 공개된 작은 데이터셋만 사용합니다. Jupyter Notebook (`Federated_learning.ipynb`)에서 연합 학습을 통해 모델을 학습합니다.

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
- **데이터 소스**: MongoDB (연결 정보는 `utils/Dataset/download_labeled_layers.py`에서 설정)
- **데이터베이스 구조**:
  - 각 실험마다 별도 DB (예: `20210909_2131_D160`)
  - `LayersModelDB`: 레이어 메타데이터 (IsLabeled 필드 포함)
  - `{db_name}_vision`: GridFS로 저장된 비전 이미지

---

## 🎯 시스템 요약

### 전체 시스템 구조

이 프로젝트는 **2단계 결함 탐지 및 분류 시스템**입니다:

#### 1단계: U-Net 모델 (결함 탐지)
- **목적**: 이미지에서 결함이 있는 위치를 픽셀 단위로 찾기
- **입력**: 3D 프린팅 레이어 이미지
- **출력**: 각 픽셀의 분류 결과 (파우더/부품/결함)
- **데이터**: 전처리된 npy 파일 필요 (공개된 작은 데이터셋 사용)
- **학습 방식**: 연합 학습 (Federated Learning)

#### 2단계: CNN 분류 모델 (결함 유형 분류)
- **목적**: U-Net이 찾은 결함이 어떤 종류인지 분류
- **입력**: U-Net이 탐지한 결함 영역 (또는 결함이 포함된 이미지)
- **출력**: 결함 유형 분류 (Super Elevation, Fail, Recoater Streaking 등)
- **데이터**: MongoDB에서 다운로드한 레이블된 이미지
- **특징**: 다중 레이블 분류 (한 이미지에 여러 결함 유형 가능)

### 개발 철학

- **원래 목표**: U-Net 모델을 개선하여 결함 탐지와 분류를 동시에 수행
- **최종 결정**: U-Net 모델을 수정하지 않고, 별도의 CNN 분류 모델을 만들어 U-Net의 출력에 연결
- **장점**: 
  - U-Net 모델의 안정성 유지
  - 분류 모델을 독립적으로 개선 가능
  - 각 모델을 서로 다른 데이터셋으로 학습 가능

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

<<<<<<< HEAD
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

### `utils/Dataset/download_labeled_layers.py`
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
>>>>>>> feature-htk
