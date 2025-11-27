# FLAM 프로젝트: 3D 프린팅 결함 탐지 및 분류 시스템

## 🎯 프로젝트 개요

**3D 프린팅 과정에서 결함을 탐지하고 분류하는 2단계 AI 시스템**을 구축하는 프로젝트입니다.

### 전체 시스템 구조

```
입력 이미지
    ↓
[U-Net 모델] → 결함 영역 탐지
    ↓
[CNN 분류 모델] → 결함 유형 분류
    ↓
출력: 결함 유형 (Super Elevation, Fail, Recoater Streaking 등)
```

---

## 🎓 CNN 기반 결함 유형 분류 모델

### 개요

이 모델은 **다중 레이블 분류 모델**로, 일반적인 분류 모델과 달리 **한 이미지에 여러 결함 유형을 동시에 예측**할 수 있습니다.

#### 주요 특징

- **다중 레이블 분류**: 한 이미지에 여러 결함 유형을 동시에 예측
- **DepositionImageModel 전용**: 현재 데이터셋은 DepositionImageModel에만 결함이 있음
- **멀티-핫 인코딩**: 각 이미지의 모든 결함 유형을 벡터로 표현
- **BCEWithLogitsLoss**: 다중 레이블 분류에 적합한 손실 함수 사용
- **클래스 불균형 처리**: 각 클래스의 샘플 수를 기반으로 자동 가중치 계산

### 동작 방식

1. **입력**: U-Net이 탐지한 결함 영역 또는 결함이 포함된 이미지 (224×224)
2. **처리**: ResNet34 백본을 통한 이미지 특징 추출 및 분석
   - ImageNet에서 사전 학습된 가중치 사용
   - 각 결함 유형에 대해 독립적으로 확률 계산
3. **출력**: 각 결함 유형별 확률 값
   - 예: [Super Elevation: 0.96, Fail: 0.94, Recoater Streaking: 0.12, ...]
   - Threshold(0.5)를 기준으로 최종 예측 결정

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

### 결함 종류 (Defect Types)

결함 유형은 데이터셋에서 동적으로 분석되며, 최소 샘플 수(`min_count`) 이상인 결함만 학습에 사용됩니다. 일반적으로 다음과 같은 결함 유형이 포함됩니다:

- **Super Elevation** (표면 높이 상승)
- **Fail** (일반 실패)
- **Recoater Streaking** (Recoater 스트리킹)
- **Laser capture timing error** (레이저 캡처 타이밍 오류)
- **Spatter** (스패터)
- **Humping** (험핑)
- **Normal** (정상, 기본 클래스)

> **참고**: 실제 결함 유형은 데이터셋에 따라 달라질 수 있으며, 학습 시작 시 자동으로 분석되어 출력됩니다.

### 모델 구조

CNN 분류 모델은 **ResNet34 기반 Transfer Learning 모델**입니다:

1. **백본**: ResNet34 (또는 ResNet18, ResNet50 선택 가능)
2. **특징 추출**: ImageNet 사전 학습 가중치를 사용한 특징 추출
3. **분류 헤드**: 커스텀 분류 레이어로 다중 레이블 분류 수행
4. **출력**: 각 결함 유형별 독립적인 확률 값 (Sigmoid 활성화)

### 하이퍼파라미터

#### 기본 설정

```python
epochs = 300                    # 학습 에포크 수
batch_size = 32                 # 배치 크기
learning_rate = 0.0001          # 학습률
min_count = 10                  # 최소 샘플 수 (이보다 적으면 클래스 제거)
image_size = 224                # 이미지 크기
weight_decay = 1e-4             # Weight Decay (정규화)
scheduler_type = "cosine"       # 학습률 스케줄러 (cosine, cosine_warmup, plateau, step)
early_stopping_threshold = 98.0 # 조기 종료 기준 (검증 정확도 %)
use_data_augmentation = True    # 데이터 증강 사용 여부
model_name = "resnet34"         # 모델 아키텍처 (resnet18, resnet34, resnet50)
pretrained = True               # 사전 학습된 가중치 사용 여부
```

#### 손실 함수

- **BCEWithLogitsLoss + 클래스 가중치**: 다중 레이블 분류에 적합
  - 각 클래스를 독립적인 이진 분류 문제로 처리
  - Sigmoid + Binary Cross Entropy 결합
  - 여러 결함이 동시에 존재하는 상황에 최적화
  - **클래스 불균형 처리**: 각 클래스의 샘플 수를 기반으로 자동 가중치 계산
    - 소수 클래스에 더 높은 가중치 부여
    - 학습 시 클래스별 가중치가 출력되어 확인 가능

### 평가 지표

- **완전 일치 정확도 (Subset Accuracy)**: 모든 레이블이 정확히 일치하는 샘플의 비율
  - 예측된 모든 결함 유형이 실제와 정확히 일치해야 정확한 것으로 간주
  - Threshold: 0.5 (각 클래스별 확률이 0.5 이상이면 활성화)
- **부분 정확도 (Hamming Accuracy)**: 예측된 레이블 중 실제 레이블과 일치하는 비율
  - 일부 레이블만 맞아도 점수를 부여하여 더 관대한 평가
- **F1 스코어**: Precision과 Recall을 결합한 지표
  - **Precision**: 예측된 레이블 중 실제로 맞는 비율
  - **Recall**: 실제 레이블 중 올바르게 예측된 비율
  - Macro, Micro, Weighted 평균 제공
  - 검증 및 테스트 단계에서 상세 지표 출력
- **조기 종료 기준**: 검증 정확도 98% 도달 시 학습 중단

#### 데이터 분할 전략

- **랜덤 셔플**: 데이터를 랜덤하게 섞은 후 분할하여 불균형 방지
- **학습 데이터 (70%)**: 모델 학습에 사용
- **검증 데이터 (15%)**: 학습 중 모델 성능 모니터링, 최적 모델 선택, 조기 종료 판단에 사용
- **테스트 데이터 (15%)**: 학습 완료 후 최종 성능 평가에만 사용 (학습 중 절대 사용하지 않음)
- **레이블 분포 확인**: 각 분할의 단일/다중 레이블 비율을 자동으로 확인하고, 5% 이상 차이가 나면 경고 메시지 출력

---

## 🔄 학습 워크플로우

### CNN 결함 분류 모델 학습 프로세스

```
1. 데이터 다운로드
   └─ MongoDB에서 레이블된 이미지 다운로드
   └─ DepositionImageModel의 TagBoxes에서 결함 유형 추출
   
2. 데이터셋 정리
   └─ 소수 클래스 및 의미 없는 이름 제거
   └─ 학습에 적합한 데이터만 선별
   
3. 데이터 분할 (70:15:15)
   └─ 랜덤 셔플 후 분할하여 불균형 방지
   └─ 학습/검증/테스트 데이터 분할
   └─ 레이블 분포 확인 및 경고
   
4. 결함 분류 모델 학습
   └─ ResNet 기반 다중 레이블 분류 모델 학습
   └─ 검증 데이터로 모델 성능 모니터링 및 조기 종료
   
5. 최종 평가
   └─ 테스트 데이터로 최종 성능 평가
   └─ 다중 레이블 정확도 측정
```

---

## 🔧 환경 설정

### 가상 환경 활성화 (Windows PowerShell)

PowerShell에서 가상 환경을 활성화할 때 실행 정책 오류가 발생할 수 있습니다.

#### 방법 1: 현재 세션에만 실행 정책 변경 (권장)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process; .\venv311\Scripts\Activate.ps1
```

#### 방법 2: 영구적으로 실행 정책 변경 (관리자 권한 필요)

관리자 권한으로 PowerShell을 열고:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

이후 `.\venv311\Scripts\Activate.ps1`만 실행

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

### 결함 분류 모델 학습

#### 1. 데이터 다운로드

```bash
python utils/Dataset/download_labeled_layers.py --metadata
```

#### 2. 데이터셋 정리

```bash
python utils/Dataset/cleanup_dataset.py --data-dir data/labeled_layers --min-count 30
```

#### 3. 결함 유형 분석 (선택사항)

```bash
python utils/Dataset/analyze_defect_types.py --data-dir data/labeled_layers
```

#### 4. 결함 분류 모델 학습

기본 설정으로 학습:

```bash
python utils/CNN/defect_type_classifier.py --data-dir data
```

커스텀 하이퍼파라미터로 학습:

```bash
python utils/CNN/defect_type_classifier.py \
    --data-dir data \
    --epochs 20 \
    --batch-size 16 \
    --min-count 30
```

#### 5. 모델 테스트

```bash
python utils/CNN/test_defect_type_classifier.py \
    --checkpoint checkpoints/defect_type_classifier_best.pth \
    --data-dir data
```

---

## 🎯 시스템 아키텍처

### 전체 시스템 구조

이 프로젝트는 **2단계 결함 탐지 및 분류 시스템**입니다:

1. **U-Net 모델**: 결함 영역 탐지
2. **CNN 분류 모델**: 결함 유형 분류

#### CNN 분류 모델 (결함 유형 분류)

- **목적**: U-Net이 찾은 결함이 어떤 종류인지 분류
- **입력**: U-Net이 탐지한 결함 영역 또는 결함이 포함된 이미지
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

## ❓ 연합 학습 (Federated Learning)

### 일반 학습 vs 연합 학습

#### 일반 학습 (Centralized Learning)

- **방식**: 모든 데이터를 한 곳에 모아서 학습
- **문제점**:
  - 데이터 프라이버시 이슈
  - 네트워크 대역폭 소모
  - 중앙 서버 부하

#### 연합 학습 (Federated Learning)

- **방식**: 각 공장의 데이터는 그대로 두고, 모델 가중치만 공유
- **장점**:
  - 데이터 프라이버시 보호
  - 네트워크 부하 감소 (가중치만 전송)
  - 분산 처리 가능
- **단점**:
  - 통신 오버헤드
  - 클라이언트 간 데이터 불균형 가능

---