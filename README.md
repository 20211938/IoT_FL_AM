# Federated Learning for Additive Manufacturing (FLAM) - 결함 검출 및 분류

## 프로젝트 개요

본 프로젝트는 2단계 파이프라인을 통해 적층 제조 공정의 결함을 검출하고 분류합니다.

## 📋 간단 요약

### 무엇을 하는가?
- **1단계**: U-Net으로 이미지에서 결함 존재 여부 검출
- **2단계**: ResNet-18로 결함이 있는 패치의 결함 유형 분류

### 학습 방식
- **알고리즘**: FedAvg (Federated Averaging)
- **모델**: ResNet-18 (ImageNet 사전 학습)
- **전략**: 라운드 0~3은 Backbone 고정, 라운드 4~는 전체 모델 학습
- **평가**: 각 클라이언트가 Train(60%)/Val(20%)/Test(20%)로 분할하여 평가

## 2단계 파이프라인

### 1단계: U-Net 기반 결함 검출

- **목적**: 이미지 전체를 입력받아 결함이 있는지 없는지를 판단
- **입력**: 이미지 전체 (Post Spreading + Post Fusion 이미지)
- **출력**: 결함 존재 여부 판단 (이진 분류: 결함 있음/없음)
- **학습 데이터**: 
  - 전처리된 npy 파일 사용 (모든 결함 유형이 클래스 2로 통합됨)
  - 연합 학습(Federated Learning) 방식으로 학습

### 2단계: Patch-based CNN 기반 결함 유형 분류

- **목적**: 결함이 발견된 이미지의 패치에 대해 결함 유형을 분류
- **입력**: 이미지 패치 (Post Spreading + Post Fusion 이미지를 그리드로 분할)
- **출력**: 결함 유형 분류 (다중 클래스 분류)
- **모델**: ResNet-18 기반 분류기 (ImageNet 사전 학습 가중치 사용)
- **학습 데이터**:
  - 원본 npy 파일에서 각 패치의 결함 유형 숫자 값들을 확인
  - **정상 부분 제외**: `0`(배경)과 `1`(파트)은 정상 부분이므로 제외
  - **결함 인식**: 나머지 값들(`-1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `11`, `14`, `255` 등)은 모두 결함으로 인식
  - **대표 결함 선택**: 각 패치의 마스크에서 0과 1을 제외한 결함 중 **가장 픽셀 수가 많은 결함 유형**을 해당 패치의 레이블로 사용
  - 데이터 증강: Albumentations를 사용한 데이터 증강 (회전, 플립, 밝기/대비 조정)

## 데이터 구조

### 디렉토리 구조

```
data/
├── 0/                    # Scanning Image 이미지 (.jpg)
├── 1/                    # Deposition Image 이미지 (.jpg)
└── annotations/          # 원본 마스크 파일 (.npy)
    ├── 0: 배경 (정상, 결함 아님)
    ├── 1: 파트 (정상, 결함 아님)
    └── -1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14 등: 결함 유형
```


## 워크플로우

### 전체 워크플로우 다이어그램

```mermaid
flowchart TD
    Start([시작]) --> Step1[1. U-Net 학습<br/>전처리된 데이터로<br/>결함 검출 모델 학습]
    
    Step1 --> Step2[2. 결함 검출<br/>U-Net으로 모든 이미지<br/>결함 존재 여부 판단]
    
    Step2 --> Step3[3. 이미지 분할<br/>각 이미지를 그리드로<br/>패치 생성 (patch_size에 따라 자동 결정)]
    
    Step3 --> Step4[4. 패치 선별<br/>결함이 포함된 패치만<br/>선별 및 리사이즈 256x256]
    
    Step4 --> Step5[5. 레이블 생성<br/>각 패치의 마스크에서<br/>가장 많은 결함 유형 선택]
    
    Step5 --> Step6[6. 데이터 증강<br/>Albumentations로<br/>데이터 증강 적용]
    
    Step6 --> Step7[7. CNN 학습<br/>ResNet-18 기반<br/>연합 학습 수행]
    
    Step7 --> Step8[8. 분산 평가<br/>각 클라이언트가<br/>자신의 테스트 데이터로 평가]
    
    Step8 --> End([완료])
    
    style Start fill:#e1f5ff
    style End fill:#e1ffe1
    style Step1 fill:#fff4e1
    style Step2 fill:#fff4e1
    style Step3 fill:#ffe1f5
    style Step4 fill:#ffe1f5
    style Step5 fill:#ffe1f5
    style Step6 fill:#e1f5ff
    style Step7 fill:#fff4e1
    style Step8 fill:#e1ffe1
```

## 설치 및 실행

### 환경 설정

1. **필수 패키지 설치**

- 필수_패키지_설치_가이드.txt 파일 확인인

### 실행 방법

#### 1단계: U-Net 학습

```python
# U_net_Federated_Learning.ipynb 실행
# 결함 검출 모델 학습
```

#### 2단계: Patch-based CNN 학습

```python
# PatchBased_CNN_Learning.ipynb 실행
# 결함 유형 분류 모델 학습
```

주요 설정:
- `SERVER_ROUNDS`: 서버 라운드 수 (기본값: 20)
- `LOCAL_EPOCHS`: 클라이언트별 로컬 에포크 수 (기본값: 3)
- `LOCAL_BATCH_SIZE`: 배치 크기 (기본값: 32)
- `LOCAL_LEARNING_RATE`: 학습률 (기본값: 1e-4)
- `FREEZE_EPOCHS`: Backbone 고정 기간 (기본값: 3)

## 주요 기능

### 연합 학습 (Federated Learning)

- **FedAvg 알고리즘**: 클라이언트별 로컬 업데이트를 가중 평균하여 서버 모델 업데이트
- **분산 평가**: 각 클라이언트가 자신의 데이터로 평가하여 프라이버시 보호
- **가중 평균**: 클라이언트별 데이터 크기에 따른 가중치 적용

### 패치 기반 분류

- **적응형 그리드 분할**: `patch_size`에 따라 그리드 크기 자동 결정 (기본값 256x256일 때 4x4 그리드, 16개 패치)
- **결함 패치 선별**: 최소 결함 비율 및 신뢰도 기준으로 결함 패치만 선별
- **동적 리사이즈**: 각 패치를 256x256 크기로 통일 (기본값, `PATCH_SIZE` 파라미터로 조정 가능)

### 데이터 증강

- **Albumentations 파이프라인**: 회전, 플립, 밝기/대비 조정
  - `RandomRotate90`: 90도 회전 (확률: 0.5)
  - `HorizontalFlip`: 수평 플립 (확률: 0.5)
  - `VerticalFlip`: 수직 플립 (확률: 0.5)
  - `RandomBrightnessContrast`: 밝기/대비 조정 (brightness_limit=0.2, contrast_limit=0.2, 확률: 0.5)
- **2채널 이미지 지원**: Post Spreading + Post Fusion 이미지 결합
- **결함 유형별 차등 증강**: 결함 유형별 샘플 수에 따라 자동으로 증강 배수 조정

### 모델 구조

- **ResNet-18 Backbone**: ImageNet 사전 학습 가중치 활용
- **전이 학습**: 초기 몇 라운드는 Backbone 고정, 이후 fine-tuning
  - 라운드 0~3: Backbone 고정, 분류 헤드만 학습
  - 라운드 4~: Backbone 해제, 전체 모델 학습
- **2채널 입력**: 첫 번째 Conv 레이어를 2채널로 수정 (Post Spreading + Post Fusion)
- **동적 클래스 수**: 데이터에서 발견된 결함 유형 수에 따라 자동으로 클래스 수 결정

## 최근 업데이트 및 개선사항

### 1. 결함 유형 전용 레이블 매핑

- **정상 부분 제외**: `0`(배경), `1`(파트), `-1`을 제외하고 결함 유형만 분류 대상으로 사용
- **결함 유형만 학습**: 실제 결함 유형(2, 3, 4, 5, 6, 7, 8, 9, 11, 14 등)에 대해서만 분류 모델 학습
- **레이블 매핑 최적화**: 결함 유형만 포함하는 레이블 매핑을 생성하여 모델 학습 효율성 향상

### 2. 전체 데이터 패치 생성 및 증강 파이프라인

- **통합 패치 생성**: `create_all_patches_with_augmentation` 함수로 전체 데이터에서 패치를 일괄 생성
- **자동 증강 적용**: 패치 생성과 동시에 Albumentations 기반 데이터 증강 자동 적용
- **결함 유형별 증강 배수 조정**: 결함 유형별 샘플 수에 따라 차등 증강 적용
  - 적은 샘플 수의 결함 유형은 더 높은 증강 배수 적용 (최대 12배)
  - 많은 샘플 수의 결함 유형은 낮은 증강 배수 적용 (최소 2배)
- **증강 전후 통계 제공**: 증강 전후 결함 유형별 패치 개수 및 비율 비교 정보 제공
- **패치 필터링**: 최소 결함 비율(`MIN_DEFECT_RATIO`) 및 최소 신뢰도(`MIN_CONFIDENCE`) 기준으로 품질이 낮은 패치 자동 제외

### 3. Non-IID 데이터 분산 방식

- **결함 유형 기반 분산**: `distribute_patches_non_iid_by_defect_type` 함수로 결함 유형 기반 Non-IID 분배
- **편향 강도 조절**: `bias_strength` 파라미터로 클라이언트 간 데이터 분포 편향 정도 조절 (기본값: 0.8)
  - `bias_strength`가 높을수록 각 클라이언트가 특정 결함 유형에 더 편향됨
  - 각 클라이언트는 순환 방식으로 주요 결함 유형이 할당됨
- **실제 환경 시뮬레이션**: 각 클라이언트가 특정 결함 유형에 편향된 데이터를 가지는 실제 환경 시뮬레이션
- **최소 패치 수 보장**: 모든 클라이언트가 최소한의 패치를 가지도록 자동 조정

### 4. 분산 평가 방식 (Distributed Testing)

- **프라이버시 보호**: 각 클라이언트가 자신의 테스트 데이터로만 평가하여 데이터 프라이버시 보호
- **Train/Val/Test 분할**: 각 클라이언트가 자신의 데이터를 60%/20%/20%로 분할
  - Train: 60% (학습용)
  - Val: 20% (검증용)
  - Test: 20% (테스트용)
- **가중 평균 정확도**: 서버가 클라이언트별 정확도를 데이터 크기 기반 가중 평균으로 계산
- **클라이언트별 성능 추적**: 각 클라이언트의 테스트 손실 및 정확도를 개별적으로 추적

### 5. 단계적 전이 학습 전략 (Progressive Fine-tuning)

- **초기 단계 (라운드 0~3)**: Backbone 고정
  - ResNet-18 backbone의 가중치를 고정 (`requires_grad=False`)
  - 분류 헤드(FC 레이어)만 학습하여 안정적인 초기 학습
  - ImageNet 사전 학습 가중치 보존
  
- **라운드 4부터**: Backbone 해제
  - `FREEZE_EPOCHS = 3` 설정에 따라 라운드 4에서 자동으로 backbone 해제
  - Backbone의 모든 파라미터를 학습 가능하게 설정 (`requires_grad=True`)
  - 전체 모델을 함께 fine-tuning하여 성능 향상

- **장점**:
  - 안정적인 학습: 초기에는 헤드만 학습하여 안정적인 학습 시작
  - 효율성: 초기에는 고정된 backbone으로 빠른 학습
  - 성능 향상: 이후 전체 모델을 fine-tuning하여 최종 성능 개선

### 6. 재현성 보장 및 메모리 관리

- **시드 설정**: 모든 랜덤 시드 고정 (Python, NumPy, PyTorch, CUDA)
  - `RANDOM_SEED = 42`로 설정하여 동일한 결과 재현 가능
- **GPU 메모리 관리**: 학습 시작 전 GPU 메모리 정리 및 동기화
- **가비지 컬렉션**: Python 가비지 컬렉션으로 메모리 최적화

### 7. 학습 파이프라인 개선

- **자동 모델 저장**: 학습 중 각 라운드별 모델 자동 저장 (`patch_cnn_models/round_*.pth`)
- **최종 모델 저장**: 학습 완료 후 최종 모델 자동 저장 (`saved_models/Defect_Classifier_FL_*.pth`)
- **중복 파일 방지**: 동일한 이름의 모델이 있으면 자동으로 번호 추가
- **학습 곡선 시각화**: 클라이언트별 학습 손실 및 정확도 곡선 자동 생성


## 프로젝트 구조

### 디렉토리 구조

```
IoT_FL_AM/
├── PatchBased_CNN_Learning.ipynb    # Patch-based CNN 학습 노트북
├── U_net_Federated_Learning.ipynb   # U-Net 연합 학습 노트북
├── requirements.txt                  # 필수 패키지 목록
├── README.md                         # 프로젝트 문서
├── utils/
│   ├── patch_cnn/                    # Patch-based CNN 모듈
│   │   ├── classifier.py             # ResNet-18 분류기 모델
│   │   ├── dataset_functions.py      # 패치 생성 및 데이터셋 함수
│   │   ├── federated_averaging.py    # 연합 학습 (FedAvg) 구현
│   │   └── visualization.py          # 학습 곡선 및 결과 시각화
│   ├── u_net/                        # U-Net 모듈
│   │   ├── unet.py                   # U-Net 모델 정의
│   │   ├── dataset_functions.py      # U-Net 데이터셋 함수
│   │   ├── federated_averaging.py    # U-Net 연합 학습
│   │   ├── image_processing.py       # 이미지 전처리
│   │   └── visualization.py          # U-Net 결과 시각화
│   ├── dataset/                      # 데이터 전처리 및 유틸리티
│   │   ├── convert_tif_to_jpg.py     # TIF -> JPG 변환
│   │   ├── data_augmentation.py      # 데이터 증강
│   │   ├── merge_data_files.py       # 데이터 파일 병합
│   │   ├── remove_large_images.py    # 대용량 이미지 제거
│   │   └── visualize_masks.py        # 마스크 시각화
│   └── analyze_defect.py             # 결함 데이터 분석 스크립트
├── data_train/                       # 학습 데이터
│   ├── 0/                            # Post Spreading 이미지
│   ├── 1/                            # Post Fusion 이미지
│   └── annotations/                  # 원본 마스크 파일 (.npy)
├── data_test/                        # 테스트 데이터
├── dataset/                          # 원본 데이터셋 (다중 재료)
│   ├── 17-4_PH_Strainless_Steel/
│   ├── FormUp_350_Maraging_Steel/
│   ├── GammaPrint-700/
│   └── ...
├── saved_models/                     # 저장된 모델
│   ├── Defect_Classifier_FL_*.pth    # Patch CNN 모델
│   └── FL_*.h5                       # U-Net 모델
├── patch_cnn_models/                 # 학습 중 저장되는 Patch CNN 모델
│   └── round_*.pth                   # 각 라운드별 모델
└── visualizations/                   # 시각화 결과
```