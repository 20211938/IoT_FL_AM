"""
결함 유형 분류 모델
실제 데이터의 결함 유형을 학습하여 분류하는 모델
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from contextlib import contextmanager
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# LabeledImageDataset은 여기서 직접 사용하지 않음


class Tee:
    """콘솔과 파일에 동시에 출력하는 클래스"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


@contextmanager
def log_to_file(log_file_path: str):
    """콘솔 출력을 파일에도 저장하는 컨텍스트 매니저"""
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    log_file = open(log_file_path, 'w', encoding='utf-8')
    original_stdout = sys.stdout
    
    try:
        # stdout을 파일과 콘솔에 동시에 출력하도록 설정
        sys.stdout = Tee(original_stdout, log_file)
        yield log_file
    finally:
        sys.stdout = original_stdout
        log_file.close()


def extract_defect_types_from_metadata(metadata: dict) -> List[str]:
    """JSON 메타데이터에서 결함 유형 추출"""
    defect_types = []
    
    # TagBoxes에서 결함 정보 추출
    if 'DepositionImageModel' in metadata:
        tag_boxes = metadata['DepositionImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                # D1, D2 등은 제거하고 실제 결함 유형만 추출
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    # ScanningImageModel은 현재 데이터셋에 결함이 없으므로 제외
    # 향후 ScanningImageModel에 결함이 추가되면 아래 주석을 해제
    # if 'ScanningImageModel' in metadata:
    #     tag_boxes = metadata['ScanningImageModel'].get('TagBoxes', [])
    #     for tag in tag_boxes:
    #         name = tag.get('Name', '').strip()
    #         comment = tag.get('Comment', '').strip()
    #         if name:
    #             defect_type = comment if comment else name
    #             if defect_type and defect_type not in defect_types:
    #                 defect_types.append(defect_type)
    
    return defect_types if defect_types else ["Normal"]


class TransformWrapper(Dataset):
    """데이터셋에 다른 transform을 적용하는 Wrapper"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label, defect_types, path = self.dataset[idx]  # defect_types는 리스트
        if self.transform:
            image = self.transform(image)
        return image, label, defect_types, path


class DefectTypeDataset(Dataset):
    """결함 유형 분류용 데이터셋"""
    
    def __init__(self, data_dir: str, defect_type_mapping: Dict[str, int], 
                 transform=None, verbose=True):
        """
        Args:
            data_dir: 데이터 디렉토리
            defect_type_mapping: 결함 유형 -> 클래스 인덱스 매핑
            transform: 이미지 변환
            verbose: 진행 상황 출력
        """
        self.data_dir = Path(data_dir)
        self.defect_type_mapping = defect_type_mapping
        self.transform = transform
        self.verbose = verbose
        self.image_paths = []
        self.labels = []  # 다중 레이블 (멀티-핫 벡터)
        self.defect_types_list = []  # 실제 결함 유형 이름 리스트
        
        if self.verbose:
            print(f"\n[데이터 로딩] 디렉토리 검색 중: {data_dir}")
        
        db_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if self.verbose:
            print(f"[데이터 로딩] 발견된 데이터베이스 디렉토리: {len(db_dirs)}개")
        
        for idx, db_dir in enumerate(db_dirs, 1):
            if self.verbose and idx % 10 == 0:
                print(f"[데이터 로딩] 처리 중: {idx}/{len(db_dirs)} 디렉토리...")
            
            img_count = 0
            for img_file in db_dir.glob("*.jpg"):
                json_file = img_file.with_suffix(".jpg.json")
                if not json_file.exists():
                    continue
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # 결함 유형 추출 (DepositionImageModel에서만 추출)
                    defect_types = extract_defect_types_from_metadata(metadata)
                    
                    # 다중 레이블 생성 (멀티-핫 인코딩)
                    # 모든 결함 유형을 포함하는 벡터 생성
                    num_classes = len(defect_type_mapping)
                    label_vector = torch.zeros(num_classes, dtype=torch.float32)
                    
                    # 발견된 모든 결함 유형에 대해 레이블 설정
                    has_valid_defect = False
                    for defect_type in defect_types:
                        if defect_type in defect_type_mapping:
                            label_idx = defect_type_mapping[defect_type]
                            label_vector[label_idx] = 1.0
                            has_valid_defect = True
                    
                    # 결함이 없거나 매핑에 없는 경우 "Normal"로 설정
                    if not has_valid_defect and "Normal" in defect_type_mapping:
                        normal_idx = defect_type_mapping["Normal"]
                        label_vector[normal_idx] = 1.0
                        defect_types = ["Normal"]
                    
                    self.image_paths.append(img_file)
                    self.labels.append(label_vector)
                    self.defect_types_list.append(defect_types)  # 리스트로 저장
                    img_count += 1
                    
                except Exception as e:
                    if self.verbose:
                        print(f"[경고] 파일 읽기 실패: {json_file} - {e}")
            
            if self.verbose and img_count > 0:
                print(f"  - {db_dir.name}: {img_count}개 이미지 발견")
        
        if self.verbose:
            print(f"[데이터 로딩] 완료! 총 {len(self.image_paths)}개 이미지 로드됨")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]  # 이미 torch tensor (멀티-핫 벡터)
        defect_types = self.defect_types_list[idx]  # 리스트
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, defect_types, str(self.image_paths[idx])


class DefectTypeClassifier(nn.Module):
    """결함 유형 분류 모델 (ResNet 기반 Transfer Learning)"""
    
    def __init__(self, num_classes: int, model_name: str = "resnet34", pretrained: bool = True):
        super(DefectTypeClassifier, self).__init__()
        
        # ResNet 백본 선택
        if model_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif model_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif model_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from ['resnet18', 'resnet34', 'resnet50']")
        
        # 특징 추출기 (마지막 FC 레이어 제외)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # 분류기 (Transfer Learning을 위한 커스텀 헤드)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 특징 추출기 가중치 고정 여부 (선택적)
        self.freeze_backbone = False
    
    def freeze_features(self):
        """백본 가중치 고정 (Transfer Learning 초기 단계용)"""
        for param in self.features.parameters():
            param.requires_grad = False
        self.freeze_backbone = True
    
    def unfreeze_features(self):
        """백본 가중치 해제 (Fine-tuning용)"""
        for param in self.features.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def analyze_defect_types(data_dir: str, min_count: int = 10) -> Dict[str, int]:
    """
    데이터셋에서 결함 유형을 분석하고 매핑 생성
    
    Args:
        data_dir: 데이터 디렉토리
        min_count: 최소 샘플 수 (이보다 적으면 제외)
    
    Returns:
        결함 유형 -> 클래스 인덱스 매핑
    """
    print("\n[결함 유형 분석]")
    print(f"  - 데이터 디렉토리: {data_dir}")
    print(f"  - 최소 샘플 수: {min_count}")
    
    defect_type_counts = Counter()
    data_path = Path(data_dir)
    
    db_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"  - 디렉토리 수: {len(db_dirs)}개")
    
    for db_dir in db_dirs:
        for img_file in db_dir.glob("*.jpg"):
            json_file = img_file.with_suffix(".jpg.json")
            if not json_file.exists():
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                defect_types = extract_defect_types_from_metadata(metadata)
                for defect_type in defect_types:
                    defect_type_counts[defect_type] += 1
            except:
                continue
    
    # 최소 샘플 수 이상인 결함 유형만 선택
    filtered_types = {k: v for k, v in defect_type_counts.items() 
                     if v >= min_count}
    
    # 정렬 (빈도순)
    sorted_types = sorted(filtered_types.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n[발견된 결함 유형] (최소 {min_count}개 이상)")
    for defect_type, count in sorted_types:
        print(f"  - {defect_type}: {count}개")
    
    # 매핑 생성 (Normal은 항상 포함)
    mapping = {"Normal": 0}
    for idx, (defect_type, _) in enumerate(sorted_types, 1):
        if defect_type != "Normal":
            mapping[defect_type] = idx
    
    print(f"\n[클래스 매핑] 총 {len(mapping)}개 클래스")
    for defect_type, idx in mapping.items():
        print(f"  - 클래스 {idx}: {defect_type}")
    
    return mapping


def train_defect_classifier(data_dir: str, epochs: int = 300, batch_size: int = 64,
                           min_defect_count: int = 10, checkpoint_dir: str = "checkpoints",
                           learning_rate: float = 0.0001, weight_decay: float = 1e-4,
                           scheduler_type: str = "cosine", use_data_augmentation: bool = True,
                           label_smoothing: float = 0.1, image_size: int = 224,
                           early_stopping_threshold: float = 98.0, log_dir: Optional[str] = None,
                           model_name: str = "resnet34", pretrained: bool = True):
    """결함 유형 분류 모델 학습
    
    Args:
        log_dir: 로그 파일을 저장할 디렉토리 (None이면 로그 저장 안 함)
    """
    
    # 로그 파일 설정
    log_file_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # 로그 파일이 설정된 경우 컨텍스트 매니저 사용
    if log_file_path:
        log_context = log_to_file(log_file_path)
        log_context.__enter__()
        print(f"[로그 저장] 로그 파일: {log_file_path}")
        print("=" * 80)
        print(f"학습 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    else:
        log_context = None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n[Device] {device}")
        
        # 결함 유형 분석
        defect_type_mapping = analyze_defect_types(data_dir, min_count=min_defect_count)
        num_classes = len(defect_type_mapping)
        
        if num_classes < 2:
            print("\n[오류] 분류할 결함 유형이 부족합니다.")
            return
    
        # 역매핑 생성 (인덱스 -> 이름)
        idx_to_name = {idx: name for name, idx in defect_type_mapping.items()}
        
        # 데이터셋 생성
        # 강화된 데이터 증강 적용 (학습 시)
        if use_data_augmentation:
            train_transform = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # 약간 큰 크기로 리사이즈
                transforms.RandomCrop(image_size),  # 랜덤 크롭
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=30),  # 회전 각도 증가
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 이동 및 스케일
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # 색상 조정 강화
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 원근 변환
                transforms.ToTensor(),  # PIL Image를 Tensor로 변환 (RandomErasing 전에 필요)
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),  # 랜덤 지우기 (Tensor에만 적용 가능)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # 검증용 변환 (증강 없음)
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 원본 데이터셋 생성 (transform 없이)
        base_dataset = DefectTypeDataset(data_dir, defect_type_mapping, transform=None, verbose=True)
        
        if len(base_dataset) == 0:
            print("\n[오류] 데이터셋이 비어있습니다.")
            return
        
        # 데이터 분할 (70% 학습, 15% 검증, 15% 테스트)
        total_size = len(base_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size  # 나머지가 테스트
        
        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            range(len(base_dataset)), [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 각각 다른 transform 적용
        train_dataset = TransformWrapper(
            torch.utils.data.Subset(base_dataset, train_indices.indices),
            transform=train_transform
        )
        val_dataset = TransformWrapper(
            torch.utils.data.Subset(base_dataset, val_indices.indices),
            transform=val_transform
        )
        # 테스트 데이터셋 추가 (검증과 동일한 transform 사용)
        test_dataset = TransformWrapper(
            torch.utils.data.Subset(base_dataset, test_indices.indices),
            transform=val_transform  # 검증과 동일하게 증강 없음
        )
        
        # GPU 사용률 최적화를 위한 DataLoader 설정
        # num_workers: 병렬 데이터 로딩 (GPU 사용 시 더 많은 워커로 증가)
        # pin_memory: GPU 전송 최적화 (CUDA 사용 시 True 권장)
        # persistent_workers: 워커 재사용으로 오버헤드 감소
        # prefetch_factor: 미리 로딩할 배치 수 (GPU 대기 시간 감소)
        if torch.cuda.is_available():
            # CPU 코어 수에 맞춰 워커 수 조정 (최대 16)
            cpu_count = os.cpu_count() or 8
            num_workers = min(16, max(8, cpu_count // 2))  # CPU 코어의 절반, 최소 8, 최대 16
            prefetch_factor = 8  # 미리 8개 배치 로딩 (GPU 대기 시간 최소화)
        else:
            num_workers = 0
            prefetch_factor = 2
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        # 테스트 DataLoader 추가
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
        print(f"\n[데이터 분할]")
        print(f"  - 학습: {len(train_dataset)}개 (70%)")
        print(f"  - 검증: {len(val_dataset)}개 (15%)")
        print(f"  - 테스트: {len(test_dataset)}개 (15%)")
        
        # 모델 생성 (ResNet 기반)
        model = DefectTypeClassifier(num_classes, model_name=model_name, pretrained=pretrained).to(device)
        
        # Transfer Learning: 초기에는 백본 고정 (선택적)
        # model.freeze_features()  # 필요시 주석 해제
        
        # 손실 함수 (다중 레이블 분류용 BCEWithLogitsLoss)
        # BCEWithLogitsLoss는 sigmoid + BCE를 결합한 것으로 다중 레이블에 적합
        criterion = nn.BCEWithLogitsLoss()
        print(f"[모델 설정] 다중 레이블 분류 모드 (BCEWithLogitsLoss 사용)")
        if label_smoothing > 0:
            print(f"[참고] Label Smoothing은 다중 레이블에서 자동으로 적용되지 않습니다.")
        
        # 옵티마이저 (Weight Decay 포함)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 학습률 스케줄러
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            print(f"[최적화] CosineAnnealingLR 스케줄러 사용 (T_max={epochs}, eta_min=1e-6)")
        elif scheduler_type == "cosine_warmup":
            # CosineAnnealingWarmRestarts 사용 (더 부드러운 학습률 감소)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            print(f"[최적화] CosineAnnealingWarmRestarts 스케줄러 사용 (T_0=10, T_mult=2)")
        elif scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
            print(f"[최적화] ReduceLROnPlateau 스케줄러 사용 (factor=0.5, patience=5)")
        else:  # "step" (기본값)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            print(f"[최적화] StepLR 스케줄러 사용 (step_size=10, gamma=0.5)")
        
        print(f"[하이퍼파라미터]")
        print(f"  - 모델: {model_name} ({'Pretrained' if pretrained else 'Random Init'})")
        print(f"  - 학습률: {learning_rate}")
        print(f"  - Weight Decay: {weight_decay}")
        print(f"  - 이미지 크기: {image_size}x{image_size}")
        print(f"  - 데이터 증강: {'활성화 (강화됨)' if use_data_augmentation else '비활성화'}")
        if label_smoothing > 0:
            print(f"  - Label Smoothing: {label_smoothing}")
        
        # Mixed Precision Training (AMP) - GPU 사용률 및 속도 향상
        use_amp = torch.cuda.is_available()
        if use_amp:
            try:
                # PyTorch 2.0+ 최신 방식
                scaler = torch.amp.GradScaler('cuda')
                print("[최적화] Mixed Precision Training (AMP) 활성화")
            except AttributeError:
                # 구버전 호환성
                scaler = torch.cuda.amp.GradScaler()
                print("[최적화] Mixed Precision Training (AMP) 활성화 (구버전)")
        else:
            scaler = None
        
        # CUDA 스트림 최적화 (GPU 사용률 안정화)
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()
            print(f"[최적화] CUDA 스트림 활성화, num_workers={num_workers}, prefetch_factor={prefetch_factor}")
        else:
            stream = None
        
        # 학습
        print(f"\n[학습 시작]")
        print(f"  - 에포크: {epochs}")
        print(f"  - 배치 크기: {batch_size}")
        print(f"  - 조기 종료 기준: 검증 정확도 {early_stopping_threshold}% 도달")
        
        # 학습 시간 측정 시작
        start_time = time.time()
        
        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
                   'test_loss': [], 'test_acc': []}
        
        for epoch in range(epochs):
            # 학습
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                # CUDA 스트림을 사용한 비동기 데이터 전송
                if stream is not None:
                    with torch.cuda.stream(stream):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    torch.cuda.current_stream().wait_stream(stream)
                else:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed Precision Training 사용
                if use_amp:
                    try:
                        # PyTorch 2.0+ 최신 방식
                        with torch.amp.autocast('cuda'):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    except AttributeError:
                        # 구버전 호환성
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
                # 다중 레이블 정확도 계산 (threshold 기반)
                # sigmoid를 적용하여 확률로 변환
                probs = torch.sigmoid(outputs.data)
                predicted = (probs > 0.5).float()  # threshold = 0.5
                
                # 각 샘플에 대해 예측된 레이블과 실제 레이블이 일치하는지 확인
                # 다중 레이블에서는 모든 레이블이 정확히 일치해야 정확한 것으로 간주
                correct = (predicted == labels).all(dim=1).sum().item()
                train_total += labels.size(0)
                train_correct += correct
            
            train_acc = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 검증
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    # CUDA 스트림을 사용한 비동기 데이터 전송
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            images = images.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                        torch.cuda.current_stream().wait_stream(stream)
                    else:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    
                    # Mixed Precision Training 사용
                    if use_amp:
                        try:
                            # PyTorch 2.0+ 최신 방식
                            with torch.amp.autocast('cuda'):
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                        except AttributeError:
                            # 구버전 호환성
                            with torch.cuda.amp.autocast():
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    
                    # 다중 레이블 정확도 계산 (threshold 기반)
                    probs = torch.sigmoid(outputs.data)
                    predicted = (probs > 0.5).float()  # threshold = 0.5
                    
                    # 각 샘플에 대해 예측된 레이블과 실제 레이블이 일치하는지 확인
                    correct = (predicted == labels).all(dim=1).sum().item()
                    val_total += labels.size(0)
                    val_correct += correct
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\n[Epoch {epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% (다중 레이블 정확도)")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}% (다중 레이블 정확도)")
            
            # 최고 성능 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, "defect_type_classifier_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'defect_type_mapping': defect_type_mapping,
                    'idx_to_name': idx_to_name,
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'history': history
                }, checkpoint_path)
                print(f"  -> 최고 모델 저장: {checkpoint_path} (Val Acc: {val_acc:.2f}%)")
            
            # 학습률 스케줄러 업데이트
            if scheduler_type == "plateau":
                scheduler.step(val_acc)  # ReduceLROnPlateau는 metric 필요
            else:
                scheduler.step()
            
            # 조기 종료: 목표 정확도 도달 시 학습 중단
            if val_acc >= early_stopping_threshold:
                print(f"\n[조기 종료] 목표 정확도 {early_stopping_threshold}% 도달!")
                print(f"  - 현재 검증 정확도: {val_acc:.2f}%")
                print(f"  - 학습 중단 (Epoch {epoch+1}/{epochs})")
                break
        
        # 학습 완료 후 테스트 데이터로 최종 평가
        print(f"\n[최종 테스트 평가]")
        print("=" * 80)
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels, _, _ in tqdm(test_loader, desc="[Test] 최종 평가"):
                # CUDA 스트림을 사용한 비동기 데이터 전송
                if stream is not None:
                    with torch.cuda.stream(stream):
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                    torch.cuda.current_stream().wait_stream(stream)
                else:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                
                # Mixed Precision Training 사용
                if use_amp:
                    try:
                        with torch.amp.autocast('cuda'):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                    except AttributeError:
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                
                # 다중 레이블 정확도 계산 (threshold 기반)
                probs = torch.sigmoid(outputs.data)
                predicted = (probs > 0.5).float()  # threshold = 0.5
                
                # 각 샘플에 대해 예측된 레이블과 실제 레이블이 일치하는지 확인
                correct = (predicted == labels).all(dim=1).sum().item()
                test_total += labels.size(0)
                test_correct += correct
        
        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"  - 테스트 손실: {avg_test_loss:.4f}")
        print(f"  - 테스트 정확도: {test_acc:.2f}% (다중 레이블 정확도)")
        print("=" * 80)
        
        # 학습 시간 측정 종료
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n[학습 완료]")
        print(f"  - 최고 검증 정확도: {best_val_acc:.2f}%")
        print(f"  - 최종 테스트 정확도: {test_acc:.2f}%")
        print(f"  - 최종 에포크: {epoch+1}/{epochs}")
        print(f"  - 총 학습 시간: {hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed_time:.2f}초)")
        if epoch > 0:
            avg_epoch_time = elapsed_time / (epoch + 1)
            print(f"  - 에포크당 평균 시간: {avg_epoch_time:.2f}초")
        
        if log_file_path:
            print("=" * 80)
            print(f"학습 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"로그 파일 저장 완료: {log_file_path}")
            print("=" * 80)
    
    finally:
        if log_context is not None:
            log_context.__exit__(None, None, None)
    
    return model, defect_type_mapping, idx_to_name, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='결함 유형 분류 모델 학습')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=300,
                       help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기 (기본값: 32, ResNet 사용 시 권장)')
    parser.add_argument('--min-count', type=int, default=10,
                       help='최소 샘플 수 (이보다 적으면 제외)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='학습률 (기본값: 0.0001, ResNet Transfer Learning 권장)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight Decay (L2 정규화, 기본값: 1e-4)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'cosine_warmup', 'plateau'],
                       help='학습률 스케줄러 타입: step, cosine, cosine_warmup, plateau (기본값: cosine)')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='데이터 증강 비활성화')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label Smoothing 값 (0.0-0.3 권장, 기본값: 0.1)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='이미지 크기 (기본값: 224, ResNet 표준 입력 크기)')
    parser.add_argument('--early-stopping-threshold', type=float, default=98.0,
                       help='조기 종료 기준 정확도 (기본값: 98.0%%)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='로그 파일을 저장할 디렉토리 (기본값: logs, None이면 로그 저장 안 함)')
    parser.add_argument('--model-name', type=str, default='resnet34',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='모델 아키텍처 선택 (기본값: resnet34)')
    parser.add_argument('--no-pretrained', action='store_true',
                       help='사전 학습된 가중치 사용 안 함 (랜덤 초기화)')
    
    args = parser.parse_args()
    
    # log_dir이 "None" 문자열이면 None으로 변환
    log_dir = None if args.log_dir.lower() == 'none' else args.log_dir
    
    train_defect_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_defect_count=args.min_count,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        use_data_augmentation=not args.no_augmentation,
        label_smoothing=args.label_smoothing,
        image_size=args.image_size,
        early_stopping_threshold=args.early_stopping_threshold,
        log_dir=log_dir,
        model_name=args.model_name,
        pretrained=not args.no_pretrained
    )

