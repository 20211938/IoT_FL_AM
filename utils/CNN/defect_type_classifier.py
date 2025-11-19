"""
결함 유형 분류 모델
실제 데이터의 결함 유형을 학습하여 분류하는 모델
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# LabeledImageDataset은 여기서 직접 사용하지 않음


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
    
    if 'ScanningImageModel' in metadata:
        tag_boxes = metadata['ScanningImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    return defect_types if defect_types else ["Normal"]


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
        self.labels = []  # 클래스 인덱스
        self.defect_types_list = []  # 실제 결함 유형 이름
        
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
                    
                    # 결함 유형 추출
                    defect_types = extract_defect_types_from_metadata(metadata)
                    
                    # 첫 번째 결함 유형을 레이블로 사용 (여러 개면 첫 번째)
                    primary_defect = defect_types[0] if defect_types else "Normal"
                    
                    # 매핑에 없으면 "Normal"로 처리
                    if primary_defect not in defect_type_mapping:
                        primary_defect = "Normal"
                    
                    label_idx = defect_type_mapping[primary_defect]
                    
                    self.image_paths.append(img_file)
                    self.labels.append(label_idx)
                    self.defect_types_list.append(primary_defect)
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
        label = self.labels[idx]
        defect_type = self.defect_types_list[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, defect_type, str(self.image_paths[idx])


class DefectTypeClassifier(nn.Module):
    """결함 유형 분류 모델 (CNN 기반)"""
    
    def __init__(self, num_classes: int):
        super(DefectTypeClassifier, self).__init__()
        
        # 특징 추출기
        self.features = nn.Sequential(
            # 첫 번째 블록
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64
            
            # 두 번째 블록
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # 세 번째 블록
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # 네 번째 블록
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
        )
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
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


def train_defect_classifier(data_dir: str, epochs: int = 20, batch_size: int = 32,
                           min_defect_count: int = 10, checkpoint_dir: str = "checkpoints"):
    """결함 유형 분류 모델 학습"""
    
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
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = DefectTypeDataset(data_dir, defect_type_mapping, transform=transform)
    
    if len(dataset) == 0:
        print("\n[오류] 데이터셋이 비어있습니다.")
        return
    
    # 데이터 분할 (80% 학습, 20% 검증)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # GPU 사용률 최적화를 위한 DataLoader 설정
    # num_workers: 병렬 데이터 로딩 (CPU 코어 수에 맞게 조정, Windows에서는 0-4 권장)
    # pin_memory: GPU 전송 최적화 (CUDA 사용 시 True 권장)
    # persistent_workers: 워커 재사용으로 오버헤드 감소 (num_workers > 0일 때만)
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    print(f"\n[데이터 분할]")
    print(f"  - 학습: {len(train_dataset)}개")
    print(f"  - 검증: {len(val_dataset)}개")
    
    # 모델 생성
    model = DefectTypeClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 학습
    print(f"\n[학습 시작]")
    print(f"  - 에포크: {epochs}")
    print(f"  - 배치 크기: {batch_size}")
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, _, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
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
        
        scheduler.step()
    
    print(f"\n[학습 완료]")
    print(f"  - 최고 검증 정확도: {best_val_acc:.2f}%")
    
    return model, defect_type_mapping, idx_to_name, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='결함 유형 분류 모델 학습')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='데이터 디렉토리')
    parser.add_argument('--epochs', type=int, default=20,
                       help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--min-count', type=int, default=10,
                       help='최소 샘플 수 (이보다 적으면 제외)')
    
    args = parser.parse_args()
    
    train_defect_classifier(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        min_defect_count=args.min_count
    )

