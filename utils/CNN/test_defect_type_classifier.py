"""
결함 유형 분류 모델 테스트 및 결과 분석 스크립트
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[경고] sklearn이 설치되지 않았습니다. 기본 분석만 수행합니다.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[경고] matplotlib/seaborn이 설치되지 않았습니다. 시각화를 건너뜁니다.")

# utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.defect_type_classifier import (
    DefectTypeDataset, 
    DefectTypeClassifier,
    analyze_defect_types,
    extract_defect_types_from_metadata
)


def load_model(checkpoint_path: str, device: torch.device):
    """체크포인트에서 모델 로드"""
    print(f"\n[모델 로드] {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    defect_type_mapping = checkpoint['defect_type_mapping']
    idx_to_name = checkpoint['idx_to_name']
    num_classes = len(defect_type_mapping)
    
    # 모델 생성 및 가중치 로드
    model = DefectTypeClassifier(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  - 클래스 수: {num_classes}")
    print(f"  - 검증 정확도: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  - 학습 에포크: {checkpoint.get('epoch', 'N/A')}")
    
    return model, defect_type_mapping, idx_to_name


def test_model(model, dataset, defect_type_mapping, idx_to_name, device, batch_size=32):
    """모델 테스트 및 예측"""
    print(f"\n[모델 테스트]")
    print(f"  - 테스트 샘플 수: {len(dataset)}")
    print(f"  - 배치 크기: {batch_size}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_labels = []
    all_defect_types = []
    all_paths = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, defect_types, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_defect_types.extend(defect_types)
            all_paths.extend(paths)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    
    print(f"  - 전체 정확도: {accuracy:.2f}% ({correct}/{total})")
    
    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'defect_types': all_defect_types,
        'paths': all_paths,
        'accuracy': accuracy
    }


def analyze_results(results: Dict, idx_to_name: Dict, defect_type_mapping: Dict):
    """결과 상세 분석"""
    predictions = results['predictions']
    labels = results['labels']
    defect_types = results['defect_types']
    
    print(f"\n[결과 분석]")
    print("=" * 60)
    
    # 클래스별 정확도
    print("\n[클래스별 정확도]")
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    for idx in sorted(class_total.keys()):
        class_name = idx_to_name[idx]
        class_acc = 100 * class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0
        print(f"  - {class_name:30s}: {class_acc:6.2f}% ({class_correct[idx]}/{class_total[idx]})")
    
    # 혼동 행렬
    print("\n[혼동 행렬 생성 중...]")
    if HAS_SKLEARN:
        cm = confusion_matrix(labels, predictions)
        
        # 분류 리포트
        print("\n[분류 리포트]")
        target_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
        report = classification_report(
            labels, predictions, 
            target_names=target_names,
            digits=4
        )
        print(report)
    else:
        # sklearn 없이 직접 계산
        num_classes = len(idx_to_name)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true_label, pred_label in zip(labels, predictions):
            cm[true_label][pred_label] += 1
        
        target_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
        
        print("\n[혼동 행렬 (수치)]")
        print("실제\\예측", end="")
        for name in target_names:
            print(f"\t{name[:10]}", end="")
        print()
        for i, true_name in enumerate(target_names):
            print(f"{true_name[:10]}", end="")
            for j in range(num_classes):
                print(f"\t{cm[i][j]}", end="")
            print()
    
    # 주요 오분류 분석
    print("\n[주요 오분류 분석] (상위 10개)")
    error_analysis = defaultdict(int)
    
    for pred, label, defect_type in zip(predictions, labels, defect_types):
        if pred != label:
            pred_name = idx_to_name[pred]
            true_name = idx_to_name[label]
            error_analysis[f"{true_name} -> {pred_name}"] += 1
    
    sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
    for error_pair, count in sorted_errors[:10]:
        print(f"  - {error_pair:50s}: {count}개")
    
    return cm, target_names


def visualize_results(cm, target_names, save_path="test_results_confusion_matrix.png"):
    """결과 시각화"""
    if not HAS_MATPLOTLIB:
        print(f"\n[시각화 건너뜀] matplotlib/seaborn이 설치되지 않았습니다.")
        return
    
    print(f"\n[결과 시각화] {save_path}")
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(14, 12))
    
    # 정규화된 혼동 행렬
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    if HAS_SKLEARN:
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': '정규화된 빈도'}
        )
    else:
        # seaborn 없이 matplotlib만 사용
        plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
        plt.colorbar(label='정규화된 빈도')
        plt.xticks(range(len(target_names)), target_names, rotation=45, ha='right')
        plt.yticks(range(len(target_names)), target_names)
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                plt.text(j, i, f'{cm_normalized[i, j]:.2f}', 
                        ha='center', va='center', color='black' if cm_normalized[i, j] < 0.5 else 'white')
    
    plt.title('혼동 행렬 (Confusion Matrix)', fontsize=16, pad=20)
    plt.xlabel('예측 레이블', fontsize=12)
    plt.ylabel('실제 레이블', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  - 저장 완료: {save_path}")
    plt.close()


def test_classifier(checkpoint_path=None, data_dir="data/labeled_layers", 
                   batch_size=32, min_defect_count=5):
    """결함 유형 분류 모델 테스트"""
    print("=" * 60)
    print("결함 유형 분류 모델 테스트 및 결과 분석")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # 체크포인트 경로 확인
    if checkpoint_path is None:
        checkpoint_dir = "checkpoints"
        checkpoint_path = os.path.join(checkpoint_dir, "defect_type_classifier_best.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"\n[오류] 체크포인트 파일이 없습니다: {checkpoint_path}")
            print("  먼저 모델을 학습하세요.")
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"\n[오류] 체크포인트 파일이 없습니다: {checkpoint_path}")
        return
    
    # 모델 로드
    model, defect_type_mapping, idx_to_name = load_model(checkpoint_path, device)
    
    # 결함 유형 분석 (테스트 데이터셋과 동일한 매핑 사용)
    print(f"\n[데이터 준비]")
    print(f"  - 데이터 디렉토리: {data_dir}")
    
    # 데이터셋 생성
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 전체 데이터셋 로드
    full_dataset = DefectTypeDataset(data_dir, defect_type_mapping, transform=transform, verbose=False)
    
    if len(full_dataset) == 0:
        print("\n[오류] 데이터셋이 비어있습니다.")
        return
    
    # 학습/검증 분할 (학습 시와 동일한 시드 사용)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  - 전체 데이터: {len(full_dataset)}개")
    print(f"  - 테스트 데이터: {len(test_dataset)}개")
    
    # 모델 테스트
    results = test_model(model, test_dataset, defect_type_mapping, idx_to_name, device, batch_size)
    
    # 결과 분석
    cm, target_names = analyze_results(results, idx_to_name, defect_type_mapping)
    
    # 시각화
    visualize_results(cm, target_names, "test_results_confusion_matrix.png")
    
    # 샘플 예측 결과
    print(f"\n[샘플 예측 결과] (10개)")
    print("=" * 60)
    for i in range(min(10, len(results['predictions']))):
        pred_idx = results['predictions'][i]
        true_idx = results['labels'][i]
        pred_name = idx_to_name[pred_idx]
        true_name = idx_to_name[true_idx]
        is_correct = "✓" if pred_idx == true_idx else "✗"
        
        print(f"\n  샘플 {i+1}: {Path(results['paths'][i]).name}")
        print(f"    - 실제: {true_name}")
        correct_mark = "정확" if pred_idx == true_idx else "오류"
        print(f"    - 예측: {pred_name} ({correct_mark})")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    
    return results, cm, target_names


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='결함 유형 분류 모델 테스트')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='체크포인트 파일 경로 (기본값: checkpoints/defect_type_classifier_best.pth)')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='테스트 데이터 디렉토리')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--min-count', type=int, default=5,
                       help='최소 샘플 수 (매핑 생성용)')
    
    args = parser.parse_args()
    
    test_classifier(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        min_defect_count=args.min_count
    )


if __name__ == "__main__":
    main()

