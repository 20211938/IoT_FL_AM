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
from utils.CNN.defect_type_classifier import (
    DefectTypeDataset, 
    DefectTypeClassifier,
    analyze_defect_types,
    extract_defect_types_from_metadata,
    custom_collate_fn
)


def load_model(checkpoint_path: str, device: torch.device):
    """체크포인트에서 모델 로드"""
    print(f"\n[모델 로드] {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    defect_type_mapping = checkpoint['defect_type_mapping']
    idx_to_name = checkpoint['idx_to_name']
    num_classes = len(defect_type_mapping)
    
    # 체크포인트에서 모델 정보 가져오기 (없으면 기본값 사용)
    model_name = checkpoint.get('model_name', 'resnet34')
    pretrained = checkpoint.get('pretrained', True)
    
    # 모델 생성 및 가중치 로드
    model = DefectTypeClassifier(num_classes, model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  - 클래스 수: {num_classes}")
    print(f"  - 모델 아키텍처: {model_name}")
    print(f"  - 검증 정확도: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"  - 학습 에포크: {checkpoint.get('epoch', 'N/A')}")
    
    return model, defect_type_mapping, idx_to_name


def test_model(model, dataset, defect_type_mapping, idx_to_name, device, batch_size=32):
    """모델 테스트 및 예측 (다중 레이블 분류)"""
    print(f"\n[모델 테스트]")
    print(f"  - 테스트 샘플 수: {len(dataset)}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 다중 레이블 분류 모드 (Threshold: 0.5)")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    all_predictions = []
    all_labels = []
    all_defect_types = []
    all_paths = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, defect_types_list, paths in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # 다중 레이블 분류: sigmoid + threshold
            probs = torch.sigmoid(outputs.data)
            predicted = (probs > 0.5).float()  # threshold = 0.5
            
            # 각 샘플에 대해 예측된 레이블과 실제 레이블이 일치하는지 확인
            # 다중 레이블에서는 모든 레이블이 정확히 일치해야 정확한 것으로 간주
            batch_correct = (predicted == labels).all(dim=1).sum().item()
            
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_defect_types.extend(defect_types_list)
            all_paths.extend(paths)
            
            total += labels.size(0)
            correct += batch_correct
    
    accuracy = 100 * correct / total
    
    print(f"  - 전체 정확도: {accuracy:.2f}% ({correct}/{total}) (다중 레이블 정확도)")
    
    return {
        'predictions': np.vstack(all_predictions) if all_predictions else np.array([]),
        'labels': np.vstack(all_labels) if all_labels else np.array([]),
        'defect_types': all_defect_types,
        'paths': all_paths,
        'accuracy': accuracy
    }


def analyze_results(results: Dict, idx_to_name: Dict, defect_type_mapping: Dict):
    """결과 상세 분석 (다중 레이블 분류)"""
    predictions = results['predictions']  # (N, num_classes) 형태의 멀티-핫 벡터
    labels = results['labels']  # (N, num_classes) 형태의 멀티-핫 벡터
    defect_types = results['defect_types']
    
    print(f"\n[결과 분석] (다중 레이블 분류)")
    print("=" * 60)
    
    # 클래스별 정확도 (각 클래스가 정확히 예측된 비율)
    print("\n[클래스별 정확도]")
    num_classes = len(idx_to_name)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred_vec, label_vec in zip(predictions, labels):
        for class_idx in range(num_classes):
            if label_vec[class_idx] == 1.0:  # 실제 레이블에 해당 클래스가 있는 경우
                class_total[class_idx] += 1
                if pred_vec[class_idx] == label_vec[class_idx]:  # 예측이 맞은 경우
                    class_correct[class_idx] += 1
    
    for idx in sorted(idx_to_name.keys()):
        class_name = idx_to_name[idx]
        class_acc = 100 * class_correct[idx] / class_total[idx] if class_total[idx] > 0 else 0
        print(f"  - {class_name:30s}: {class_acc:6.2f}% ({class_correct[idx]}/{class_total[idx]})")
    
    # 다중 레이블 분류에서는 혼동 행렬이 복잡하므로 간단한 통계만 제공
    print("\n[클래스별 예측 통계]")
    num_classes = len(idx_to_name)
    target_names = [idx_to_name[i] for i in sorted(idx_to_name.keys())]
    
    # 각 클래스별로 True Positive, False Positive, False Negative 계산
    print("\n클래스별 상세 통계:")
    for class_idx in range(num_classes):
        class_name = target_names[class_idx]
        tp = fp = fn = tn = 0
        
        for pred_vec, label_vec in zip(predictions, labels):
            pred_positive = pred_vec[class_idx] == 1.0
            label_positive = label_vec[class_idx] == 1.0
            
            if pred_positive and label_positive:
                tp += 1
            elif pred_positive and not label_positive:
                fp += 1
            elif not pred_positive and label_positive:
                fn += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {class_name:30s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
    
    # 간단한 혼동 행렬 (각 클래스별 예측/실제 일치 여부)
    cm = np.zeros((num_classes, 2, 2), dtype=int)  # 각 클래스별로 2x2 행렬
    for class_idx in range(num_classes):
        for pred_vec, label_vec in zip(predictions, labels):
            pred_positive = pred_vec[class_idx] == 1.0
            label_positive = label_vec[class_idx] == 1.0
            
            if pred_positive and label_positive:
                cm[class_idx][0][0] += 1  # TP
            elif pred_positive and not label_positive:
                cm[class_idx][0][1] += 1  # FP
            elif not pred_positive and label_positive:
                cm[class_idx][1][0] += 1  # FN
            else:
                cm[class_idx][1][1] += 1  # TN
    
    return cm, target_names


def visualize_results(cm, target_names, save_path="test_results_confusion_matrix.png"):
    """결과 시각화 (다중 레이블 분류용 - 각 클래스별 Precision/Recall)"""
    if not HAS_MATPLOTLIB:
        print(f"\n[시각화 건너뜀] matplotlib/seaborn이 설치되지 않았습니다.")
        return
    
    print(f"\n[결과 시각화] {save_path}")
    
    # 각 클래스별 Precision/Recall을 막대 그래프로 시각화
    num_classes = len(target_names)
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_idx in range(num_classes):
        tp = cm[class_idx][0][0]
        fp = cm[class_idx][0][1]
        fn = cm[class_idx][1][0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # 막대 그래프
    x = np.arange(num_classes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('클래스', fontsize=12)
    ax.set_ylabel('점수', fontsize=12)
    ax.set_title('클래스별 Precision, Recall, F1-Score (다중 레이블 분류)', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
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
    
    # 데이터셋 생성 (학습 시와 동일한 transform 사용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 학습 시와 동일한 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])
    
    # 전체 데이터셋 로드
    full_dataset = DefectTypeDataset(data_dir, defect_type_mapping, transform=transform, verbose=False)
    
    if len(full_dataset) == 0:
        print("\n[오류] 데이터셋이 비어있습니다.")
        return
    
    # 데이터 분할 (학습 시와 동일한 70:15:15 분할)
    # 학습 시와 동일한 시드 사용하여 테스트 데이터셋만 추출
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 테스트 데이터셋만 사용
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices.indices)
    
    print(f"  - 전체 데이터: {len(full_dataset)}개")
    print(f"  - 테스트 데이터: {len(test_dataset)}개 (15%)")
    
    # 모델 테스트
    results = test_model(model, test_dataset, defect_type_mapping, idx_to_name, device, batch_size)
    
    # 결과 분석
    cm, target_names = analyze_results(results, idx_to_name, defect_type_mapping)
    
    # 시각화
    visualize_results(cm, target_names, "test_results_confusion_matrix.png")
    
    # 샘플 예측 결과 (다중 레이블)
    print(f"\n[샘플 예측 결과] (10개)")
    print("=" * 60)
    for i in range(min(10, len(results['predictions']))):
        pred_vec = results['predictions'][i]
        label_vec = results['labels'][i]
        
        # 예측된 클래스들
        pred_classes = [idx_to_name[j] for j in range(len(pred_vec)) if pred_vec[j] == 1.0]
        # 실제 클래스들
        true_classes = [idx_to_name[j] for j in range(len(label_vec)) if label_vec[j] == 1.0]
        
        is_correct = (pred_vec == label_vec).all()
        correct_mark = "정확" if is_correct else "오류"
        
        print(f"\n  샘플 {i+1}: {Path(results['paths'][i]).name}")
        print(f"    - 실제: {', '.join(true_classes) if true_classes else 'Normal'}")
        print(f"    - 예측: {', '.join(pred_classes) if pred_classes else 'Normal'} ({correct_mark})")
    
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

