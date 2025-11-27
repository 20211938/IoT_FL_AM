"""
학습 로그 파일을 읽어서 학습 히스토리 그래프를 생성하는 스크립트
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# 경로 유틸리티 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import LOGS_DIR, OUTPUT_DIR, to_str

# 로그 파일에서 정규 표현식으로 데이터 추출
PATTERN_EPOCH = re.compile(r'\[Epoch (\d+)/(\d+)\]')
PATTERN_TRAIN_LOSS = re.compile(r'Train Loss: ([\d.]+)')
PATTERN_TRAIN_ACC = re.compile(r'Train Acc: ([\d.]+)%')
PATTERN_VAL_LOSS = re.compile(r'Val Loss: ([\d.]+)')
PATTERN_VAL_ACC = re.compile(r'Val Acc: ([\d.]+)%')
PATTERN_TEST_LOSS = re.compile(r'테스트 손실: ([\d.]+)')
PATTERN_TEST_ACC = re.compile(r'테스트 정확도: ([\d.]+)%')


def parse_log_file(log_file_path: str) -> Dict:
    """
    로그 파일을 파싱하여 학습 히스토리 추출
    
    Args:
        log_file_path: 로그 파일 경로
        
    Returns:
        학습 히스토리 딕셔너리 (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    """
    print(f"\n[로그 파일 파싱] {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"[오류] 로그 파일을 찾을 수 없습니다: {log_file_path}")
        return {}
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_epoch = None
    train_loss = None
    train_acc = None
    val_loss = None
    val_acc = None
    test_loss = None
    test_acc = None
    
    for line in lines:
        # 에포크 정보 추출
        epoch_match = PATTERN_EPOCH.search(line)
        if epoch_match:
            # 이전 에포크 데이터 저장
            if current_epoch is not None:
                if train_loss is not None:
                    history['train_loss'].append(train_loss)
                if train_acc is not None:
                    history['train_acc'].append(train_acc)
                if val_loss is not None:
                    history['val_loss'].append(val_loss)
                if val_acc is not None:
                    history['val_acc'].append(val_acc)
            
            current_epoch = int(epoch_match.group(1))
            train_loss = None
            train_acc = None
            val_loss = None
            val_acc = None
        
        # Train Loss 추출
        train_loss_match = PATTERN_TRAIN_LOSS.search(line)
        if train_loss_match:
            train_loss = float(train_loss_match.group(1))
        
        # Train Acc 추출
        train_acc_match = PATTERN_TRAIN_ACC.search(line)
        if train_acc_match:
            train_acc = float(train_acc_match.group(1))
        
        # Val Loss 추출
        val_loss_match = PATTERN_VAL_LOSS.search(line)
        if val_loss_match:
            val_loss = float(val_loss_match.group(1))
        
        # Val Acc 추출
        val_acc_match = PATTERN_VAL_ACC.search(line)
        if val_acc_match:
            val_acc = float(val_acc_match.group(1))
        
        # Test Loss 추출
        test_loss_match = PATTERN_TEST_LOSS.search(line)
        if test_loss_match:
            test_loss = float(test_loss_match.group(1))
        
        # Test Acc 추출
        test_acc_match = PATTERN_TEST_ACC.search(line)
        if test_acc_match:
            test_acc = float(test_acc_match.group(1))
    
    # 마지막 에포크 데이터 저장
    if current_epoch is not None:
        if train_loss is not None:
            history['train_loss'].append(train_loss)
        if train_acc is not None:
            history['train_acc'].append(train_acc)
        if val_loss is not None:
            history['val_loss'].append(val_loss)
        if val_acc is not None:
            history['val_acc'].append(val_acc)
    
    # 테스트 결과 저장 (마지막 한 번만 있음)
    if test_loss is not None:
        history['test_loss'] = [test_loss]
    if test_acc is not None:
        history['test_acc'] = [test_acc]
    
    print(f"  - 추출된 에포크 수: {len(history['train_loss'])}")
    print(f"  - Train Loss: {len(history['train_loss'])}개")
    print(f"  - Train Acc: {len(history['train_acc'])}개")
    print(f"  - Val Loss: {len(history['val_loss'])}개")
    print(f"  - Val Acc: {len(history['val_acc'])}개")
    if history['test_loss']:
        print(f"  - Test Loss: {len(history['test_loss'])}개")
    if history['test_acc']:
        print(f"  - Test Acc: {len(history['test_acc'])}개")
    
    return history


def plot_training_history_from_log(history: Dict, save_path: str = "training_history_from_log.png"):
    """
    학습 히스토리 그래프 생성 (손실 및 정확도)
    
    Args:
        history: 학습 히스토리 딕셔너리
        save_path: 그래프 저장 경로
    """
    if not history or not history['train_loss']:
        print("[경고] 학습 히스토리가 없습니다.")
        return
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 2개의 서브플롯: 손실 및 정확도
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 손실 그래프
    if history['train_loss']:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', 
                linewidth=2, marker='o', markersize=4)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', 
                linewidth=2, marker='s', markersize=4)
    if history.get('test_loss') and history['test_loss']:
        test_loss_val = history['test_loss'][-1]
        ax1.axhline(y=test_loss_val, color='g', linestyle='--', 
                   label=f'Test Loss: {test_loss_val:.4f}', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss (from log)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 정확도 그래프
    if history['train_acc']:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', 
                linewidth=2, marker='o', markersize=4)
    if history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', 
                linewidth=2, marker='s', markersize=4)
    if history.get('test_acc') and history['test_acc']:
        test_acc_val = history['test_acc'][-1]
        ax2.axhline(y=test_acc_val, color='g', linestyle='--', 
                   label=f'Test Acc: {test_acc_val:.2f}%', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy (from log)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[학습 히스토리 그래프]")
    print(f"  - 저장 경로: {save_path}")
    print(f"  - 총 에포크: {len(epochs)}")
    
    # 통계 출력
    if history['train_acc']:
        print(f"  - 최종 Train Acc: {history['train_acc'][-1]:.2f}%")
        print(f"  - 최고 Train Acc: {max(history['train_acc']):.2f}%")
    if history['val_acc']:
        print(f"  - 최종 Val Acc: {history['val_acc'][-1]:.2f}%")
        print(f"  - 최고 Val Acc: {max(history['val_acc']):.2f}%")
    if history.get('test_acc') and history['test_acc']:
        print(f"  - Test Acc: {history['test_acc'][-1]:.2f}%")


def find_latest_log_file(log_dir: str = "logs") -> Optional[str]:
    """
    로그 디렉토리에서 최신 로그 파일 찾기
    
    Args:
        log_dir: 로그 디렉토리 경로
        
    Returns:
        최신 로그 파일 경로 또는 None
    """
    if not os.path.exists(log_dir):
        return None
    
    log_files = list(Path(log_dir).glob("training_log_*.txt"))
    if not log_files:
        return None
    
    # 파일명의 타임스탬프 기준으로 정렬 (최신 것 먼저)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(log_files[0])


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='학습 로그 파일에서 그래프 생성')
    parser.add_argument('--log-file', type=str, default=None,
                       help='로그 파일 경로 (기본값: logs 디렉토리의 최신 파일)')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='로그 디렉토리 (기본값: logs)')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 파일 경로 (기본값: 로그 파일과 같은 디렉토리)')
    
    args = parser.parse_args()
    
    # 로그 파일 경로 결정
    log_file_path = args.log_file
    if log_file_path is None:
        log_file_path = find_latest_log_file(args.log_dir)
        if log_file_path is None:
            print(f"[오류] 로그 파일을 찾을 수 없습니다. --log-file 옵션으로 경로를 지정하세요.")
            return
        print(f"[찾은 로그 파일] {log_file_path}")
    
    # 로그 파일 파싱
    history = parse_log_file(log_file_path)
    
    if not history or not history['train_loss']:
        print("[오류] 로그 파일에서 학습 히스토리를 추출할 수 없습니다.")
        return
    
    # 출력 파일 경로 결정
    output_path = args.output
    if output_path is None:
        log_file_path_obj = Path(log_file_path)
        if log_file_path_obj.parent.exists():
            output_path = to_str(log_file_path_obj.parent / "training_history_from_log.png")
        else:
            output_path = to_str(OUTPUT_DIR / "training_history_from_log.png")
    
    # 그래프 생성
    plot_training_history_from_log(history, output_path)
    
    print("\n" + "=" * 80)
    print("그래프 생성 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

