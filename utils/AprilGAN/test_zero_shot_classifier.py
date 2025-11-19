"""
AprilGAN 기반 제로샷 결함 분류 모델 테스트 스크립트
학습된 모델을 사용하여 결함 분류 성능을 평가합니다.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.AprilGAN.zero_shot_defect_classifier import AprilGANZeroShotClassifier, LabeledImageDataset


def test_model(checkpoint_path=None, data_dir="data/labeled_layers", 
               max_eval_samples=100, num_sample_tests=10):
    """
    학습된 모델을 테스트합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로 (None이면 최신 체크포인트 사용)
        data_dir: 테스트 데이터 디렉토리
        max_eval_samples: 평가할 최대 샘플 수
        num_sample_tests: 상세 테스트할 샘플 수
    """
    print("=" * 60)
    print("AprilGAN 기반 제로샷 결함 분류 모델 테스트")
    print("=" * 60)
    
    # 체크포인트 경로 확인
    checkpoint_dir = "checkpoints"
    if checkpoint_path is None:
        if not os.path.exists(checkpoint_dir):
            print(f"\n[오류] 체크포인트 디렉토리가 없습니다: {checkpoint_dir}")
            print("  먼저 모델을 학습하세요.")
            return
        
        checkpoints = sorted(Path(checkpoint_dir).glob("aprilgan_epoch_*.pth"))
        if not checkpoints:
            print(f"\n[오류] 체크포인트 파일이 없습니다: {checkpoint_dir}")
            print("  먼저 모델을 학습하세요.")
            return
        
        checkpoint_path = str(checkpoints[-1])
        print(f"\n[체크포인트] 최신 체크포인트 사용: {checkpoint_path}")
    else:
        if not os.path.exists(checkpoint_path):
            print(f"\n[오류] 체크포인트 파일이 없습니다: {checkpoint_path}")
            return
        print(f"\n[체크포인트] 지정된 체크포인트 사용: {checkpoint_path}")
    
    # 모델 초기화
    print("\n[1단계] 모델 초기화")
    classifier = AprilGANZeroShotClassifier()
    classifier.initialize_model(
        model_path=checkpoint_path,
        lr_g=0.0003,
        lr_d=0.0001
    )
    print("  - 모델 로드 완료")
    
    # 데이터 로드
    print("\n[2단계] 데이터 로딩")
    dataset = classifier.load_labeled_data(data_dir)
    
    if len(dataset) == 0:
        print("\n[오류] 로드된 이미지가 없습니다. 데이터 디렉토리를 확인하세요.")
        return
    
    print(f"  - 총 {len(dataset)}개 이미지 로드됨")
    
    # 학습 히스토리 시각화 (있는 경우)
    if classifier.training_history and len(classifier.training_history.get('g_losses', [])) > 0:
        print("\n[3단계] 학습 히스토리 시각화")
        try:
            classifier.plot_training_history()
            print("  - 학습 히스토리 시각화 완료")
        except Exception as e:
            print(f"  - 시각화 오류: {e}")
    
    # 전체 데이터셋 평가
    print("\n[4단계] 전체 데이터셋 평가")
    evaluation_results = classifier.evaluate_classification(
        dataset=dataset,
        batch_size=32,
        max_samples=max_eval_samples
    )
    
    print(f"\n[평가 결과 요약]")
    print(f"  - 평가된 샘플 수: {evaluation_results['total_samples']}")
    print(f"  - 정확도: {evaluation_results['accuracy']:.2%}")
    print(f"  - 이상 탐지된 샘플: {evaluation_results['anomaly_detected']}개")
    print(f"  - 정상 샘플: {evaluation_results['total_samples'] - evaluation_results['anomaly_detected']}개")
    
    # 샘플 이미지 상세 테스트
    print(f"\n[5단계] 샘플 이미지 상세 테스트 ({num_sample_tests}개)")
    
    # 정상/결함 샘플 분류
    normal_samples = []
    defect_samples = []
    
    print("  - 샘플 분류 중...")
    for i in range(min(500, len(dataset))):  # 최대 500개까지 검색
        _, metadata, _ = dataset[i]
        if metadata.get('IsDefected', False):
            defect_samples.append(i)
        else:
            normal_samples.append(i)
        
        # 충분한 샘플을 찾으면 중단
        if len(normal_samples) >= num_sample_tests and len(defect_samples) >= num_sample_tests:
            break
    
    print(f"  - 정상 샘플: {len(normal_samples)}개 발견")
    print(f"  - 결함 샘플: {len(defect_samples)}개 발견")
    
    # 정상 샘플 테스트
    if normal_samples:
        print(f"\n  [정상 샘플 테스트] (상위 {min(num_sample_tests, len(normal_samples))}개)")
        correct_normal = 0
        for idx, sample_idx in enumerate(normal_samples[:num_sample_tests]):
            sample_image, sample_metadata, sample_path = dataset[sample_idx]
            img_pil = transforms.ToPILImage()(sample_image)
            
            print(f"\n    샘플 {idx+1}: {Path(sample_path).name}")
            print(f"      - 실제 레이블: 정상")
            
            classification = classifier.classify_defects(img_pil, verbose=False)
            is_correct = not classification['is_anomaly']
            
            if is_correct:
                correct_normal += 1
            
            print(f"      - 예측: {'정상' if not classification['is_anomaly'] else '이상'} {'(정확)' if is_correct else '(오류)'}")
            print(f"      - 이상 점수: {classification['anomaly_score']:.4f}")
            print(f"      - 예측 결함: {classification['predicted_defect']}")
        
        normal_accuracy = correct_normal / min(num_sample_tests, len(normal_samples))
        print(f"\n    정상 샘플 정확도: {normal_accuracy:.2%} ({correct_normal}/{min(num_sample_tests, len(normal_samples))})")
    
    # 결함 샘플 테스트
    if defect_samples:
        print(f"\n  [결함 샘플 테스트] (상위 {min(num_sample_tests, len(defect_samples))}개)")
        correct_defect = 0
        for idx, sample_idx in enumerate(defect_samples[:num_sample_tests]):
            sample_image, sample_metadata, sample_path = dataset[sample_idx]
            img_pil = transforms.ToPILImage()(sample_image)
            true_labels = classifier.extract_defect_info_from_metadata(sample_metadata)
            
            print(f"\n    샘플 {idx+1}: {Path(sample_path).name}")
            print(f"      - 실제 레이블: {true_labels}")
            
            classification = classifier.classify_defects(img_pil, verbose=False)
            is_correct = classification['is_anomaly']
            
            if is_correct:
                correct_defect += 1
            
            print(f"      - 예측: {'이상' if classification['is_anomaly'] else '정상'} {'(정확)' if is_correct else '(오류)'}")
            print(f"      - 이상 점수: {classification['anomaly_score']:.4f}")
            print(f"      - 예측 결함: {classification['predicted_defect']}")
            print(f"      - 결함 유형별 점수 (상위 3개):")
            for defect_type, score in sorted(classification['defect_scores'].items(), 
                                             key=lambda x: x[1], reverse=True)[:3]:
                print(f"        • {defect_type}: {score:.4f}")
        
        defect_accuracy = correct_defect / min(num_sample_tests, len(defect_samples))
        print(f"\n    결함 샘플 정확도: {defect_accuracy:.2%} ({correct_defect}/{min(num_sample_tests, len(defect_samples))})")
    else:
        print("\n  [경고] 결함 샘플을 찾을 수 없습니다.")
    
    # 첫 번째 샘플 상세 분석
    print(f"\n[6단계] 첫 번째 샘플 상세 분석")
    sample_image, sample_metadata, sample_path = dataset[0]
    img_pil = transforms.ToPILImage()(sample_image)
    
    print(f"\n  - 샘플 이미지 경로: {sample_path}")
    print(f"  - 레이어 번호: {sample_metadata.get('LayerNum', 'N/A')}")
    print(f"  - 결함 여부: {sample_metadata.get('IsDefected', 'N/A')}")
    
    # 실제 레이블 정보
    true_labels = classifier.extract_defect_info_from_metadata(sample_metadata)
    print(f"  - 실제 레이블: {true_labels}")
    
    # 결함 분류
    classification = classifier.classify_defects(img_pil, verbose=True)
    
    print(f"\n  [분류 결과 요약]")
    print(f"    - 이상 탐지: {'예' if classification['is_anomaly'] else '아니오'}")
    print(f"    - 이상 점수: {classification['anomaly_score']:.4f}")
    print(f"    - 예측된 결함: {classification['predicted_defect']}")
    print(f"\n    [결함 유형별 점수]")
    for defect_type, score in sorted(classification['defect_scores'].items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"      - {defect_type:20s}: {score:.4f}")
    
    # 시각화
    print(f"\n[7단계] 결과 시각화")
    try:
        classifier.visualize_defects(img_pil, classification)
        print("  - 시각화 완료")
    except Exception as e:
        print(f"  - 시각화 오류: {e}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AprilGAN 기반 제로샷 결함 분류 모델 테스트')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='체크포인트 파일 경로 (지정하지 않으면 최신 체크포인트 사용)')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='테스트 데이터 디렉토리 (기본값: data/labeled_layers)')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='평가할 최대 샘플 수 (기본값: 100)')
    parser.add_argument('--num-tests', type=int, default=10,
                       help='상세 테스트할 샘플 수 (기본값: 10)')
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        max_eval_samples=args.max_samples,
        num_sample_tests=args.num_tests
    )


if __name__ == "__main__":
    main()

