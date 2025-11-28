"""
결함 검출 및 분류 파이프라인 함수들
"""
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

from utils.cnn.defect_detection import detect_defects_with_unet, scan_data_directory, load_image_for_cnn
from utils.cnn.classifier import load_cnn_model


def detect_defects_batch(unet_model, data_dir, file_list, tileSize=128, defect_threshold=0.01):
    """
    U-Net을 사용하여 여러 이미지에 대해 결함 검출 수행
    
    Args:
        unet_model: 학습된 U-Net 모델
        data_dir: data 디렉터리 경로
        file_list: 처리할 파일명 리스트
        tileSize: 타일 크기
        defect_threshold: 결함 검출 임계값
        
    Returns:
        검출 결과 리스트 (각 결과는 file_name, has_defect, defect_ratio, mask 포함)
    """
    print("\n" + "="*60)
    print("1단계: U-Net 결함 검출 시작")
    print("="*60)
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    
    results = []
    
    print(f"총 {len(file_list)}개 파일 처리 중...")
    
    for idx, file_name in enumerate(file_list):
        try:
            img0_path = image_path0 / f"{file_name}.jpg"
            img1_path = image_path1 / f"{file_name}.jpg"
            
            # U-Net으로 결함 검출
            has_defect, mask = detect_defects_with_unet(
                unet_model, 
                str(img0_path), 
                str(img1_path), 
                tileSize=tileSize, 
                threshold=defect_threshold
            )
            
            # 결함 픽셀 비율 계산
            defect_ratio = np.sum(mask == 2) / mask.size if mask.size > 0 else 0.0
            
            results.append({
                'file_name': file_name,
                'has_defect': has_defect,
                'defect_ratio': defect_ratio,
                'mask': mask  # 나중에 시각화를 위해 저장
            })
            
            if (idx + 1) % 50 == 0:
                defect_count = sum(1 for r in results if r['has_defect'])
                print(f"  처리 중: {idx + 1}/{len(file_list)} 파일 완료 (결함 발견: {defect_count}개)")
        
        except Exception as e:
            print(f"  경고: {file_name} 처리 중 오류 발생: {e}")
            results.append({
                'file_name': file_name,
                'has_defect': False,
                'defect_ratio': 0.0,
                'error': str(e)
            })
            continue
    
    defect_count = sum(1 for r in results if r['has_defect'])
    print(f"\nU-Net 검출 완료! 총 {defect_count}개의 결함 이미지 발견")
    
    return results


def analyze_unet_results(unet_results, output_file='unet_detection_results.csv'):
    """
    U-Net 검출 결과 분석 및 요약
    
    Args:
        unet_results: U-Net 검출 결과 리스트
        output_file: 결과 저장 파일명
        
    Returns:
        DataFrame (마스크 제외)
    """
    # U-Net 검출 결과를 DataFrame으로 변환
    df_unet = pd.DataFrame([{
        'file_name': r['file_name'],
        'has_defect': r['has_defect'],
        'defect_ratio': r['defect_ratio']
    } for r in unet_results])
    
    # 결과 요약
    print("\n" + "="*60)
    print("U-Net 검출 결과 요약")
    print("="*60)
    print(f"총 파일 수: {len(df_unet)}")
    print(f"결함 발견: {df_unet['has_defect'].sum()}개")
    print(f"결함 없음: {(~df_unet['has_defect']).sum()}개")
    
    if df_unet['has_defect'].sum() > 0:
        defect_ratios = df_unet[df_unet['has_defect']]['defect_ratio']
        print(f"\n결함 픽셀 비율 통계:")
        print(f"  평균: {defect_ratios.mean():.6f}")
        print(f"  최소: {defect_ratios.min():.6f}")
        print(f"  최대: {defect_ratios.max():.6f}")
        print(f"  중앙값: {defect_ratios.median():.6f}")
    
    # 결과 저장
    df_unet.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nU-Net 검출 결과가 {output_file}에 저장되었습니다.")
    
    # 결과 미리보기
    print("\n" + "="*60)
    print("U-Net 검출 결과 미리보기 (상위 20개)")
    print("="*60)
    print(df_unet.head(20).to_string())
    
    return df_unet


def visualize_unet_results(df_unet):
    """
    U-Net 검출 결과 시각화
    
    Args:
        df_unet: U-Net 검출 결과 DataFrame
    """
    if df_unet['has_defect'].sum() > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 결함 발견 여부 분포
        defect_counts = df_unet['has_defect'].value_counts()
        axes[0].bar(['결함 없음', '결함 발견'], [defect_counts.get(False, 0), defect_counts.get(True, 0)])
        axes[0].set_ylabel('개수')
        axes[0].set_title('U-Net 검출 결과: 결함 발견 여부')
        axes[0].grid(axis='y', alpha=0.3)
        
        # 결함 픽셀 비율 분포
        defect_ratios = df_unet[df_unet['has_defect']]['defect_ratio']
        if len(defect_ratios) > 0:
            axes[1].hist(defect_ratios, bins=20, edgecolor='black')
            axes[1].set_xlabel('결함 픽셀 비율')
            axes[1].set_ylabel('빈도')
            axes[1].set_title('결함 픽셀 비율 분포')
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print("시각화할 결함 데이터가 없습니다.")


def visualize_defect_samples(unet_results, data_dir, num_samples=6):
    """
    결함이 발견된 이미지 샘플 시각화
    
    Args:
        unet_results: U-Net 검출 결과 리스트
        data_dir: data 디렉터리 경로
        num_samples: 시각화할 샘플 개수
    """
    defect_images = [r for r in unet_results if r['has_defect']]
    
    if len(defect_images) == 0:
        print("시각화할 결함 이미지가 없습니다.")
        return
    
    # 상위 N개 샘플 선택 (결함 비율이 높은 순서로)
    defect_images_sorted = sorted(defect_images, key=lambda x: x['defect_ratio'], reverse=True)[:num_samples]
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    
    # 그리드 크기 계산
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(defect_images_sorted):
        file_name = result['file_name']
        img0_path = image_path0 / f"{file_name}.jpg"
        
        # 원본 이미지 로드
        img = Image.open(img0_path)
        
        # 마스크 오버레이
        mask = result['mask']
        
        axes[idx].imshow(img, cmap='gray')
        # 마스크 오버레이 (결함 영역만)
        defect_mask = (mask == 2).astype(np.float32)
        axes[idx].imshow(defect_mask, alpha=0.3, cmap='Reds')
        axes[idx].set_title(f"{file_name}\n결함 비율: {result['defect_ratio']:.4f}")
        axes[idx].axis('off')
    
    # 빈 subplot 숨기기
    for idx in range(len(defect_images_sorted), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def load_cnn_model_safe(cnn_model_path, device=None):
    """
    CNN 모델 안전하게 로드
    
    Args:
        cnn_model_path: CNN 모델 경로
        device: PyTorch device (None이면 자동 감지)
        
    Returns:
        (cnn_model, label_mapping) 튜플
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"PyTorch device: {device}")
    
    if cnn_model_path and Path(cnn_model_path).exists() and cnn_model_path != 'none':
        print("CNN 모델 로드 중...")
        # .h5 파일이면 경고, .pth 또는 .pt 파일이어야 함
        if cnn_model_path.endswith('.h5'):
            print("경고: .h5 파일은 TensorFlow 모델입니다. PyTorch 모델(.pth 또는 .pt)이 필요합니다.")
            return None, None
        else:
            # num_classes는 모델에서 자동으로 추론됨
            cnn_model = load_cnn_model(cnn_model_path, device=device)
            print("CNN 모델 로드 완료!")
            print(f"모델 device: {next(cnn_model.parameters()).device}")
            label_mapping = None  # 모델에 저장된 매핑이 있다면 로드 필요
            return cnn_model, label_mapping
    else:
        print("경고: CNN 모델이 없습니다. 먼저 CNN 모델을 학습해야 합니다.")
        print("결함 검출만 수행합니다.")
        return None, None


def classify_defects_with_cnn(cnn_model, unet_results, data_dir, device=None, 
                               target_size=(640, 640), label_mapping=None):
    """
    CNN을 사용하여 결함이 발견된 이미지들에 대해 결함 유형 분류
    
    Args:
        cnn_model: 학습된 CNN 모델
        unet_results: U-Net 검출 결과 리스트
        data_dir: data 디렉터리 경로
        device: PyTorch device
        target_size: CNN 입력 이미지 크기
        label_mapping: 레이블 매핑 딕셔너리
        
    Returns:
        최종 결과 리스트 (U-Net 결과 + CNN 분류 결과)
    """
    if cnn_model is None:
        print("\nCNN 모델이 없어서 분류를 수행하지 않습니다.")
        return unet_results
    
    print("\n" + "="*60)
    print("3단계: CNN 결함 유형 분류 시작")
    print("="*60)
    
    # 결함이 발견된 이미지들만 필터링
    defect_file_names = [r['file_name'] for r in unet_results if r['has_defect']]
    print(f"분류할 결함 이미지: {len(defect_file_names)}개")
    
    if len(defect_file_names) == 0:
        print("분류할 결함 이미지가 없습니다.")
        return unet_results
    
    # CNN 모델을 eval 모드로 설정
    cnn_model.eval()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    
    final_results = []
    
    print(f"PyTorch device: {device}")
    
    for idx, file_name in enumerate(defect_file_names):
        try:
            img0_path = image_path0 / f"{file_name}.jpg"
            img1_path = image_path1 / f"{file_name}.jpg"
            
            # CNN 입력을 위한 이미지 로드
            image = load_image_for_cnn(str(img0_path), str(img1_path), target_size)
            
            # NumPy 배열을 PyTorch 텐서로 변환
            # (H, W, C) -> (C, H, W)로 변환하고 배치 차원 추가
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            # CNN으로 결함 유형 분류
            with torch.no_grad():
                outputs = cnn_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 레이블 매핑 적용
            if label_mapping:
                # label_mapping을 역으로 변환 (인덱스 -> 원본 값)
                reverse_mapping = {v: k for k, v in label_mapping.items()}
                defect_type = reverse_mapping.get(predicted_class, predicted_class)
            else:
                defect_type = predicted_class
            
            # U-Net 결과에 CNN 분류 결과 추가
            unet_result = next((r for r in unet_results if r['file_name'] == file_name), None)
            if unet_result:
                final_result = unet_result.copy()
                final_result['defect_type'] = defect_type
                final_result['defect_type_label'] = predicted_class
                final_result['confidence'] = float(confidence)
                final_results.append(final_result)
            
            if (idx + 1) % 10 == 0:
                print(f"  처리 중: {idx + 1}/{len(defect_file_names)} 파일 완료")
        
        except Exception as e:
            print(f"  경고: {file_name} 처리 중 오류 발생: {e}")
            # 오류가 발생한 경우 U-Net 결과만 포함
            unet_result = next((r for r in unet_results if r['file_name'] == file_name), None)
            if unet_result:
                final_result = unet_result.copy()
                final_result['error'] = str(e)
                final_results.append(final_result)
            continue
    
    # 결함이 없는 이미지들도 최종 결과에 추가
    for unet_result in unet_results:
        if not unet_result['has_defect']:
            final_result = unet_result.copy()
            final_result['defect_type'] = None
            final_result['defect_type_label'] = None
            final_result['confidence'] = None
            final_results.append(final_result)
    
    print(f"\nCNN 분류 완료!")
    
    return final_results


def analyze_final_results(final_results, cnn_model=None, output_file='defect_classification_results.csv'):
    """
    최종 결과 분석 및 요약
    
    Args:
        final_results: 최종 결과 리스트
        cnn_model: CNN 모델 (None이면 분류 결과 없음)
        output_file: 결과 저장 파일명
        
    Returns:
        DataFrame
    """
    if not final_results:
        print("\n결과가 없습니다.")
        return None
    
    # 마스크는 제외하고 저장
    df_results = pd.DataFrame([{
        'file_name': r['file_name'],
        'has_defect': r['has_defect'],
        'defect_ratio': r.get('defect_ratio', 0.0),
        'defect_type': r.get('defect_type', None),
        'defect_type_label': r.get('defect_type_label', None),
        'confidence': r.get('confidence', None)
    } for r in final_results])
    
    # 결과 요약
    print("\n" + "="*60)
    print("최종 결과 요약")
    print("="*60)
    print(f"총 파일 수: {len(df_results)}")
    print(f"결함 발견: {df_results['has_defect'].sum()}개")
    print(f"결함 없음: {(~df_results['has_defect']).sum()}개")
    
    if cnn_model is not None and df_results['has_defect'].sum() > 0:
        print("\n" + "="*60)
        print("결함 유형별 분포")
        print("="*60)
        defect_types = df_results[df_results['has_defect']]['defect_type'].value_counts().sort_index()
        for defect_type, count in defect_types.items():
            print(f"  결함 유형 {defect_type}: {count}개")
        
        print("\n" + "="*60)
        print("신뢰도 통계")
        print("="*60)
        confidences = df_results[df_results['has_defect']]['confidence']
        confidences = confidences.dropna()
        if len(confidences) > 0:
            print(f"  평균 신뢰도: {confidences.mean():.4f}")
            print(f"  최소 신뢰도: {confidences.min():.4f}")
            print(f"  최대 신뢰도: {confidences.max():.4f}")
    
    # 결과 저장
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n최종 결과가 {output_file}에 저장되었습니다.")
    
    # 결과 미리보기
    print("\n" + "="*60)
    print("최종 결과 미리보기 (상위 20개)")
    print("="*60)
    print(df_results.head(20).to_string())
    
    return df_results


def analyze_defect_by_type(final_results, cnn_model=None):
    """
    결함 유형별 상세 분석
    
    Args:
        final_results: 최종 결과 리스트
        cnn_model: CNN 모델 (None이면 분류 결과 없음)
    """
    if not final_results or cnn_model is None:
        print("\n결함이 발견되고 분류된 이미지가 없습니다.")
        return
    
    defect_images = [r for r in final_results if r['has_defect'] and r.get('defect_type') is not None]
    
    if len(defect_images) == 0:
        print("\n결함이 발견되고 분류된 이미지가 없습니다.")
        return
    
    print(f"\n" + "="*60)
    print(f"결함이 발견되고 분류된 이미지: {len(defect_images)}개")
    print("="*60)
    
    # 결함 유형별로 그룹화
    defect_by_type = defaultdict(list)
    for r in defect_images:
        defect_by_type[r['defect_type']].append({
            'file_name': r['file_name'],
            'confidence': r['confidence']
        })
    
    print("\n결함 유형별 이미지 목록:")
    for defect_type in sorted(defect_by_type.keys()):
        files = defect_by_type[defect_type]
        print(f"\n  결함 유형 {defect_type}: {len(files)}개")
        
        # 신뢰도 순으로 정렬
        files_sorted = sorted(files, key=lambda x: x['confidence'], reverse=True)
        
        # 상위 10개 출력
        print(f"    상위 10개 파일:")
        for i, file_info in enumerate(files_sorted[:10], 1):
            print(f"      {i}. {file_info['file_name']} (신뢰도: {file_info['confidence']:.4f})")
        
        if len(files) > 10:
            print(f"      ... 외 {len(files) - 10}개")
