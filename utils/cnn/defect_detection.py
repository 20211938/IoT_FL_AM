import numpy as np
import torch
from PIL import Image
from collections import Counter
import os
from pathlib import Path
from utils.cnn.classifier import initialize_cnn_classifier, load_cnn_model
from utils.u_net.image_processing import preprocess_image


def get_defect_type_from_npy(npy_path):
    '''
    원본 npy 파일에서 결함 유형 숫자 값들의 빈도를 계산하여 가장 많이 나타나는 결함 유형을 반환
    
    Args:
        npy_path: 원본 npy 파일 경로
    
    Returns:
        가장 많이 나타나는 결함 유형 숫자, 빈도 딕셔너리
    '''
    mask = np.load(npy_path)
    
    # 모든 값에 대해 빈도 계산 (필터링 없음)
    if mask.size == 0:
        return None, {}
    
    # 각 결함 유형의 빈도 계산
    defect_counts = Counter(mask.flatten())
    
    # 가장 많이 나타나는 결함 유형
    most_common_defect = defect_counts.most_common(1)[0][0]
    
    return most_common_defect, dict(defect_counts)


def detect_defects_with_unet(unet_model, image_path0, image_path1, tileSize=128, threshold=0.01):
    '''
    U-Net 모델을 사용하여 이미지에 결함이 있는지 검출
    
    Args:
        unet_model: 학습된 U-Net 모델 (TensorFlow)
        image_path0: Post Spreading 이미지 경로
        image_path1: Post Fusion 이미지 경로
        tileSize: 타일 크기
        threshold: 결함으로 판단할 픽셀 비율 임계값
    
    Returns:
        결함 존재 여부 (True/False), 예측 마스크
    '''
    # 이미지 로드
    im0 = Image.open(image_path0)
    im1 = Image.open(image_path1)
    
    # 더미 마스크 (전처리 함수 사용을 위해)
    dummy_mask = np.zeros((im0.size[1], im0.size[0]), dtype=np.int32)
    
    # 전처리 (타일로 분할)
    splitImages, _ = preprocess_image(im0, im1, dummy_mask, tileSize)
    
    # U-Net 예측
    predictions = unet_model.predict(splitImages, verbose=0)
    predicted_mask = tf.argmax(predictions, axis=-1)
    
    # 타일을 원본 이미지 크기로 재구성
    rightcrop = im0.size[0] // tileSize * tileSize
    bottomcrop = im0.size[1] // tileSize * tileSize
    
    num_tiles_per_row = rightcrop // tileSize
    num_tiles_per_col = bottomcrop // tileSize
    
    full_mask = np.zeros((bottomcrop, rightcrop), dtype=np.int32)
    
    for i, tile_mask in enumerate(predicted_mask.numpy()):
        row = i // num_tiles_per_row
        col = i % num_tiles_per_row
        y_start = row * tileSize
        y_end = y_start + tileSize
        x_start = col * tileSize
        x_end = x_start + tileSize
        full_mask[y_start:y_end, x_start:x_end] = tile_mask
    
    # 결함 픽셀 비율 계산
    defect_ratio = np.sum(full_mask == 2) / full_mask.size
    
    has_defect = defect_ratio > threshold
    
    return has_defect, full_mask


def load_image_for_cnn(image_path0, image_path1, target_size=(640, 640)):
    '''
    CNN 입력을 위한 이미지 로드 및 전처리
    
    Args:
        image_path0: Post Spreading 이미지 경로
        image_path1: Post Fusion 이미지 경로
        target_size: 목표 이미지 크기 (height, width)
    
    Returns:
        전처리된 이미지 배열 (height, width, 2) - NumPy 배열
    '''
    # 이미지 로드
    im0 = Image.open(image_path0)
    im1 = Image.open(image_path1)
    
    # 크기 조정
    im0 = im0.resize((target_size[1], target_size[0]), Image.LANCZOS)
    im1 = im1.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
    # NumPy 배열로 변환
    imarray0 = np.array(im0)
    imarray1 = np.array(im1)
    
    # 그레이스케일 변환
    if len(imarray0.shape) == 3 and imarray0.shape[2] == 3:
        imarray0 = np.mean(imarray0, axis=2, keepdims=True)
    elif len(imarray0.shape) == 2:
        imarray0 = np.expand_dims(imarray0, axis=2)
    
    if len(imarray1.shape) == 3 and imarray1.shape[2] == 3:
        imarray1 = np.mean(imarray1, axis=2, keepdims=True)
    elif len(imarray1.shape) == 2:
        imarray1 = np.expand_dims(imarray1, axis=2)
    
    # 정규화 [0, 1]
    imarray0 = imarray0.astype(np.float32) / 255.0
    imarray1 = imarray1.astype(np.float32) / 255.0
    
    # 채널 결합
    combined = np.concatenate([imarray0, imarray1], axis=-1)
    
    return combined


def scan_data_directory(data_dir='data'):
    '''
    data 디렉터리를 스캔하여 모든 이미지 파일 쌍을 찾음
    
    Args:
        data_dir: data 디렉터리 경로
    
    Returns:
        파일명 리스트 (확장자 제외)
    '''
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    annotations_path = data_path / 'annotations'
    
    if not image_path0.exists():
        raise ValueError(f"{image_path0} 디렉터리가 존재하지 않습니다.")
    
    # 0 폴더의 모든 jpg 파일 찾기
    jpg_files = list(image_path0.glob('*.jpg'))
    
    file_list = []
    for jpg_file in jpg_files:
        file_stem = jpg_file.stem
        img1_path = image_path1 / f"{file_stem}.jpg"
        npy_path = annotations_path / f"{file_stem}.npy"
        
        # 모든 파일이 존재하는지 확인
        if img1_path.exists() and npy_path.exists():
            file_list.append(file_stem)
    
    return sorted(file_list)


def detect_and_classify_defects(unet_model, cnn_model, data_dir='data', 
                                tileSize=128, target_size=(640, 640),
                                defect_threshold=0.01, label_mapping=None, device=None):
    '''
    U-Net으로 결함을 검출하고 CNN으로 결함 유형을 분류
    
    Args:
        unet_model: 학습된 U-Net 모델 (TensorFlow)
        cnn_model: 학습된 CNN 분류 모델 (PyTorch)
        data_dir: data 디렉터리 경로
        tileSize: U-Net 타일 크기
        target_size: CNN 입력 이미지 크기
        defect_threshold: 결함 검출 임계값
        label_mapping: 레이블 매핑 딕셔너리 (None이면 원본 값 반환)
        device: PyTorch device ('cuda' or 'cpu', None for auto-detect)
    
    Returns:
        결과 딕셔너리 리스트 (각 이미지에 대한 정보)
    '''
    import tensorflow as tf
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # CNN 모델을 eval 모드로 설정
    cnn_model.eval()
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    annotations_path = data_path / 'annotations'
    
    # 파일 리스트 가져오기
    file_list = scan_data_directory(data_dir)
    
    results = []
    
    print(f"총 {len(file_list)}개 파일 처리 중...")
    print(f"PyTorch device: {device}")
    
    for idx, file_name in enumerate(file_list):
        try:
            img0_path = image_path0 / f"{file_name}.jpg"
            img1_path = image_path1 / f"{file_name}.jpg"
            npy_path = annotations_path / f"{file_name}.npy"
            
            # U-Net으로 결함 검출
            has_defect, mask = detect_defects_with_unet(
                unet_model, str(img0_path), str(img1_path), tileSize, defect_threshold
            )
            
            result = {
                'file_name': file_name,
                'has_defect': has_defect,
                'defect_type': None,
                'defect_type_label': None,
                'confidence': None
            }
            
            if has_defect:
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
                
                result['defect_type'] = defect_type
                result['defect_type_label'] = predicted_class
                result['confidence'] = float(confidence)
            
            results.append(result)
            
            if (idx + 1) % 50 == 0:
                defect_count = sum(1 for r in results if r['has_defect'])
                print(f"  처리 중: {idx + 1}/{len(file_list)} 파일 완료 (결함 발견: {defect_count}개)")
        
        except Exception as e:
            print(f"  경고: {file_name} 처리 중 오류 발생: {e}")
            results.append({
                'file_name': file_name,
                'has_defect': False,
                'error': str(e)
            })
            continue
    
    defect_count = sum(1 for r in results if r['has_defect'])
    print(f"\n완료! 총 {defect_count}개의 결함 이미지 발견")
    
    return results