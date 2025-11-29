"""
데이터 증강 스크립트
각 이미지와 annotation에 대해 15개의 증강 데이터를 생성합니다.
- 회전: 0, 90, 180, 270도 (4가지)
- 좌우 반전: 없음, 있음 (2가지)
- 상하 반전: 없음, 있음 (2가지)
총 16가지 조합 (원본 포함, 원본 제외하면 15개)
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import itertools


def rotate_image(image, angle):
    """이미지를 지정된 각도로 회전"""
    if angle == 0:
        return image
    return image.rotate(-angle, expand=False, fillcolor=0)


def rotate_mask(mask, angle):
    """마스크를 지정된 각도로 회전"""
    if angle == 0:
        return mask
    # PIL Image로 변환하여 회전
    mask_img = Image.fromarray(mask.astype(np.uint8))
    rotated = mask_img.rotate(-angle, expand=False, fillcolor=0)
    return np.array(rotated)


def flip_image(image, flip_lr=False, flip_ud=False):
    """이미지를 좌우/상하 반전"""
    if flip_lr:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip_ud:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def flip_mask(mask, flip_lr=False, flip_ud=False):
    """마스크를 좌우/상하 반전"""
    if flip_lr:
        mask = np.fliplr(mask)
    if flip_ud:
        mask = np.flipud(mask)
    return mask


def augment_image_and_mask(image, mask, rotation, flip_lr, flip_ud):
    """
    이미지와 마스크에 증강 적용
    
    Args:
        image: PIL Image
        mask: numpy array
        rotation: 회전 각도 (0, 90, 180, 270)
        flip_lr: 좌우 반전 여부
        flip_ud: 상하 반전 여부
    
    Returns:
        augmented_image: 증강된 PIL Image
        augmented_mask: 증강된 numpy array
    """
    # 회전 적용
    aug_image = rotate_image(image, rotation)
    aug_mask = rotate_mask(mask, rotation)
    
    # 반전 적용
    aug_image = flip_image(aug_image, flip_lr, flip_ud)
    aug_mask = flip_mask(aug_mask, flip_lr, flip_ud)
    
    return aug_image, aug_mask


def generate_augmentation_combinations():
    """
    모든 증강 조합 생성
    원본(rotation=0, flip_lr=False, flip_ud=False)을 포함하여 16개
    """
    rotations = [0, 90, 180, 270]
    flip_lr_options = [False, True]
    flip_ud_options = [False, True]
    
    combinations = list(itertools.product(
        rotations, flip_lr_options, flip_ud_options
    ))
    
    return combinations


def get_augmentation_suffix(rotation, flip_lr, flip_ud):
    """증강 조합에 대한 파일명 접미사 생성"""
    suffix_parts = []
    
    if rotation != 0:
        suffix_parts.append(f"rot{rotation}")
    
    if flip_lr:
        suffix_parts.append("fliplr")
    
    if flip_ud:
        suffix_parts.append("flipud")
    
    if not suffix_parts:
        return "_aug0"  # 원본
    
    return "_" + "_".join(suffix_parts)


def process_dataset(data_dir='data_test'):
    """
    모든 데이터셋에 대해 데이터 증강 수행
    증강된 데이터는 data_augments 폴더에 저장되며, 원본 데이터는 그대로 유지됩니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"오류: {data_dir} 디렉토리를 찾을 수 없습니다.")
        return
    
    # data_augments 폴더 생성 (원본 데이터와 같은 레벨)
    output_path = data_path.parent / 'data_augments'
    output_path.mkdir(exist_ok=True)
    print(f"증강 데이터 저장 경로: {output_path}")
    
    # 데이터 구조 확인: data_dir 바로 아래에 0, 1, annotations 폴더가 있는지 확인
    folder_0 = data_path / '0'
    folder_1 = data_path / '1'
    annotations_folder = data_path / 'annotations'
    
    # 평면 구조인지 확인 (data_test 구조)
    if folder_0.exists() and folder_1.exists() and annotations_folder.exists():
        print("평면 구조 감지: data_dir/0, data_dir/1, data_dir/annotations")
        
        # 출력 폴더 구조 생성 (data_augments/0, 1, annotations)
        output_folder_0 = output_path / '0'
        output_folder_1 = output_path / '1'
        output_annotations_folder = output_path / 'annotations'
        
        output_folder_0.mkdir(parents=True, exist_ok=True)
        output_folder_1.mkdir(parents=True, exist_ok=True)
        output_annotations_folder.mkdir(parents=True, exist_ok=True)
        
        # 증강 조합 생성
        aug_combinations = generate_augmentation_combinations()
        print(f"각 이미지당 {len(aug_combinations)}개의 증강 데이터를 생성합니다.")
        
        total_images = 0
        total_augmented = 0
        
        # 0 폴더의 모든 jpg 파일 찾기
        jpg_files = list(folder_0.glob('*.jpg'))
        
        if not jpg_files:
            print(f"경고: {folder_0}에 jpg 파일을 찾을 수 없습니다.")
            return
        
        print(f"{len(jpg_files)}개의 이미지 파일 발견")
        
        # 각 이미지 처리
        for jpg_file in tqdm(jpg_files, desc="증강 처리"):
            file_stem = jpg_file.stem  # 파일명에서 확장자 제거 (예: '000001')
            
            # 원본 이미지 파일 경로
            img_0_path = folder_0 / f"{file_stem}.jpg"
            img_1_path = folder_1 / f"{file_stem}.jpg"
            mask_path = annotations_folder / f"{file_stem}.npy"
            
            # 파일 존재 확인
            if not img_0_path.exists():
                print(f"  경고: {img_0_path}를 찾을 수 없습니다.")
                continue
            if not img_1_path.exists():
                print(f"  경고: {img_1_path}를 찾을 수 없습니다.")
                continue
            if not mask_path.exists():
                print(f"  경고: {mask_path}를 찾을 수 없습니다.")
                continue
            
            # 이미지와 마스크 로드
            try:
                img_0 = Image.open(img_0_path)
                img_1 = Image.open(img_1_path)
                mask = np.load(mask_path)
            except Exception as e:
                print(f"  오류: {file_stem} 파일 로드 실패 - {e}")
                continue
            
            total_images += 1
            
            # 각 증강 조합에 대해 처리
            for rotation, flip_lr, flip_ud in aug_combinations:
                # 증강 적용
                aug_img_0, aug_mask = augment_image_and_mask(
                    img_0, mask, rotation, flip_lr, flip_ud
                )
                aug_img_1, _ = augment_image_and_mask(
                    img_1, mask, rotation, flip_lr, flip_ud
                )
                
                # 파일명 접미사 생성
                suffix = get_augmentation_suffix(rotation, flip_lr, flip_ud)
                
                # 원본인 경우 (rotation=0, flip_lr=False, flip_ud=False)
                # 원본은 원래 위치에 그대로 두고, data_augments에는 저장하지 않음
                if suffix == "_aug0":
                    continue
                
                # 증강된 파일을 data_augments 폴더에 저장
                aug_img_0_path = output_folder_0 / f"{file_stem}{suffix}.jpg"
                aug_img_1_path = output_folder_1 / f"{file_stem}{suffix}.jpg"
                aug_mask_path = output_annotations_folder / f"{file_stem}{suffix}.npy"
                
                try:
                    aug_img_0.save(aug_img_0_path, quality=95)
                    aug_img_1.save(aug_img_1_path, quality=95)
                    np.save(aug_mask_path, aug_mask)
                    total_augmented += 1
                except Exception as e:
                    print(f"  오류: {file_stem}{suffix} 저장 실패 - {e}")
        
        print(f"\n=== 증강 완료 ===")
        print(f"원본 이미지 수: {total_images}")
        print(f"생성된 증강 데이터 수: {total_augmented}")
        print(f"증강 데이터 저장 위치: {output_path}")
        print(f"예상 총 데이터 수: {total_images * len(aug_combinations)} = {total_images} × {len(aug_combinations)}")
        
    else:
        # 중첩 구조인 경우 (기존 로직)
        print("중첩 구조 감지: data_dir/데이터셋명/0, data_dir/데이터셋명/1")
        
        # 모든 데이터셋 폴더 찾기
        dataset_folders = [d for d in data_path.iterdir() 
                          if d.is_dir() and (d / '0').exists() and (d / '1').exists()]
        
        if not dataset_folders:
            print(f"오류: {data_dir}에 데이터셋 폴더를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(dataset_folders)}개의 데이터셋을 찾았습니다.")
        
        # 증강 조합 생성
        aug_combinations = generate_augmentation_combinations()
        print(f"각 이미지당 {len(aug_combinations)}개의 증강 데이터를 생성합니다.")
        
        total_images = 0
        total_augmented = 0
        
        # 각 데이터셋 처리
        for dataset_folder in dataset_folders:
            print(f"\n처리 중: {dataset_folder.name}")
            
            folder_0 = dataset_folder / '0'
            folder_1 = dataset_folder / '1'
            annotations_folder = dataset_folder / 'annotations'
            
            if not annotations_folder.exists():
                print(f"  경고: {annotations_folder} 폴더를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # 출력 폴더 구조 생성 (data_augments/데이터셋명/0, 1, annotations)
            output_dataset_folder = output_path / dataset_folder.name
            output_folder_0 = output_dataset_folder / '0'
            output_folder_1 = output_dataset_folder / '1'
            output_annotations_folder = output_dataset_folder / 'annotations'
            
            output_folder_0.mkdir(parents=True, exist_ok=True)
            output_folder_1.mkdir(parents=True, exist_ok=True)
            output_annotations_folder.mkdir(parents=True, exist_ok=True)
            
            # 0 폴더의 모든 jpg 파일 찾기
            jpg_files = list(folder_0.glob('*.jpg'))
            
            if not jpg_files:
                print(f"  경고: {folder_0}에 jpg 파일을 찾을 수 없습니다.")
                continue
            
            print(f"  {len(jpg_files)}개의 이미지 파일 발견")
            
            # 각 이미지 처리
            for jpg_file in tqdm(jpg_files, desc=f"  {dataset_folder.name}"):
                file_stem = jpg_file.stem  # 파일명에서 확장자 제거 (예: '0000000')
                
                # 원본 이미지 파일 경로
                img_0_path = folder_0 / f"{file_stem}.jpg"
                img_1_path = folder_1 / f"{file_stem}.jpg"
                mask_path = annotations_folder / f"{file_stem}.npy"
                
                # 파일 존재 확인
                if not img_0_path.exists():
                    print(f"    경고: {img_0_path}를 찾을 수 없습니다.")
                    continue
                if not img_1_path.exists():
                    print(f"    경고: {img_1_path}를 찾을 수 없습니다.")
                    continue
                if not mask_path.exists():
                    print(f"    경고: {mask_path}를 찾을 수 없습니다.")
                    continue
                
                # 이미지와 마스크 로드
                try:
                    img_0 = Image.open(img_0_path)
                    img_1 = Image.open(img_1_path)
                    mask = np.load(mask_path)
                except Exception as e:
                    print(f"    오류: {file_stem} 파일 로드 실패 - {e}")
                    continue
                
                total_images += 1
                
                # 각 증강 조합에 대해 처리
                for rotation, flip_lr, flip_ud in aug_combinations:
                    # 증강 적용
                    aug_img_0, aug_mask = augment_image_and_mask(
                        img_0, mask, rotation, flip_lr, flip_ud
                    )
                    aug_img_1, _ = augment_image_and_mask(
                        img_1, mask, rotation, flip_lr, flip_ud
                    )
                    
                    # 파일명 접미사 생성
                    suffix = get_augmentation_suffix(rotation, flip_lr, flip_ud)
                    
                    # 원본인 경우 (rotation=0, flip_lr=False, flip_ud=False)
                    # 원본은 원래 위치에 그대로 두고, data_augments에는 저장하지 않음
                    if suffix == "_aug0":
                        continue
                    
                    # 증강된 파일을 data_augments 폴더에 저장
                    aug_img_0_path = output_folder_0 / f"{file_stem}{suffix}.jpg"
                    aug_img_1_path = output_folder_1 / f"{file_stem}{suffix}.jpg"
                    aug_mask_path = output_annotations_folder / f"{file_stem}{suffix}.npy"
                    
                    try:
                        aug_img_0.save(aug_img_0_path, quality=95)
                        aug_img_1.save(aug_img_1_path, quality=95)
                        np.save(aug_mask_path, aug_mask)
                        total_augmented += 1
                    except Exception as e:
                        print(f"    오류: {file_stem}{suffix} 저장 실패 - {e}")
            
            print(f"  완료: {dataset_folder.name}")
        
        print(f"\n=== 증강 완료 ===")
        print(f"원본 이미지 수: {total_images}")
        print(f"생성된 증강 데이터 수: {total_augmented}")
        print(f"증강 데이터 저장 위치: {output_path}")
        print(f"예상 총 데이터 수: {total_images * len(aug_combinations)} = {total_images} × {len(aug_combinations)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터 증강 스크립트')
    parser.add_argument('--data_dir', type=str, default='data_test',
                       help='데이터 디렉토리 경로 (기본값: data_test)')
    
    args = parser.parse_args()
    
    process_dataset(args.data_dir)

