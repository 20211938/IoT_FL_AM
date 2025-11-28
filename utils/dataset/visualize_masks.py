"""
data 폴더 내의 모든 .tif 이미지에 .npy 마스크를 오버레이하여 시각화하는 스크립트
"""
import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap

# 한글 폰트 설정 (필요시)
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'font.size': 12})


def create_colored_mask(mask, alpha=0.5):
    """
    마스크를 컬러맵으로 변환합니다.
    
    Args:
        mask: numpy 배열 마스크 (0: powder, 1: part, 2: defect)
        alpha: 투명도 (0-1)
    
    Returns:
        컬러 마스크 이미지 (RGBA)
    """
    # 마스크를 0-1 범위로 정규화
    normalized_mask = mask.astype(float) / 2.0  # 0, 1, 2 -> 0, 0.5, 1
    
    # 컬러맵 생성 (파란색 -> 노란색 -> 빨간색)
    # 0: powder (파란색), 1: part (노란색), 2: defect (빨간색)
    colors = ['#0000FF', '#FFFF00', '#FF0000']  # blue, yellow, red
    cmap = ListedColormap(colors)
    
    # 마스크를 컬러로 변환
    colored_mask = cmap(normalized_mask)
    
    # 알파 채널 설정 (마스크가 있는 부분만 표시)
    colored_mask[:, :, 3] = alpha * (mask > 0).astype(float)
    
    return colored_mask


def visualize_mask_overlay(image_path, mask_path, output_path, image_type='0'):
    """
    이미지에 마스크를 오버레이하여 시각화합니다.
    
    Args:
        image_path: 이미지 파일 경로 (.tif)
        mask_path: 마스크 파일 경로 (.npy)
        output_path: 출력 파일 경로
        image_type: 이미지 타입 ('0' 또는 '1')
    """
    try:
        # 이미지 로드
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # 그레이스케일 이미지를 RGB로 변환
        if len(img_array.shape) == 2:
            img_rgb = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 1:
            img_rgb = np.repeat(img_array, 3, axis=2)
        else:
            img_rgb = img_array
        
        # 이미지 정규화 (0-255 범위로)
        if img_rgb.max() > 255:
            img_rgb = (img_rgb / img_rgb.max() * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
        
        # 마스크 로드
        mask = np.load(mask_path)
        
        # 마스크 크기 조정 (이미지 크기에 맞춤)
        if mask.shape != img_rgb.shape[:2]:
            # 마스크를 이미지 크기에 맞게 리사이즈
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = mask_pil.resize((img_rgb.shape[1], img_rgb.shape[0]), Image.NEAREST)
            mask = np.array(mask_pil)
        
        # 마스크 전처리 (기존 코드와 동일)
        mask_processed = np.copy(mask)
        mask_processed[mask_processed == -1] = 0  # unlabeled를 powder로
        mask_processed[(mask_processed != 0) & (mask_processed != 1)] = 2  # defect 통합
        
        # 컬러 마스크 생성
        colored_mask = create_colored_mask(mask_processed, alpha=0.6)
        
        # 파일명 추출
        file_name = Path(image_path).stem
        
        # 시각화 생성
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'File: {file_name} (Type: {image_type})', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # 원본 이미지
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            axes[0].imshow(img_rgb[:, :, 0], cmap='gray')
        else:
            axes[0].imshow(img_rgb)
        axes[0].set_title(f'Original Image ({image_type})', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 마스크만 (컬러맵 사용)
        im = axes[1].imshow(mask_processed, cmap='viridis', vmin=0, vmax=2)
        axes[1].set_title('Segmentation Mask\n(0: Powder, 1: Part, 2: Defect)', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.set_ticklabels(['Powder', 'Part', 'Defect'])
        cbar.set_label('Class', rotation=270, labelpad=15)
        
        # 오버레이
        if len(img_array.shape) == 2 or img_array.shape[2] == 1:
            axes[2].imshow(img_rgb[:, :, 0], cmap='gray')
        else:
            axes[2].imshow(img_rgb)
        axes[2].imshow(colored_mask)
        axes[2].set_title('Mask Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return True
        
    except Exception as e:
        print(f"오류 발생 ({image_path}): {str(e)}")
        return False


def process_all_datasets(data_dir='data', output_dir='visualized'):
    """
    data 폴더 내의 모든 데이터셋에 대해 마스크 시각화를 수행합니다.
    
    Args:
        data_dir: 데이터 폴더 경로
        output_dir: 출력 폴더 경로
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"오류: {data_dir} 폴더를 찾을 수 없습니다.")
        return
    
    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    
    # 모든 데이터셋 폴더 찾기
    dataset_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not dataset_folders:
        print("데이터셋 폴더를 찾을 수 없습니다.")
        return
    
    total_success = 0
    total_fail = 0
    
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        print(f"\n처리 중: {dataset_name}")
        
        # 0, 1 폴더와 annotations 폴더 확인
        folder_0 = dataset_folder / '0'
        folder_1 = dataset_folder / '1'
        annotations_folder = dataset_folder / 'annotations'
        
        if not annotations_folder.exists():
            print(f"  경고: {dataset_name}에 annotations 폴더가 없습니다. 건너뜁니다.")
            continue
        
        # 출력 폴더 생성
        dataset_output = output_path / dataset_name
        dataset_output.mkdir(exist_ok=True)
        
        # 0 폴더 처리
        if folder_0.exists():
            tif_files_0 = list(folder_0.glob('*.tif'))
            print(f"  폴더 0: {len(tif_files_0)}개 파일 발견")
            
            for tif_file in tif_files_0:
                file_stem = tif_file.stem
                mask_file = annotations_folder / f'{file_stem}.npy'
                
                if mask_file.exists():
                    output_file = dataset_output / f'{file_stem}_0_overlay.png'
                    if visualize_mask_overlay(tif_file, mask_file, output_file, '0'):
                        total_success += 1
                    else:
                        total_fail += 1
                else:
                    print(f"    경고: {file_stem}.npy 마스크 파일을 찾을 수 없습니다.")
        
        # 1 폴더 처리
        if folder_1.exists():
            tif_files_1 = list(folder_1.glob('*.tif'))
            print(f"  폴더 1: {len(tif_files_1)}개 파일 발견")
            
            for tif_file in tif_files_1:
                file_stem = tif_file.stem
                mask_file = annotations_folder / f'{file_stem}.npy'
                
                if mask_file.exists():
                    output_file = dataset_output / f'{file_stem}_1_overlay.png'
                    if visualize_mask_overlay(tif_file, mask_file, output_file, '1'):
                        total_success += 1
                    else:
                        total_fail += 1
                else:
                    print(f"    경고: {file_stem}.npy 마스크 파일을 찾을 수 없습니다.")
    
    print(f"\n{'='*50}")
    print(f"처리 완료!")
    print(f"  성공: {total_success}개")
    print(f"  실패: {total_fail}개")
    print(f"  출력 폴더: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='data 폴더 내의 모든 .tif 이미지에 마스크를 오버레이하여 시각화')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='데이터 폴더 경로 (기본값: data)')
    parser.add_argument('--output-dir', type=str, default='visualized',
                        help='출력 폴더 경로 (기본값: visualized)')
    
    args = parser.parse_args()
    
    process_all_datasets(args.data_dir, args.output_dir)

