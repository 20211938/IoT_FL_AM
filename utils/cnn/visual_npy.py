import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from pathlib import Path
from collections import Counter

# 한글 폰트 설정
try:
    # Windows에서 한글 폰트 설정
    import platform
    if platform.system() == 'Windows':
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
    else:
        # Linux/Mac의 경우
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
except:
    # 폰트 설정 실패 시 기본 폰트 사용
    pass

def visualize_defect_types_samples(data_dir, client_identifier_dict, label_mapping, min_pixels=100):
    """
    각 결함 유형별로 이미지 하나씩 선택하여 마스킹을 이미지 위에 겹쳐서 시각화
    
    Args:
        data_dir: data 디렉터리 경로
        client_identifier_dict: 클라이언트별 파일 리스트 딕셔너리
        label_mapping: 레이블 매핑 딕셔너리 (원본 값 -> 인덱스)
        min_pixels: 선택할 최소 픽셀 수 (기본값: 100)
    """
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    annotations_path = data_path / 'annotations'
    
    # 역 매핑 생성 (인덱스 -> 원본 값)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # 각 결함 유형별 색상 매핑 (RGB 값) - 시각적으로 구분이 잘 되는 색상
    defect_color_map = {
        -1: (139, 69, 19),        # 갈색 (Brown) - Unlabeled
        0: (0, 0, 255),            # 파란색 (Blue) - Powder
        1: (50, 205, 50),          # 라임 그린 (Lime Green) - Part
        3: (255, 0, 0),            # 빨간색 (Red)
        4: (255, 140, 0),          # 다크 오렌지 (Dark Orange)
        5: (0, 255, 0),            # 초록색 (Green)
        6: (0, 191, 255),          # 딥 스카이 블루 (Deep Sky Blue)
        7: (255, 20, 147),         # 딥 핑크 (Deep Pink)
        8: (255, 105, 180),        # 핫 핑크 (Hot Pink)
        9: (138, 43, 226),         # 블루 바이올렛 (Blue Violet)
        11: (255, 215, 0),         # 골드 (Gold)
        14: (255, 69, 0),          # 레드 오렌지 (Red Orange)
        255: (255, 20, 20),        # 밝은 빨간색 (Bright Red) - 흰색 대신
    }
    
    # 각 결함 유형별로 이미지 하나씩 찾기 (픽셀 수가 min_pixels 이상인 것만)
    defect_type_samples = {}  # {원본_값: (file_name, mask, pixel_count)}
    defect_type_candidates = {}  # {원본_값: [(file_name, mask, pixel_count), ...]}
    
    print(f"각 결함 유형별 이미지 검색 중... (최소 픽셀 수: {min_pixels}개 이상)")
    
    # 먼저 모든 후보 수집
    for file_list in client_identifier_dict.values():
        for file_name in file_list:
            npy_path = annotations_path / f"{file_name}.npy"
            if not npy_path.exists():
                continue
            
            mask = np.load(str(npy_path))
            unique_values = np.unique(mask)
            
            # 각 결함 유형별로 픽셀 수 확인
            for defect_value in unique_values:
                if defect_value in label_mapping:
                    pixel_count = np.sum(mask == defect_value)
                    
                    # 최소 픽셀 수 이상인 경우만 후보에 추가
                    if pixel_count >= min_pixels:
                        if defect_value not in defect_type_candidates:
                            defect_type_candidates[defect_value] = []
                        defect_type_candidates[defect_value].append((file_name, mask, pixel_count))
    
    # 각 결함 유형별로 픽셀 수가 가장 많은 이미지 선택
    for defect_value, candidates in defect_type_candidates.items():
        if len(candidates) > 0:
            # 픽셀 수가 많은 순서로 정렬하여 가장 많은 것 선택
            candidates_sorted = sorted(candidates, key=lambda x: x[2], reverse=True)
            file_name, mask, pixel_count = candidates_sorted[0]
            defect_type_samples[defect_value] = (file_name, mask, pixel_count)
            print(f"  발견: 결함 유형 {defect_value} -> {file_name} (픽셀 수: {pixel_count})")
    
    # 최소 픽셀 수 미만으로 인해 찾지 못한 결함 유형 확인
    missing_types = set(label_mapping.keys()) - set(defect_type_samples.keys())
    if missing_types:
        print(f"\n경고: 다음 결함 유형은 최소 픽셀 수({min_pixels}개) 이상인 이미지를 찾지 못했습니다:")
        for defect_value in sorted(missing_types):
            if defect_value in defect_type_candidates:
                max_pixels = max([c[2] for c in defect_type_candidates[defect_value]])
                print(f"  - 결함 유형 {defect_value}: 최대 픽셀 수 {max_pixels}개 (요구: {min_pixels}개)")
            else:
                print(f"  - 결함 유형 {defect_value}: 해당 유형을 포함하는 이미지 없음")
    
    print(f"\n총 {len(defect_type_samples)}개 결함 유형의 샘플 이미지 발견")
    
    if len(defect_type_samples) == 0:
        print("시각화할 샘플 이미지가 없습니다.")
        return
    
    # 그리드 크기 계산
    num_samples = len(defect_type_samples)
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 정렬된 결함 유형 순서로 시각화
    sorted_defect_types = sorted(defect_type_samples.keys())
    
    for idx, defect_value in enumerate(sorted_defect_types):
        file_name, mask, pixel_count_original = defect_type_samples[defect_value]
        
        # 이미지 로드
        img0_path = image_path0 / f"{file_name}.jpg"
        img1_path = image_path1 / f"{file_name}.jpg"
        
        try:
            img0 = Image.open(img0_path)
        except Exception as e:
            print(f"  경고: {file_name} 이미지 로드 실패 - {e}")
            axes[idx].text(0.5, 0.5, f'로드 실패\n{file_name}', 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].axis('off')
            continue
        
        # 이미지를 NumPy 배열로 변환
        img0_array = np.array(img0)
        
        # 그레이스케일이면 RGB로 변환
        if len(img0_array.shape) == 2:
            img0_rgb = np.stack([img0_array] * 3, axis=-1)
        elif len(img0_array.shape) == 3 and img0_array.shape[2] == 1:
            img0_rgb = np.repeat(img0_array, 3, axis=2)
        else:
            img0_rgb = img0_array
        
        # 이미지 정규화
        if img0_rgb.max() > 255:
            img0_rgb = (img0_rgb / img0_rgb.max() * 255).astype(np.uint8)
        else:
            img0_rgb = img0_rgb.astype(np.uint8)
        
        # 마스크 크기 조정 (이미지 크기에 맞춤)
        if mask.shape != img0_rgb.shape[:2]:
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = mask_pil.resize((img0_rgb.shape[1], img0_rgb.shape[0]), Image.NEAREST)
            mask_resized = np.array(mask_pil)
        else:
            mask_resized = mask
        
        # 이미지에 마스크 오버레이
        overlay_img = img0_rgb.copy().astype(float)
        
        # 해당 결함 유형의 색상 가져오기 (없으면 기본 색상 사용)
        if defect_value in defect_color_map:
            color = defect_color_map[defect_value]
        else:
            # 해시 기반 색상 생성
            color = tuple((hash(defect_value) % 200 + 55) for _ in range(3))
        
        # 해당 결함 유형 영역만 마스킹
        mask_region = (mask_resized == defect_value).astype(float)
        
        if np.sum(mask_region) > 0:
            # 알파 블렌딩 (0.6 = 60% 색상, 40% 원본 이미지)
            alpha = 0.6
            for c in range(3):
                overlay_img[:, :, c] = (
                    overlay_img[:, :, c] * (1 - mask_region * alpha) + 
                    color[c] * mask_region * alpha
                )
        
        overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)
        
        # 시각화
        axes[idx].imshow(overlay_img)
        
        # 제목 설정
        label_idx = label_mapping.get(defect_value, -1)
        title = f'결함 유형: {defect_value} (레이블 인덱스: {label_idx})\n'
        title += f'파일명: {file_name}\n'
        
        # 해당 결함 유형의 픽셀 수 계산 (리사이즈 후)
        pixel_count = np.sum(mask_resized == defect_value)
        total_pixels = mask_resized.size
        ratio = (pixel_count / total_pixels) * 100
        title += f'픽셀 수: {pixel_count} ({ratio:.2f}%)'
        
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')
    
    # 빈 subplot 숨기기
    for idx in range(len(defect_type_samples), len(axes)):
        axes[idx].axis('off')
    
    # 범례 추가
    legend_elements = []
    for defect_value in sorted_defect_types:
        if defect_value in defect_color_map:
            color = defect_color_map[defect_value]
        else:
            color = tuple((hash(defect_value) % 200 + 55) for _ in range(3))
        
        label_idx = label_mapping.get(defect_value, -1)
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=np.array(color)/255, 
                        edgecolor='black', label=f'{defect_value} (인덱스: {label_idx})')
        )
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=min(4, len(legend_elements)), 
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.suptitle(f'각 결함 유형별 샘플 이미지 ({len(defect_type_samples)}개, 최소 {min_pixels}픽셀 이상)', fontsize=14, y=0.995)
    plt.show()