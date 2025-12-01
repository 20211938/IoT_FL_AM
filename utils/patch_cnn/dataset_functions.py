import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.stats import dirichlet



def load_image_for_patch(image_path0, image_path1, target_size=None):
    """
    패치 생성을 위한 이미지 로드
    
    Args:
        image_path0: Post Spreading 이미지 경로
        image_path1: Post Fusion 이미지 경로
        target_size: 목표 크기 (None이면 원본 크기 유지)
    
    Returns:
        이미지 배열 (H, W, 2)
    """
    im0 = Image.open(image_path0)
    im1 = Image.open(image_path1)
    
    if target_size:
        im0 = im0.resize((target_size[1], target_size[0]), Image.LANCZOS)
        im1 = im1.resize((target_size[1], target_size[0]), Image.LANCZOS)
    
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


def get_albumentations_transform(is_training=True):
    """
    Albumentations를 사용한 데이터 증강 파이프라인 (2채널 이미지용)
    """
    if is_training:
        transform = A.Compose([
            # 기본 기하학적 변환만
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 약한 픽셀 레벨 증강
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # 0.3에서 0.2로 감소
                contrast_limit=0.2,    # 0.3에서 0.2로 감소
                p=0.5  # 0.7에서 0.5로 감소
            ),
            
            # 정규화
            A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
            ToTensorV2(transpose_mask=False)
        ])
    else:
        # 검증/테스트용: 정규화만
        transform = A.Compose([
            A.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),
            ToTensorV2(transpose_mask=False)
        ])
    
    return transform


def get_defect_type_for_patch(patch_mask, ignore_labels=[0, 1]):
    """
    패치 마스크에서 결함 유형 결정 (가장 많이 나타나는 결함 유형)
    
    Args:
        patch_mask: 패치 마스크 배열 (H, W)
        ignore_labels: 무시할 레이블 리스트 (정상 부분)
    
    Returns:
        defect_type: 결함 유형 (None이면 결함 없음)
        confidence: 신뢰도 (해당 결함 유형의 비율)
    """
    # 결함 픽셀만 필터링
    defect_pixels = patch_mask[~np.isin(patch_mask, ignore_labels)]
    
    if len(defect_pixels) == 0:
        return None, 0.0
    
    # 가장 많이 나타나는 결함 유형
    unique_values, counts = np.unique(defect_pixels, return_counts=True)
    most_common_idx = np.argmax(counts)
    defect_type = int(unique_values[most_common_idx])
    confidence = float(counts[most_common_idx] / len(defect_pixels))
    
    return defect_type, confidence


class DefectPatchDataset(Dataset):
    """
    결함 패치 분류용 데이터셋 (단일 레이블)
    """
    def __init__(self, patches, labels, transform=None):
        """
        Args:
            patches: 패치 리스트 (numpy 배열)
            labels: 레이블 리스트 (정수, 각 패치의 결함 유형)
            transform: Albumentations transform
        """
        self.patches = patches
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx].copy()
        label = self.labels[idx]
        
        # Albumentations 적용
        if self.transform:
            patch_uint8 = (patch * 255).astype(np.uint8)
            transformed = self.transform(image=patch_uint8)
            patch_tensor = transformed['image']
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        
        return patch_tensor, torch.tensor(label, dtype=torch.long)


def create_defect_only_patch_dataset(
    client_identifier_dict, 
    data_dir, 
    patch_size=(512, 512), 
    patch_overlap=0,
    min_defect_ratio=0.1,
    min_confidence=0.2,
    label_mapping=None, 
    transform=None
):
    """
    결함 픽셀만 포함하는 패치 데이터셋 생성 (결함 유형 분류용)
    
    Args:
        client_identifier_dict: 클라이언트별 파일 리스트 딕셔너리
        data_dir: 데이터 디렉터리 경로
        patch_size: 패치 크기 (height, width)
        patch_overlap: 패치 간 겹침 비율 (현재 미사용)
        min_defect_ratio: 패치 내 최소 결함 픽셀 비율
        min_confidence: 최소 신뢰도
        label_mapping: 레이블 매핑 딕셔너리 (원본 값 -> 인덱스)
        transform: Albumentations transform
    
    Returns:
        imageDict: 클라이언트별 이미지 텐서 딕셔너리
        labelDict: 클라이언트별 레이블 텐서 딕셔너리 (각 패치의 결함 유형)
        metadataDict: 클라이언트별 메타데이터 딕셔너리 (파일명, 위치 정보)
    """
    imageDict = {}
    labelDict = {}
    metadataDict = {}  # 메타데이터 추가
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    annotations_path = data_path / 'annotations'
    
    # 첫 번째 패스: 전체 데이터셋에서 각 결함 유형의 픽셀 비율 계산
    print("전체 데이터셋에서 결함 유형별 픽셀 비율 계산 중...")
    defect_type_pixel_counts = {}  # 각 결함 유형별 픽셀 수
    total_defect_pixels = 0  # 전체 결함 픽셀 수
    
    for clientID, file_list in client_identifier_dict.items():
        for file_name in file_list:
            npy_path = annotations_path / f"{file_name}.npy"
            if not npy_path.exists():
                continue
            
            mask = np.load(str(npy_path))
            
            # 각 결함 유형별 픽셀 수 계산 (0, 1 제외)
            unique_values, counts = np.unique(mask, return_counts=True)
            for val, count in zip(unique_values, counts):
                if val not in [0, 1]:  # 정상 부분 제외
                    if val not in defect_type_pixel_counts:
                        defect_type_pixel_counts[val] = 0
                    defect_type_pixel_counts[val] += count
                    total_defect_pixels += count
    
    # 레이블 매핑 적용하여 비율 계산
    defect_type_ratios = {}
    if label_mapping:
        for defect_type, pixel_count in defect_type_pixel_counts.items():
            mapped_label = label_mapping.get(defect_type, -1)
            if mapped_label != -1:
                if mapped_label not in defect_type_ratios:
                    defect_type_ratios[mapped_label] = 0
                defect_type_ratios[mapped_label] += pixel_count
    else:
        defect_type_ratios = defect_type_pixel_counts.copy()
    
    # 비율 계산
    if total_defect_pixels > 0:
        for label in defect_type_ratios:
            defect_type_ratios[label] = defect_type_ratios[label] / total_defect_pixels
    else:
        defect_type_ratios = {label: 0.0 for label in defect_type_ratios}
    
    print(f"결함 유형별 픽셀 비율: {defect_type_ratios}")
    
    # 두 번째 패스: 각 클라이언트별로 패치 생성 및 증강
    for clientID, file_list in client_identifier_dict.items():
        print(f"{clientID}...")
        
        all_patches = []
        all_labels = []  # 각 패치의 결함 유형 레이블 (단일 값)
        all_metadata = []  # 메타데이터 추가
        
        # 패치 통계 추적
        total_patches_processed = 0
        excluded_patches = 0
        
        for file_idx, file_name in enumerate(file_list):
            img0_path = image_path0 / f"{file_name}.jpg"
            img1_path = image_path1 / f"{file_name}.jpg"
            npy_path = annotations_path / f"{file_name}.npy"
            
            if not (img0_path.exists() and img1_path.exists() and npy_path.exists()):
                continue
            
            # 이미지 로드
            image = load_image_for_patch(str(img0_path), str(img1_path), target_size=None)
            
            # 마스크 로드
            mask = np.load(str(npy_path))
            
            # 이미지와 마스크 크기 확인
            h, w = image.shape[:2]
            mask_h, mask_w = mask.shape[:2]
            
            if h != mask_h or w != mask_w:
                print(f"  경고: {file_name} - 이미지 크기 ({h}, {w})와 마스크 크기 ({mask_h}, {mask_w})가 다릅니다.")
                from scipy.ndimage import zoom
                zoom_factors = (h / mask_h, w / mask_w)
                mask = zoom(mask, zoom_factors, order=0)
            
            # 이미지를 패치로 나누기
            patches, positions, patch_sizes = create_patches_adaptive_grid(image, patch_size=patch_size)
            
            patches_before_filter = len(patches)
            total_patches_processed += patches_before_filter
            
            # 각 패치에 대해 처리
            for patch, (y_start, x_start), (patch_h, patch_w) in zip(patches, positions, patch_sizes):
                # 패치 마스크 추출
                y_end = min(y_start + patch_h, mask.shape[0])
                x_end = min(x_start + patch_w, mask.shape[1])
                patch_mask = mask[y_start:y_end, x_start:x_end]
                
                # 패치 이미지를 고정 크기로 리사이즈
                from scipy.ndimage import zoom
                target_h, target_w = patch_size[0], patch_size[1]
                zoom_h = target_h / patch.shape[0]
                zoom_w = target_w / patch.shape[1]
                patch_resized = zoom(patch, (zoom_h, zoom_w, 1.0), order=1)
                
                # 패치 마스크도 동일한 크기로 리사이즈
                patch_mask_resized = zoom(patch_mask, (zoom_h, zoom_w), order=0)
                patch_mask_resized = patch_mask_resized.astype(patch_mask.dtype)
                
                # 결함 유형 결정
                defect_type, confidence = get_defect_type_for_patch(patch_mask_resized, ignore_labels=[0, 1])
                
                if defect_type is not None and confidence >= min_confidence:
                    # 레이블 매핑 적용
                    if label_mapping:
                        mapped_label = label_mapping.get(defect_type, -1)
                        if mapped_label == -1:
                            excluded_patches += 1
                            continue
                    else:
                        mapped_label = defect_type
                    
                    all_patches.append(patch_resized)
                    all_labels.append(mapped_label)
                    all_metadata.append({
                        'file_name': file_name,
                        'y_start': y_start,
                        'y_end': y_end,
                        'x_start': x_start,
                        'x_end': x_end,
                        'patch_mask': patch_mask_resized.copy(),
                        'defect_type': mapped_label  # 결함 유형 저장
                    })
                else:
                    excluded_patches += 1
            
            if (file_idx + 1) % 20 == 0:
                print(f"  처리 중: {file_idx + 1}/{len(file_list)} 파일 완료 (패치: {len(all_patches)}개)")
        
        excluded_patches = total_patches_processed - len(all_patches)
        
        if len(all_patches) == 0:
            print(f"{clientID}: 총 {total_patches_processed}개 패치 중 {excluded_patches}개 제외됨 (사용 가능한 패치 없음)")
            continue
        else:
            print(f"{clientID}: 총 {total_patches_processed}개 패치 중 {excluded_patches}개 제외됨, {len(all_patches)}개 사용")
        
        # 데이터셋 생성
        dataset = DefectPatchDataset(all_patches, all_labels, transform)
        
        # 증강 적용 - 결함 유형별 픽셀 비율에 따라 다른 augmentation_factor 사용
        images = []
        labels = []
        metadata_list = []
        
        # 각 패치별로 결함 유형에 따라 다른 augmentation_factor 적용
        for i in range(len(dataset)):
            defect_type = all_metadata[i]['defect_type']
            defect_type_ratio = defect_type_ratios.get(defect_type, 0.0)
            augmentation_factor = get_augmentation_factor_by_defect_type_ratio(defect_type_ratio)
            
            for aug_idx in range(augmentation_factor):
                img, label = dataset[i]
                images.append(img)
                labels.append(all_labels[i])
                metadata_list.append(all_metadata[i])
        
        imageDict[clientID] = torch.stack(images)
        labelDict[clientID] = torch.tensor(labels, dtype=torch.long)
        metadataDict[clientID] = metadata_list
        
        # 증강 통계 출력
        aug_factors_by_type = {}
        for i, defect_type in enumerate([meta['defect_type'] for meta in all_metadata]):
            defect_type_ratio = defect_type_ratios.get(defect_type, 0.0)
            aug_factor = get_augmentation_factor_by_defect_type_ratio(defect_type_ratio)
            if defect_type not in aug_factors_by_type:
                aug_factors_by_type[defect_type] = aug_factor
        
        # 증강 이후 결함 유형 분포 및 비율 계산
        label_distribution = dict(Counter(labels))
        total_samples = len(labels)
        label_ratios = {label: count / total_samples * 100 for label, count in label_distribution.items()}
        sorted_ratios = sorted(label_ratios.items(), key=lambda x: x[1], reverse=True)
        
        print(f"원본 파일: {len(file_list)}개")
        print(f"생성된 결함 패치: {len(all_patches)}개")
        print(f"증강 적용 후: {len(images)}개")
        print(f"결함 유형별 augmentation_factor: {aug_factors_by_type}")
        print(f"결함 유형 비율 (증강 이후):")
        for label, ratio in sorted_ratios:
            count = label_distribution[label]
            print(f"  결함 {label}: {count}개 ({ratio:.2f}%)")
    
    return imageDict, labelDict, metadataDict

def create_patches_adaptive_grid(image_array, patch_size=(256, 256)):
    """
    patch_size에 따라 그리드 크기를 자동으로 조정하여 패치 생성
    작은 patch_size일수록 더 많은 패치 생성
    """
    h, w = image_array.shape[:2]
    
    # patch_size에 따라 그리드 크기 결정
    # 예: 512x512 -> 3x3, 256x256 -> 4x4, 128x128 -> 5x5
    if patch_size[0] >= 512:
        grid_size = 3
    elif patch_size[0] >= 256:
        grid_size = 4
    elif patch_size[0] >= 128:
        grid_size = 5
    else:
        grid_size = 6
    
    patch_h_base = h // grid_size
    patch_w_base = w // grid_size
    
    patches = []
    positions = []
    patch_sizes = []
    
    for row in range(grid_size):
        for col in range(grid_size):
            y_start = row * patch_h_base
            x_start = col * patch_w_base
            
            if row == grid_size - 1:
                y_end = h
            else:
                y_end = y_start + patch_h_base
            
            if col == grid_size - 1:
                x_end = w
            else:
                x_end = x_start + patch_w_base
            
            patch = image_array[y_start:y_end, x_start:x_end, :]
            patches.append(patch)
            positions.append((y_start, x_start))
            patch_sizes.append((patch.shape[0], patch.shape[1]))
    
    return patches, positions, patch_sizes

def get_label_mapping(data_dir, client_identifier_dict):
    """
    데이터에서 모든 레이블 유형 매핑 생성 (0과 1 포함, 모든 결함 포함)
    Semantic Segmentation을 위해 모든 레이블 값에 대해 매핑 생성
    """
    data_path = Path(data_dir)
    annotations_path = data_path / 'annotations'
    
    all_label_types = set()
    
    for file_list in client_identifier_dict.values():
        for file_name in file_list:
            npy_path = annotations_path / f"{file_name}.npy"
            if npy_path.exists():
                mask = np.load(str(npy_path))
                unique_values = np.unique(mask)
                # 모든 레이블 유형 포함 (0과 1도 포함)
                all_label_types.update(unique_values)
    
    sorted_types = sorted(all_label_types)
    label_mapping = {label_type: idx for idx, label_type in enumerate(sorted_types)}
    num_classes = len(sorted_types)
    
    print(f"발견된 모든 레이블 유형: {sorted_types}")
    print(f"레이블 매핑: {label_mapping}")
    print(f"총 클래스 수: {num_classes}")
    
    return label_mapping, num_classes

def calculate_defect_ratio(patch_mask, ignore_labels=[0, 1]):
    """
    패치 마스크에서 결함 픽셀 비율 계산
    
    Args:
        patch_mask: 패치 마스크 배열 (H, W)
        ignore_labels: 무시할 레이블 리스트 (정상 부분)
    
    Returns:
        defect_ratio: 결함 픽셀 비율 (0.0 ~ 1.0)
    """
    total_pixels = patch_mask.size
    defect_pixels = np.sum(~np.isin(patch_mask, ignore_labels))
    defect_ratio = defect_pixels / total_pixels if total_pixels > 0 else 0.0
    return defect_ratio


def get_augmentation_factor_by_defect_type_ratio(defect_type_ratio,
                                                 very_high_threshold=0.20,    # 20% 이상
                                                 high_threshold=0.15,         # 15% 이상
                                                 mid_high_threshold=0.10,     # 10% 이상
                                                 mid_threshold=0.05,          # 5% 이상
                                                 low_threshold=0.03,          # 3% 이상
                                                 very_high_aug_factor=2,      # 매우 높은 비율: 1배
                                                 high_aug_factor=4,           # 높은 비율: 2배
                                                 mid_high_aug_factor=6,       # 중간-높은 비율: 4배
                                                 mid_aug_factor=8,           # 중간 비율: 6배
                                                 low_aug_factor=10,           # 낮은 비율: 8배
                                                 very_low_aug_factor=12):    # 매우 낮은 비율: 10배
    """
    전체 데이터셋에서 계산된 결함 유형별 픽셀 비율에 따라 augmentation_factor 결정
    1배부터 10배까지 증강 적용
    
    Args:
        defect_type_ratio: 해당 결함 유형의 전체 픽셀 비율 (0.0 ~ 1.0)
        very_high_threshold: 매우 높은 비율 임계값 (20% 이상)
        high_threshold: 높은 비율 임계값 (15% 이상)
        mid_high_threshold: 중간-높은 비율 임계값 (10% 이상)
        mid_threshold: 중간 비율 임계값 (5% 이상)
        low_threshold: 낮은 비율 임계값 (3% 이상)
        very_high_aug_factor: 매우 높은 비율일 때 augmentation_factor (1배)
        high_aug_factor: 높은 비율일 때 augmentation_factor (2배)
        mid_high_aug_factor: 중간-높은 비율일 때 augmentation_factor (4배)
        mid_aug_factor: 중간 비율일 때 augmentation_factor (6배)
        low_aug_factor: 낮은 비율일 때 augmentation_factor (8배)
        very_low_aug_factor: 매우 낮은 비율일 때 augmentation_factor (10배)
    
    Returns:
        augmentation_factor: 증강 배수 (1~10)
    """
    if defect_type_ratio >= very_high_threshold:
        return very_high_aug_factor      # 20% 이상: 1배
    elif defect_type_ratio >= high_threshold:
        return high_aug_factor           # 15-20%: 2배
    elif defect_type_ratio >= mid_high_threshold:
        return mid_high_aug_factor       # 10-15%: 4배
    elif defect_type_ratio >= mid_threshold:
        return mid_aug_factor            # 5-10%: 6배
    elif defect_type_ratio >= low_threshold:
        return low_aug_factor            # 3-5%: 8배
    else:
        return very_low_aug_factor       # 3% 미만: 10배


def analyze_file_defect_distribution(data_dir, all_files):
    """
    각 파일의 결함 유형 분포 분석
    
    Args:
        data_dir: 데이터 디렉터리 경로
        all_files: 분석할 파일 리스트
    
    Returns:
        file_defect_distribution: {file_name: {defect_type: pixel_count, ...}, ...}
    """
    data_path = Path(data_dir)
    annotations_path = data_path / 'annotations'
    
    file_defect_distribution = {}
    
    for file_name in all_files:
        npy_path = annotations_path / f"{file_name}.npy"
        if not npy_path.exists():
            continue
        
        mask = np.load(str(npy_path))
        unique_values, counts = np.unique(mask, return_counts=True)
        
        defect_dist = {}
        for val, count in zip(unique_values, counts):
            if val not in [0, 1]:  # 정상 부분 제외
                defect_dist[int(val)] = count
        
        file_defect_distribution[file_name] = defect_dist
    
    return file_defect_distribution


def distribute_files_non_iid_by_defect_type(all_files, num_clients, data_dir, 
                                             label_mapping=None, bias_strength=0.8):
    """
    결함 종류 기반 Non-IID 데이터 분배
    각 클라이언트에 특정 결함 유형이 편향되도록 파일 할당
    
    Args:
        all_files: 전체 파일 리스트
        num_clients: 클라이언트 수
        data_dir: 데이터 디렉터리
        label_mapping: 레이블 매핑 (None이면 원본 레이블 사용)
        bias_strength: 편향 강도 (0.0~1.0, 높을수록 더 편향)
    
    Returns:
        clientIdentifierDict: {client_id: [file_list], ...}
    """
    import random
    
    # 파일별 결함 분포 분석
    print("파일별 결함 분포 분석 중...")
    file_defect_dist = analyze_file_defect_distribution(data_dir, all_files)
    
    # 모든 결함 유형 수집
    all_defect_types = set()
    for dist in file_defect_dist.values():
        all_defect_types.update(dist.keys())
    all_defect_types = sorted(list(all_defect_types))
    
    if label_mapping:
        # 레이블 매핑 적용
        mapped_defect_types = sorted([label_mapping.get(dt, dt) for dt in all_defect_types 
                                      if label_mapping.get(dt, -1) != -1])
    else:
        mapped_defect_types = all_defect_types
    
    num_defect_types = len(mapped_defect_types)
    
    if num_defect_types == 0:
        print("경고: 결함 유형을 찾을 수 없습니다. 랜덤 분배를 사용합니다.")
        random.shuffle(all_files)
        files_per_client = len(all_files) // num_clients
        return {f'client{i+1}': all_files[i*files_per_client:(i+1)*files_per_client] 
                for i in range(num_clients)}
    
    # 각 클라이언트에 주요 결함 유형 할당 (순환 방식)
    # 예: client1 → 결함0, client2 → 결함1, client3 → 결함2, ...
    client_primary_defect = {}
    for i in range(num_clients):
        client_id = f'client{i+1}'
        # 결함 유형을 순환하여 할당
        primary_defect = mapped_defect_types[i % num_defect_types]
        client_primary_defect[client_id] = primary_defect
        print(f"  {client_id}의 주요 결함 유형: {primary_defect}")
    
    # 각 파일을 클라이언트에 할당
    client_files = {f'client{i+1}': [] for i in range(num_clients)}
    
    # 파일을 주요 결함 유형별로 그룹화
    files_by_defect = {defect_type: [] for defect_type in mapped_defect_types}
    
    for file_name in all_files:
        if file_name not in file_defect_dist:
            # 결함 정보가 없으면 랜덤 할당
            client_id = f'client{random.randint(1, num_clients)}'
            client_files[client_id].append(file_name)
            continue
        
        file_dist = file_defect_dist[file_name]
        total_pixels = sum(file_dist.values())
        
        if total_pixels == 0:
            # 결함이 없으면 랜덤 할당
            client_id = f'client{random.randint(1, num_clients)}'
            client_files[client_id].append(file_name)
            continue
        
        # 파일에서 가장 많은 결함 유형 찾기
        main_defect_type = max(file_dist.items(), key=lambda x: x[1])[0]
        
        # 레이블 매핑 적용
        if label_mapping:
            main_defect_type = label_mapping.get(main_defect_type, -1)
            if main_defect_type == -1:
                client_id = f'client{random.randint(1, num_clients)}'
                client_files[client_id].append(file_name)
                continue
        
        if main_defect_type in mapped_defect_types:
            files_by_defect[main_defect_type].append(file_name)
        else:
            # 알 수 없는 결함 유형이면 랜덤 할당
            client_id = f'client{random.randint(1, num_clients)}'
            client_files[client_id].append(file_name)
    
    # 각 결함 유형의 파일들을 해당 클라이언트에 할당
    for defect_type, files in files_by_defect.items():
        # 이 결함 유형을 주요 결함으로 가진 클라이언트 찾기
        target_clients = [cid for cid, primary in client_primary_defect.items() 
                         if primary == defect_type]
        
        if not target_clients:
            # 해당 결함 유형을 주요 결함으로 가진 클라이언트가 없으면 랜덤 할당
            for file_name in files:
                client_id = f'client{random.randint(1, num_clients)}'
                client_files[client_id].append(file_name)
            continue
        
        # 파일을 해당 클라이언트들에 분배
        random.shuffle(files)
        files_per_target = len(files) // len(target_clients)
        
        for idx, target_client in enumerate(target_clients):
            start_idx = idx * files_per_target
            if idx == len(target_clients) - 1:
                end_idx = len(files)
            else:
                end_idx = (idx + 1) * files_per_target
            
            # 편향 강도에 따라 일부 파일을 다른 클라이언트에도 할당 가능
            assigned_files = files[start_idx:end_idx]
            for file_name in assigned_files:
                if random.random() < bias_strength:
                    # 편향 강도에 따라 주요 클라이언트에 할당
                    client_files[target_client].append(file_name)
                else:
                    # 일부는 랜덤 클라이언트에 할당 (IID 성분)
                    client_id = f'client{random.randint(1, num_clients)}'
                    client_files[client_id].append(file_name)
    
    # 최소 파일 수 보장
    min_files = min(len(files) for files in client_files.values())
    if min_files == 0:
        print("경고: 일부 클라이언트에 파일이 할당되지 않았습니다.")
        for client_id, files in client_files.items():
            if len(files) == 0:
                max_client = max(client_files.keys(), key=lambda k: len(client_files[k]))
                if len(client_files[max_client]) > 1:
                    client_files[client_id].append(client_files[max_client].pop())
    
    print(f"\nNon-IID 분배 완료 (편향 강도: {bias_strength}):")
    for client_id, files in client_files.items():
        primary_defect = client_primary_defect.get(client_id, "N/A")
        print(f"  {client_id}: {len(files)}개 파일 (주요 결함: {primary_defect})")
    
    return client_files



def create_all_patches_with_augmentation(
    all_files,
    data_dir,
    patch_size=(512, 512),
    patch_overlap=0,
    min_defect_ratio=0.1,
    min_confidence=0.2,
    label_mapping=None,
    transform=None
):
    """
    전체 파일에서 패치 생성 및 증강 (클라이언트 분배 전)
    
    Args:
        all_files: 전체 파일 리스트
        data_dir: 데이터 디렉터리 경로
        patch_size: 패치 크기 (height, width)
        patch_overlap: 패치 간 겹침 비율
        min_defect_ratio: 패치 내 최소 결함 픽셀 비율
        min_confidence: 최소 신뢰도
        label_mapping: 레이블 매핑 딕셔너리
        transform: Albumentations transform
    
    Returns:
        all_patches: 모든 패치 리스트 (numpy 배열)
        all_labels: 모든 레이블 리스트
        all_metadata: 모든 메타데이터 리스트
        defect_type_ratios: 결함 유형별 패치 비율 (패치 개수 기반)
    """
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    annotations_path = data_path / 'annotations'
    
    # 패치 생성
    print("전체 파일에서 패치 생성 중...")
    all_patches = []
    all_labels = []
    all_metadata = []
    
    total_patches_processed = 0
    excluded_patches = 0
    
    for file_idx, file_name in enumerate(all_files):
        img0_path = image_path0 / f"{file_name}.jpg"
        img1_path = image_path1 / f"{file_name}.jpg"
        npy_path = annotations_path / f"{file_name}.npy"
        
        if not (img0_path.exists() and img1_path.exists() and npy_path.exists()):
            continue
        
        # 이미지 로드
        image = load_image_for_patch(str(img0_path), str(img1_path), target_size=None)
        
        # 마스크 로드
        mask = np.load(str(npy_path))
        
        # 이미지와 마스크 크기 확인
        h, w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if h != mask_h or w != mask_w:
            from scipy.ndimage import zoom
            zoom_factors = (h / mask_h, w / mask_w)
            mask = zoom(mask, zoom_factors, order=0)
        
        # 이미지를 패치로 나누기
        patches, positions, patch_sizes = create_patches_adaptive_grid(image, patch_size=patch_size)
        
        patches_before_filter = len(patches)
        total_patches_processed += patches_before_filter
        
        # 각 패치에 대해 처리
        for patch, (y_start, x_start), (patch_h, patch_w) in zip(patches, positions, patch_sizes):
            # 패치 마스크 추출
            y_end = min(y_start + patch_h, mask.shape[0])
            x_end = min(x_start + patch_w, mask.shape[1])
            patch_mask = mask[y_start:y_end, x_start:x_end]
            
            # 패치 이미지를 고정 크기로 리사이즈
            from scipy.ndimage import zoom
            target_h, target_w = patch_size[0], patch_size[1]
            zoom_h = target_h / patch.shape[0]
            zoom_w = target_w / patch.shape[1]
            patch_resized = zoom(patch, (zoom_h, zoom_w, 1.0), order=1)
            
            # 패치 마스크도 동일한 크기로 리사이즈
            patch_mask_resized = zoom(patch_mask, (zoom_h, zoom_w), order=0)
            patch_mask_resized = patch_mask_resized.astype(patch_mask.dtype)
            
            # 결함 유형 결정
            defect_type, confidence = get_defect_type_for_patch(patch_mask_resized, ignore_labels=[0, 1])
            
            if defect_type is not None and confidence >= min_confidence:
                # 레이블 매핑 적용
                if label_mapping:
                    mapped_label = label_mapping.get(defect_type, -1)
                    if mapped_label == -1:
                        excluded_patches += 1
                        continue
                else:
                    mapped_label = defect_type
                
                all_patches.append(patch_resized)
                all_labels.append(mapped_label)
                all_metadata.append({
                    'file_name': file_name,
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'patch_mask': patch_mask_resized.copy(),
                    'defect_type': mapped_label
                })
            else:
                excluded_patches += 1
        
        if (file_idx + 1) % 20 == 0:
            print(f"  처리 중: {file_idx + 1}/{len(all_files)} 파일 완료 (패치: {len(all_patches)}개)")
    
    excluded_patches = total_patches_processed - len(all_patches)
    print(f"총 {total_patches_processed}개 패치 중 {excluded_patches}개 제외됨, {len(all_patches)}개 사용")
    
    # 패치 생성 후 결함 유형별 패치 개수 비율 계산
    print("\n패치 생성 후 결함 유형별 패치 개수 비율 계산 중...")
    patch_count_by_type = dict(Counter(all_labels))
    total_patches = len(all_labels)
    
    defect_type_ratios = {}
    if total_patches > 0:
        for defect_type, count in patch_count_by_type.items():
            defect_type_ratios[defect_type] = count / total_patches
    else:
        defect_type_ratios = {}
    
    print(f"결함 유형별 패치 비율: {defect_type_ratios}")
    print("\n결함 유형별 패치 개수:")
    for defect_type in sorted(patch_count_by_type.keys()):
        count = patch_count_by_type[defect_type]
        ratio = defect_type_ratios.get(defect_type, 0.0) * 100
        print(f"  결함 {defect_type}: {count}개 ({ratio:.2f}%)")
    
    # 증강 적용
    print("\n증강 적용 중...")
    dataset = DefectPatchDataset(all_patches, all_labels, transform)
    
    augmented_patches = []
    augmented_labels = []
    augmented_metadata = []
    
    # 각 패치별로 결함 유형에 따라 다른 augmentation_factor 적용
    for i in range(len(dataset)):
        defect_type = all_metadata[i]['defect_type']
        defect_type_ratio = defect_type_ratios.get(defect_type, 0.0)
        augmentation_factor = get_augmentation_factor_by_defect_type_ratio(defect_type_ratio)
        
        for aug_idx in range(augmentation_factor):
            img, label = dataset[i]
            augmented_patches.append(img)
            augmented_labels.append(all_labels[i])
            augmented_metadata.append(all_metadata[i])
    
    print(f"\n원본 패치: {len(all_patches)}개")
    print(f"증강 적용 후: {len(augmented_patches)}개")
    
    # 증강 전 결함 유형 분포 계산
    pre_aug_label_distribution = dict(Counter(all_labels))
    pre_aug_total = len(all_labels)
    pre_aug_ratios = {label: count / pre_aug_total * 100 for label, count in pre_aug_label_distribution.items()}
    
    # 증강 이후 결함 유형 분포 계산
    post_aug_label_distribution = dict(Counter(augmented_labels))
    post_aug_total = len(augmented_labels)
    post_aug_ratios = {label: count / post_aug_total * 100 for label, count in post_aug_label_distribution.items()}
    
    # 모든 결함 유형 수집 (정렬)
    all_defect_types = sorted(set(list(pre_aug_label_distribution.keys()) + list(post_aug_label_distribution.keys())))
    
    # 증강 배수 계산 (각 결함 유형별)
    aug_factors = {}
    for defect_type in all_defect_types:
        pre_count = pre_aug_label_distribution.get(defect_type, 0)
        post_count = post_aug_label_distribution.get(defect_type, 0)
        if pre_count > 0:
            aug_factors[defect_type] = post_count / pre_count
        else:
            aug_factors[defect_type] = 0
    
    # 보기 좋게 출력
    print("\n" + "=" * 90)
    print("증강 전후 결함 유형별 비교")
    print("=" * 90)
    print(f"{'결함 유형':<12} {'증강 전':<20} {'증강 후':<20} {'증강 배수':<12} {'비율 변화':<15}")
    print("-" * 90)
    
    # 비율 순으로 정렬 (증강 후 비율 기준)
    sorted_defects = sorted(all_defect_types, key=lambda x: post_aug_ratios.get(x, 0), reverse=True)
    
    for defect_type in sorted_defects:
        pre_count = pre_aug_label_distribution.get(defect_type, 0)
        pre_ratio = pre_aug_ratios.get(defect_type, 0.0)
        post_count = post_aug_label_distribution.get(defect_type, 0)
        post_ratio = post_aug_ratios.get(defect_type, 0.0)
        aug_factor = aug_factors.get(defect_type, 0.0)
        ratio_change = post_ratio - pre_ratio
        
        # 증강 배수 포맷팅
        if aug_factor > 0:
            aug_factor_str = f"{aug_factor:.2f}x"
        else:
            aug_factor_str = "N/A"
        
        # 비율 변화 포맷팅 (부호 포함)
        if ratio_change > 0:
            ratio_change_str = f"+{ratio_change:.2f}%"
        else:
            ratio_change_str = f"{ratio_change:.2f}%"
        
        print(f"결함 {defect_type:<8} {pre_count:>6}개 ({pre_ratio:>5.2f}%)  {post_count:>6}개 ({post_ratio:>5.2f}%)  {aug_factor_str:>10}  {ratio_change_str:>12}")
    
    print("=" * 90)
    print(f"총계:        {pre_aug_total:>6}개 (100.00%)  {post_aug_total:>6}개 (100.00%)")
    print("=" * 90)
    
    return augmented_patches, augmented_labels, augmented_metadata, defect_type_ratios


def distribute_patches_non_iid_by_defect_type(
    all_patches,
    all_labels,
    all_metadata,
    num_clients,
    bias_strength=0.8
):
    """
    증강된 패치를 결함 유형 기반 Non-IID로 클라이언트에 분배
    
    Args:
        all_patches: 모든 패치 리스트 (torch.Tensor 또는 numpy 배열)
        all_labels: 모든 레이블 리스트
        all_metadata: 모든 메타데이터 리스트
        num_clients: 클라이언트 수
        bias_strength: 편향 강도 (0.0~1.0)
    
    Returns:
        imageDict: 클라이언트별 이미지 텐서 딕셔너리
        labelDict: 클라이언트별 레이블 텐서 딕셔너리
        metadataDict: 클라이언트별 메타데이터 딕셔너리
    """
    import random
    
    # 패치를 결함 유형별로 그룹화
    patches_by_defect = {}
    for i, label in enumerate(all_labels):
        if label not in patches_by_defect:
            patches_by_defect[label] = []
        patches_by_defect[label].append(i)
    
    # 모든 결함 유형 수집
    all_defect_types = sorted(list(patches_by_defect.keys()))
    num_defect_types = len(all_defect_types)
    
    if num_defect_types == 0:
        print("경고: 결함 유형을 찾을 수 없습니다. 랜덤 분배를 사용합니다.")
        indices = list(range(len(all_patches)))
        random.shuffle(indices)
        patches_per_client = len(indices) // num_clients
        client_indices = {f'client{i+1}': indices[i*patches_per_client:(i+1)*patches_per_client] 
                         for i in range(num_clients)}
    else:
        # 각 클라이언트에 주요 결함 유형 할당 (순환 방식)
        client_primary_defect = {}
        for i in range(num_clients):
            client_id = f'client{i+1}'
            primary_defect = all_defect_types[i % num_defect_types]
            client_primary_defect[client_id] = primary_defect
            print(f"  {client_id}의 주요 결함 유형: {primary_defect}")
        
        # 각 클라이언트에 패치 인덱스 할당
        client_indices = {f'client{i+1}': [] for i in range(num_clients)}
        
        # 각 결함 유형의 패치들을 해당 클라이언트에 할당
        for defect_type, patch_indices in patches_by_defect.items():
            # 이 결함 유형을 주요 결함으로 가진 클라이언트 찾기
            target_clients = [cid for cid, primary in client_primary_defect.items() 
                             if primary == defect_type]
            
            if not target_clients:
                # 해당 결함 유형을 주요 결함으로 가진 클라이언트가 없으면 랜덤 할당
                for idx in patch_indices:
                    client_id = f'client{random.randint(1, num_clients)}'
                    client_indices[client_id].append(idx)
                continue
            
            # 패치를 해당 클라이언트들에 분배
            random.shuffle(patch_indices)
            patches_per_target = len(patch_indices) // len(target_clients)
            
            for idx, target_client in enumerate(target_clients):
                start_idx = idx * patches_per_target
                if idx == len(target_clients) - 1:
                    end_idx = len(patch_indices)
                else:
                    end_idx = (idx + 1) * patches_per_target
                
                # 편향 강도에 따라 일부 패치를 다른 클라이언트에도 할당 가능
                assigned_indices = patch_indices[start_idx:end_idx]
                for patch_idx in assigned_indices:
                    if random.random() < bias_strength:
                        # 편향 강도에 따라 주요 클라이언트에 할당
                        client_indices[target_client].append(patch_idx)
                    else:
                        # 일부는 랜덤 클라이언트에 할당 (IID 성분)
                        client_id = f'client{random.randint(1, num_clients)}'
                        client_indices[client_id].append(patch_idx)
    
    # 최소 패치 수 보장
    min_patches = min(len(indices) for indices in client_indices.values())
    if min_patches == 0:
        print("경고: 일부 클라이언트에 패치가 할당되지 않았습니다.")
        for client_id, indices in client_indices.items():
            if len(indices) == 0:
                max_client = max(client_indices.keys(), key=lambda k: len(client_indices[k]))
                if len(client_indices[max_client]) > 1:
                    client_indices[client_id].append(client_indices[max_client].pop())
    
    # 클라이언트별 데이터 구성
    imageDict = {}
    labelDict = {}
    metadataDict = {}
    
    # all_patches가 torch.Tensor인지 확인
    is_tensor = isinstance(all_patches[0], torch.Tensor)
    
    for client_id, indices in client_indices.items():
        if len(indices) == 0:
            continue
        
        client_patches = [all_patches[i] for i in indices]
        client_labels = [all_labels[i] for i in indices]
        client_metadata = [all_metadata[i] for i in indices]
        
        if is_tensor:
            imageDict[client_id] = torch.stack(client_patches)
        else:
            imageDict[client_id] = torch.stack([torch.from_numpy(p).permute(2, 0, 1).float() if isinstance(p, np.ndarray) else p for p in client_patches])
        
        labelDict[client_id] = torch.tensor(client_labels, dtype=torch.long)
        metadataDict[client_id] = client_metadata
        
        # 클라이언트별 통계
        label_distribution = dict(Counter(client_labels))
        total_samples = len(client_labels)
        label_ratios = {label: count / total_samples * 100 for label, count in label_distribution.items()}
        sorted_ratios = sorted(label_ratios.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{client_id}: {len(indices)}개 패치")
        print(f"  결함 유형 분포:")
        for label, ratio in sorted_ratios[:5]:  # 상위 5개만 출력
            count = label_distribution[label]
            print(f"    결함 {label}: {count}개 ({ratio:.2f}%)")
    
    print(f"\nNon-IID 분배 완료 (편향 강도: {bias_strength}):")
    for client_id, indices in client_indices.items():
        primary_defect = client_primary_defect.get(client_id, "N/A") if num_defect_types > 0 else "N/A"
        print(f"  {client_id}: {len(indices)}개 패치 (주요 결함: {primary_defect})")
    
    return imageDict, labelDict, metadataDict