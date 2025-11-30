import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from collections import Counter
import albumentations as A
from albumentations.pytorch import ToTensorV2



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
    
    for clientID, file_list in client_identifier_dict.items():
        print(f"{clientID}...")
        
        all_patches = []
        all_labels = []  # 각 패치의 결함 유형 레이블 (단일 값)
        all_metadata = []  # 메타데이터 추가: (file_name, y_start, y_end, x_start, x_end, patch_mask)
        
        # 패치 통계 추적
        total_patches_processed = 0  # 처리된 총 패치 수
        excluded_patches = 0  # 제외된 패치 수
        
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
                # 마스크를 이미지 크기에 맞게 리사이즈
                from scipy.ndimage import zoom
                zoom_factors = (h / mask_h, w / mask_w)
                mask = zoom(mask, zoom_factors, order=0)  # order=0: nearest neighbor
            
            # 이미지를 9등분(3x3 그리드)으로 나누기
            patches, positions, patch_sizes = create_patches_9grid(image)
            
            # 이 이미지에서 처리된 패치 수 (항상 9개)
            patches_before_filter = len(patches)
            total_patches_processed += patches_before_filter
            
            # 모든 패치를 동일한 크기로 맞추기 (가장 큰 패치 크기 사용)
            max_patch_h = max(ph for ph, pw in patch_sizes)
            max_patch_w = max(pw for ph, pw in patch_sizes)
            
            # 각 패치에 대해 처리
            for patch, (y_start, x_start), (patch_h, patch_w) in zip(patches, positions, patch_sizes):
                # 패치 마스크 추출
                y_end = min(y_start + patch_h, mask.shape[0])
                x_end = min(x_start + patch_w, mask.shape[1])
                patch_mask = mask[y_start:y_end, x_start:x_end]
                
                # 패치 이미지를 고정 크기로 리사이즈 (scipy.ndimage.zoom 사용)
                from scipy.ndimage import zoom
                target_h, target_w = patch_size[0], patch_size[1]
                zoom_h = target_h / patch.shape[0]
                zoom_w = target_w / patch.shape[1]
                # 2채널 이미지이므로 채널은 그대로 유지 (1.0)
                patch_resized = zoom(patch, (zoom_h, zoom_w, 1.0), order=1)  # order=1: bilinear interpolation
                
                # 패치 마스크도 동일한 크기로 리사이즈 (nearest neighbor로 레이블 보존)
                patch_mask_resized = zoom(patch_mask, (zoom_h, zoom_w), order=0)  # order=0: nearest neighbor
                patch_mask_resized = patch_mask_resized.astype(patch_mask.dtype)
                
                # 결함 유형 결정 (리사이즈된 마스크 사용)
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
                    
                    all_patches.append(patch_resized)  # 리사이즈된 패치 사용
                    all_labels.append(mapped_label)
                    all_metadata.append({
                        'file_name': file_name,
                        'y_start': y_start,
                        'y_end': y_end,
                        'x_start': x_start,
                        'x_end': x_end,
                        'patch_mask': patch_mask_resized.copy()
                    })
                else:
                    excluded_patches += 1
            
            if (file_idx + 1) % 20 == 0:
                print(f"  처리 중: {file_idx + 1}/{len(file_list)} 파일 완료 (패치: {len(all_patches)}개)")
        
        # 제외된 패치 수 계산 (더 정확하게)
        excluded_patches = total_patches_processed - len(all_patches)
        
        if len(all_patches) == 0:
            print(f"{clientID}: 총 {total_patches_processed}개 패치 중 {excluded_patches}개 제외됨 (사용 가능한 패치 없음)")
            continue
        else:
            print(f"{clientID}: 총 {total_patches_processed}개 패치 중 {excluded_patches}개 제외됨, {len(all_patches)}개 사용")
        
        # 데이터셋 생성
        dataset = DefectPatchDataset(all_patches, all_labels, transform)
        
        # 증강 적용
        augmentation_factor = 6
        images = []
        labels = []
        metadata_list = []  # 증강 후 메타데이터
        
        for aug_idx in range(augmentation_factor):
            for i in range(len(dataset)):
                img, label = dataset[i]
                images.append(img)
                labels.append(all_labels[i])  # 레이블은 증강과 무관하게 동일
                # 메타데이터도 증강과 함께 저장 (증강과 무관하게 동일)
                metadata_list.append(all_metadata[i])
        
        imageDict[clientID] = torch.stack(images)
        labelDict[clientID] = torch.tensor(labels, dtype=torch.long)
        metadataDict[clientID] = metadata_list  # 메타데이터 저장
        
        print(f"원본 파일: {len(file_list)}개")
        print(f"생성된 결함 패치: {len(all_patches)}개")
        print(f"증강 적용 후: {len(images)}개 (증강 배수: {augmentation_factor})")
        print(f"Image Tensor Shape: {imageDict[clientID].shape}")
        print(f"Label Tensor Shape: {labelDict[clientID].shape}")
        print(f"결함 유형 분포: {dict(Counter(labels))}")
    
    return imageDict, labelDict, metadataDict

def create_patches_9grid(image_array):
    """
    이미지를 항상 9등분(3x3 그리드)으로 분할
    이미지 크기와 상관없이 항상 9개의 패치 생성
    
    Args:
        image_array: 이미지 배열 (H, W, C)
    
    Returns:
        patches: 패치 리스트
        positions: 패치 위치 리스트 [(y_start, x_start), ...]
        patch_sizes: 각 패치의 실제 크기 리스트 [(h, w), ...]
    """
    h, w = image_array.shape[:2]
    
    # 각 패치의 기본 크기 계산
    patch_h_base = h // 3
    patch_w_base = w // 3
    
    patches = []
    positions = []
    patch_sizes = []
    
    # 3x3 그리드로 나누기
    for row in range(3):
        for col in range(3):
            # 각 패치의 시작 위치
            y_start = row * patch_h_base
            x_start = col * patch_w_base
            
            # 마지막 행/열인 경우 남은 부분 모두 포함
            if row == 2:
                y_end = h
            else:
                y_end = y_start + patch_h_base
            
            if col == 2:
                x_end = w
            else:
                x_end = x_start + patch_w_base
            
            # 패치 추출
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