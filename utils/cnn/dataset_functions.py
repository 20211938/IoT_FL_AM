import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from collections import Counter
from utils.cnn.defect_detection import load_image_for_cnn, get_defect_type_from_npy


class DefectClassificationDataset(Dataset):
    """
    결함 유형 분류를 위한 PyTorch Dataset
    """
    def __init__(self, file_list, data_dir, target_size=(640, 640), label_mapping=None):
        """
        Args:
            file_list: 파일명 리스트 (확장자 제외)
            data_dir: data 디렉터리 경로
            target_size: 목표 이미지 크기 (height, width)
            label_mapping: 레이블 매핑 딕셔너리 (원본 값 -> 인덱스)
        """
        self.file_list = file_list
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.label_mapping = label_mapping
        
        self.image_path0 = self.data_dir / '0'
        self.image_path1 = self.data_dir / '1'
        self.annotations_path = self.data_dir / 'annotations'
        
        # 레이블 추출
        self.labels = []
        for file_name in file_list:
            npy_path = self.annotations_path / f"{file_name}.npy"
            defect_type, _ = get_defect_type_from_npy(str(npy_path))
            
            if defect_type is not None:
                if label_mapping:
                    label = label_mapping.get(defect_type, -1)
                else:
                    label = defect_type
            else:
                label = -1  # 결함이 없는 경우
            
            self.labels.append(label)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        label = self.labels[idx]
        
        img0_path = self.image_path0 / f"{file_name}.jpg"
        img1_path = self.image_path1 / f"{file_name}.jpg"
        
        # 이미지 로드 및 전처리
        image = load_image_for_cnn(str(img0_path), str(img1_path), self.target_size)
        
        # NumPy 배열을 PyTorch 텐서로 변환 (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


def create_cnn_dataset(client_identifier_dict, data_dir, target_size=(640, 640), label_mapping=None):
    """
    클라이언트별 CNN 데이터셋 생성
    
    Args:
        client_identifier_dict: 클라이언트별 파일 리스트 딕셔너리
        data_dir: data 디렉터리 경로
        target_size: 목표 이미지 크기
        label_mapping: 레이블 매핑 딕셔너리
    
    Returns:
        imageDict: 클라이언트별 이미지 텐서 딕셔너리
        labelDict: 클라이언트별 레이블 텐서 딕셔너리
    """
    imageDict = {}
    labelDict = {}
    
    for clientID, file_list in client_identifier_dict.items():
        print(f"{clientID}...")
        dataset = DefectClassificationDataset(file_list, data_dir, target_size, label_mapping)
        
        images = []
        labels = []
        
        for i in range(len(dataset)):
            img, label = dataset[i]
            # -1 레이블 필터링 (유효하지 않은 레이블 제거)
            if label.item() == -1:
                continue
            images.append(img)
            labels.append(label)
            
            if (i + 1) % 50 == 0:
                print(f"  처리 중: {i + 1}/{len(dataset)} 파일 완료")
        
        if len(images) == 0:
            print(f"경고: {clientID}에 유효한 데이터가 없습니다!")
            continue
            
        imageDict[clientID] = torch.stack(images)
        labelDict[clientID] = torch.stack(labels)
        
        print(f"Contains {len(file_list)} images...")
        print(f"Valid images: {len(images)} (filtered {len(file_list) - len(images)} invalid labels)")
        print(f"Image Tensor Shape: {imageDict[clientID].shape}")
        print(f"Label Shape: {labelDict[clientID].shape}")
        print(f"Label distribution: {Counter(labelDict[clientID].numpy())}")
    
    return imageDict, labelDict


def unwrap_client_data(imageDict, labelDict, clientIDs):
    """
    클라이언트 데이터를 하나의 텐서로 결합
    
    Args:
        imageDict: 클라이언트별 이미지 딕셔너리
        labelDict: 클라이언트별 레이블 딕셔너리
        clientIDs: 클라이언트 ID 리스트
    
    Returns:
        images: 결합된 이미지 텐서
        labels: 결합된 레이블 텐서
    """
    images = []
    labels = []
    
    for clientID in clientIDs:
        images.append(imageDict[clientID])
        labels.append(labelDict[clientID])
    
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return images, labels


# utils/cnn/dataset_functions.py의 get_label_mapping 함수 수정
def get_label_mapping(data_dir, client_identifier_dict):
    """
    데이터에서 결함 유형 레이블 매핑 생성
    npy 파일에 있는 모든 고유한 값을 수집합니다.
    
    Args:
        data_dir: data 디렉터리 경로
        client_identifier_dict: 클라이언트별 파일 리스트 딕셔너리
    
    Returns:
        label_mapping: 원본 값 -> 인덱스 매핑 딕셔너리
        num_classes: 클래스 개수
    """
    data_path = Path(data_dir)
    annotations_path = data_path / 'annotations'
    
    all_defect_types = set()
    
    # 모든 파일에서 결함 유형 수집 (모든 고유한 값)
    for file_list in client_identifier_dict.values():
        for file_name in file_list:
            npy_path = annotations_path / f"{file_name}.npy"
            if npy_path.exists():
                # npy 파일을 직접 로드하여 모든 고유한 값 수집
                mask = np.load(str(npy_path))
                unique_values = np.unique(mask)
                all_defect_types.update(unique_values)
    
    # 정렬하여 매핑 생성
    sorted_types = sorted(all_defect_types)
    label_mapping = {defect_type: idx for idx, defect_type in enumerate(sorted_types)}
    num_classes = len(sorted_types)
    
    print(f"발견된 결함 유형: {sorted_types}")
    print(f"레이블 매핑: {label_mapping}")
    print(f"총 클래스 수: {num_classes}")
    
    return label_mapping, num_classes