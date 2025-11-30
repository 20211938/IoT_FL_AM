import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# utils/cnn/classifier.py의 CNNClassifier 클래스 수정 제안

import torch.nn as nn

class CNNClassifier(nn.Module):
    '''
    PyTorch 기반 CNN 모델 for defect type classification (GroupNorm 사용 버전)
    '''
    def __init__(self, num_classes, input_channels=2, num_groups=32):
        super(CNNClassifier, self).__init__()
        
        # 첫 번째 Conv 블록
        self.conv1_1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.gn1_1 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.gn1_2 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 두 번째 Conv 블록
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2_1 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.gn2_2 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 세 번째 Conv 블록
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3_1 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.gn3_2 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 네 번째 Conv 블록
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn4_1 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.gn4_2 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 다섯 번째 Conv 블록
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.gn5_1 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.gn5_2 = nn.GroupNorm(num_groups=min(32, num_groups), num_channels=512)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # 첫 번째 Conv 블록
        x = F.relu(self.gn1_1(self.conv1_1(x)))
        x = F.relu(self.gn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # 두 번째 Conv 블록
        x = F.relu(self.gn2_1(self.conv2_1(x)))
        x = F.relu(self.gn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # 세 번째 Conv 블록
        x = F.relu(self.gn3_1(self.conv3_1(x)))
        x = F.relu(self.gn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # 네 번째 Conv 블록
        x = F.relu(self.gn4_1(self.conv4_1(x)))
        x = F.relu(self.gn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        
        # 다섯 번째 Conv 블록
        x = F.relu(self.gn5_1(self.conv5_1(x)))
        x = F.relu(self.gn5_2(self.conv5_2(x)))
        x = self.pool5(x)
        
        # Global Average Pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# 나머지 함수들(initialize_cnn_classifier, load_cnn_model, save_cnn_model)은 동일


def initialize_cnn_classifier(num_classes, input_channels=2, device=None):
    '''
    Initialize and return a PyTorch CNN model for defect type classification
    
    Args:
        num_classes: Number of defect type classes
        input_channels: Number of input channels (default: 2 for Post Spreading + Post Fusion)
        device: Device to place model on ('cuda' or 'cpu', None for auto-detect)
    
    Returns:
        CNN model (PyTorch)
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model = CNNClassifier(num_classes, input_channels)
    model = model.to(device)
    
    return model


def load_cnn_model(model_path, num_classes=None, device=None):
    '''
    Load a trained PyTorch CNN model
    
    Args:
        model_path: Path to saved model (.pth or .pt file)
        num_classes: Number of classes (if None, will try to infer from model)
        device: Device to load model on ('cuda' or 'cpu', None for auto-detect)
    
    Returns:
        Loaded CNN model
    '''
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Checkpoint에서 num_classes 추출
    if num_classes is None:
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        elif 'model_state_dict' in checkpoint:
            # 모델 구조에서 추론
            last_layer_key = [k for k in checkpoint['model_state_dict'].keys() if 'fc3' in k][0]
            num_classes = checkpoint['model_state_dict'][last_layer_key].shape[0]
        else:
            raise ValueError("num_classes를 확인할 수 없습니다. 명시적으로 지정해주세요.")
    
    model = CNNClassifier(num_classes)
    
    # state_dict 가져오기
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # BatchNorm 관련 키 필터링 (현재 모델은 BatchNorm이 없음)
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('bn')}
    
    # 필터링된 state_dict 로드
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


def save_cnn_model(model, model_path, num_classes=None):
    '''
    Save a PyTorch CNN model
    
    Args:
        model: PyTorch model to save
        model_path: Path to save model
        num_classes: Number of classes (optional, for reference)
    '''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes if num_classes else getattr(model, 'num_classes', None)
    }
    torch.save(checkpoint, model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")