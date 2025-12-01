import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18DefectClassifier(nn.Module):
    """
    ResNet-18 기반 결함 유형 분류 모델
    ImageNet 사전 학습 가중치 사용
    """
    def __init__(self, num_classes, input_channels=2, pretrained=True, freeze_backbone=False):
        super(ResNet18DefectClassifier, self).__init__()
        
        # ResNet-18 모델 로드
        resnet = models.resnet18(pretrained=pretrained)
        
        # 입력 채널이 2인 경우 첫 번째 Conv 레이어 수정
        if input_channels != 3:
            original_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            with torch.no_grad():
                if pretrained:
                    new_conv.weight.data = original_conv.weight.data[:, :input_channels, :, :].mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            resnet.conv1 = new_conv
        
        # Backbone 고정
        if freeze_backbone:
            for param in resnet.parameters():
                param.requires_grad = False
        
        # ResNet의 마지막 FC 레이어를 제거하고 새로운 분류 레이어 추가
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 FC 제거
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    def unfreeze_backbone(self):
        """Backbone의 모든 레이어를 학습 가능하도록 설정"""
        for param in self.backbone.parameters():
            param.requires_grad = True


def initialize_defect_classifier(num_classes, input_channels=2, device=None, 
                                pretrained=True, freeze_backbone=False):
    """
    결함 유형 분류 모델 초기화
    
    Args:
        num_classes: 결함 유형 클래스 수 (0, 1 제외)
        input_channels: 입력 채널 수 (기본값: 2)
        device: PyTorch device
        pretrained: ImageNet 사전 학습 가중치 사용 여부
        freeze_backbone: Backbone 고정 여부
    
    Returns:
        초기화된 ResNet-18 분류 모델
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model = ResNet18DefectClassifier(
        num_classes=num_classes,
        input_channels=input_channels,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    return model

def load_defect_classifier(model_path, num_classes=None, device=None):
    """
    저장된 결함 분류 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        num_classes: 클래스 수 (None이면 체크포인트에서 추론)
        device: PyTorch device
    
    Returns:
        로드된 모델
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # num_classes 추출
    if num_classes is None:
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        else:
            raise ValueError("num_classes를 확인할 수 없습니다. 명시적으로 지정해주세요.")
    
    # 모델 초기화
    model = ResNet18DefectClassifier(num_classes=num_classes, input_channels=2, pretrained=False)
    
    # 가중치 로드
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model

def save_patch_cnn_model(model, model_path, num_classes=None):
    """
    결함 분류 모델 저장
    
    Args:
        model: 저장할 모델
        model_path: 저장 경로
        num_classes: 클래스 수 (None이면 모델에서 자동 추출)
    """
    if num_classes is None:
        # 모델의 FC 레이어에서 클래스 수 추출
        num_classes = model.fc.out_features
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes
    }
    torch.save(checkpoint, model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")