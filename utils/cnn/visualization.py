import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'font.size': 12})

import torch
from utils.cnn.dataset_functions import unwrap_client_data


def plot_training_curves(lossDict, accuracyDict, testLoss, testAccuracy, clientIDs):
    """
    학습 곡선 시각화
    
    Args:
        lossDict: 클라이언트별 손실 딕셔너리
        accuracyDict: 클라이언트별 정확도 딕셔너리
        testLoss: 테스트 손실 리스트
        testAccuracy: 테스트 정확도 리스트
        clientIDs: 클라이언트 ID 리스트
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 클라이언트별 손실
    axes[0, 0].set_title('Client Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    for clientID in clientIDs:
        if len(lossDict[clientID]) > 0:
            axes[0, 0].plot(lossDict[clientID], label=clientID, alpha=0.7)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 클라이언트별 정확도
    axes[0, 1].set_title('Client Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    for clientID in clientIDs:
        if len(accuracyDict[clientID]) > 0:
            axes[0, 1].plot(accuracyDict[clientID], label=clientID, alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 테스트 손실
    axes[1, 0].set_title('Test Loss')
    axes[1, 0].set_xlabel('Server Round')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].plot(testLoss, marker='o', linewidth=2, markersize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 테스트 정확도
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Server Round')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].plot(testAccuracy, marker='o', linewidth=2, markersize=8, color='green')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(model, imageDictTest, labelDictTest, testClients,
                         clientIdentifierDict, data_dir, num_samples=6, device=None):
    """
    테스트 세트에 대한 예측 결과 시각화
    
    Args:
        model: 학습된 CNN 모델
        imageDictTest: 테스트 이미지 딕셔너리
        labelDictTest: 테스트 레이블 딕셔너리
        testClients: 테스트 클라이언트 리스트
        clientIdentifierDict: 클라이언트별 파일 리스트 딕셔너리
        data_dir: data 디렉터리 경로
        num_samples: 시각화할 샘플 개수
        device: PyTorch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    model = model.to(device)
    
    from pathlib import Path
    from PIL import Image
    
    data_path = Path(data_dir)
    image_path0 = data_path / '0'
    
    testImages, testLabels = unwrap_client_data(imageDictTest, labelDictTest, testClients)
    testImages = testImages.to(device)
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(testImages)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probabilities, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
    
    # 샘플 선택 (정확히 예측된 것과 잘못 예측된 것 모두)
    correct_indices = (predicted.cpu() == testLabels).nonzero(as_tuple=True)[0]
    incorrect_indices = (predicted.cpu() != testLabels).nonzero(as_tuple=True)[0]
    
    # 샘플 선택
    samples = []
    if len(correct_indices) > 0:
        samples.extend(correct_indices[:num_samples//2].tolist())
    if len(incorrect_indices) > 0:
        samples.extend(incorrect_indices[:num_samples//2].tolist())
    
    if len(samples) < num_samples:
        # 부족하면 랜덤 선택
        remaining = num_samples - len(samples)
        all_indices = list(range(len(testImages)))
        remaining_indices = [i for i in all_indices if i not in samples]
        import random
        samples.extend(random.sample(remaining_indices, min(remaining, len(remaining_indices))))
    
    samples = samples[:num_samples]
    
    # 파일명 가져오기
    file_names = []
    curr_idx = 0
    for clientID in testClients:
        for file_name in clientIdentifierDict[clientID]:
            if curr_idx in samples:
                file_names.append((curr_idx, file_name))
            curr_idx += 1
    
    # 시각화
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if len(samples) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (sample_idx, file_name) in enumerate(file_names):
        # 원본 이미지 로드
        img0_path = image_path0 / f"{file_name}.jpg"
        img = Image.open(img0_path)
        
        true_label = testLabels[sample_idx].item()
        pred_label = predicted[sample_idx].item()
        confidence = confidences[sample_idx].item()
        is_correct = true_label == pred_label
        
        axes[idx].imshow(img, cmap='gray')
        title = f'{file_name}\n'
        title += f'True: {true_label}, Pred: {pred_label}\n'
        title += f'Conf: {confidence:.3f}'
        if is_correct:
            title += ' ✓'
        else:
            title += ' ✗'
        axes[idx].set_title(title, color='green' if is_correct else 'red')
        axes[idx].axis('off')
    
    # 빈 subplot 숨기기
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()