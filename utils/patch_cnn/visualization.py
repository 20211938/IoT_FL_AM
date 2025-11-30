import matplotlib.pyplot as plt
import numpy as np
import torch

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def plot_training_curves(lossDict, accuracyDict, testLoss, testAccuracy, clientIDs):
    """
    학습 곡선 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 클라이언트별 손실
    ax1 = axes[0, 0]
    for clientID in clientIDs:
        if clientID in lossDict:
            ax1.plot(lossDict[clientID], label=f'{clientID}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Client Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 클라이언트별 정확도
    ax2 = axes[0, 1]
    for clientID in clientIDs:
        if clientID in accuracyDict:
            ax2.plot(accuracyDict[clientID], label=f'{clientID}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Client Training Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # 테스트 손실
    ax3 = axes[1, 0]
    ax3.plot(testLoss, 'r-', linewidth=2, label='Test Loss')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Loss')
    ax3.set_title('Test Loss')
    ax3.legend()
    ax3.grid(True)
    
    # 테스트 정확도
    ax4 = axes[1, 1]
    ax4.plot(testAccuracy, 'b-', linewidth=2, label='Test Accuracy')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Test Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()


def visualize_test_results(
    model_path=None,
    data_dir='data_test/',
    total_files=78,
    patch_size=(512, 512),  # 호환성을 위해 유지하지만 사용되지 않음
    patch_overlap=0,
    output_dir='visualizations',
    device=None,
    subdivide_small_images=True  # 이제 사용되지 않음
):
    """
    저장된 모델로 테스트 데이터를 시각화
    이미지를 항상 9등분(3x3 그리드)으로 나누어 처리
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from PIL import Image
    from utils.patch_cnn.classifier import load_defect_classifier
    from utils.patch_cnn.dataset_functions import (
        load_image_for_patch, create_patches_9grid, 
        get_albumentations_transform, get_label_mapping
    )
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Device 설정
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 모델 경로 찾기
    if model_path is None:
        saved_model_dir = Path('saved_models')
        model_files = list(saved_model_dir.glob('Defect_Classifier_FL_*.pth'))
        if not model_files:
            raise FileNotFoundError("저장된 모델을 찾을 수 없습니다. saved_models 폴더를 확인하세요.")
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"자동으로 모델 찾음: {model_path}")
    else:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    print(f"모델 로드: {model_path}")
    
    # 데이터 경로 설정
    data_path = Path(data_dir)
    annotations_path = data_path / 'annotations'
    image_path0 = data_path / '0'
    image_path1 = data_path / '1'
    
    # 출력 디렉터리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 레이블 매핑 생성 (학습 시와 동일하게)
    all_files = [f'{i:06d}' for i in range(1, total_files + 1)]
    
    # 모든 레이블 유형 찾기
    all_label_types = set()
    for file_name in all_files:
        npy_path = annotations_path / f"{file_name}.npy"
        if npy_path.exists():
            mask = np.load(str(npy_path))
            unique_values = np.unique(mask)
            all_label_types.update(unique_values)
    
    sorted_types = sorted(all_label_types)
    all_label_mapping = {label_type: idx for idx, label_type in enumerate(sorted_types)}
    
    # 결함 유형만 포함하는 레이블 매핑 생성 (0, 1, -1 제외)
    defect_only_label_mapping = {}
    defect_classes = []
    for original_label, mapped_label in all_label_mapping.items():
        if original_label not in [0, 1, -1]:
            defect_only_label_mapping[original_label] = len(defect_classes)
            defect_classes.append(original_label)
    
    num_defect_classes = len(defect_classes)
    print(f"결함 유형 클래스 수: {num_defect_classes}")
    print(f"결함 유형 매핑: {defect_only_label_mapping}")
    print(f"결함 유형 원본 값: {defect_classes}")
    
    # 역매핑 생성 (인덱스 -> 원본 결함 값)
    reverse_mapping = {idx: defect_class for defect_class, idx in defect_only_label_mapping.items()}
    
    # 모델 로드
    model = load_defect_classifier(
        model_path=str(model_path),
        num_classes=num_defect_classes,
        device=device
    )
    model.eval()
    
    # 증강 없는 transform (테스트용)
    test_transform = get_albumentations_transform(is_training=False)
    
    # 각 이미지에 대해 시각화
    for file_idx, file_name in enumerate(all_files):
        img0_path = image_path0 / f"{file_name}.jpg"
        img1_path = image_path1 / f"{file_name}.jpg"
        npy_path = annotations_path / f"{file_name}.npy"
        
        if not (img0_path.exists() and img1_path.exists() and npy_path.exists()):
            print(f"파일 누락: {file_name}")
            continue
        
        print(f"\n처리 중: {file_name} ({file_idx + 1}/{total_files})")
        
        # 이미지 로드 (증강 없음)
        image = load_image_for_patch(str(img0_path), str(img1_path), target_size=None)
        mask = np.load(str(npy_path))
        
        # 이미지와 마스크 크기 확인
        h, w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        
        if h != mask_h or w != mask_w:
            print(f"  경고: 이미지 크기 ({h}, {w})와 마스크 크기 ({mask_h}, {mask_w})가 다릅니다. 마스크를 리사이즈합니다.")
            from scipy.ndimage import zoom
            zoom_factors = (h / mask_h, w / mask_w)
            mask = zoom(mask, zoom_factors, order=0)  # order=0: nearest neighbor
        
        # 원본 이미지 (첫 번째 채널만 사용하여 표시)
        original_image = image[:, :, 0]
        
        # 이미지를 9등분(3x3 그리드)으로 나누기
        patches, positions, patch_sizes = create_patches_9grid(image)
        
        # 모든 패치를 동일한 크기로 맞추기 (가장 큰 패치 크기 사용)
        max_patch_h = max(ph for ph, pw in patch_sizes)
        max_patch_w = max(pw for ph, pw in patch_sizes)
        
        # 각 패치에 대해 모델 예측
        patch_predictions = []
        patch_confidences = []
        
        with torch.no_grad():
            for patch, (y_start, x_start), (patch_h, patch_w) in zip(patches, positions, patch_sizes):
                # 패치를 동일한 크기로 패딩 (학습 시와 동일하게)
                padded_patch = np.zeros((max_patch_h, max_patch_w, patch.shape[2]), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
                
                # 패치를 텐서로 변환
                patch_uint8 = (patch * 255).astype(np.uint8)
                transformed = test_transform(image=patch_uint8)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
                
                # 모델 예측
                outputs = model(patch_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_class].item()
                
                patch_predictions.append(pred_class)
                patch_confidences.append(confidence)
        
        # 예측 결과를 전체 이미지 크기로 재구성
        prediction_map = np.zeros((h, w), dtype=np.int32) - 1  # -1은 예측 없음
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        for idx, ((y_start, x_start), pred_class, confidence, (patch_h, patch_w)) in enumerate(
            zip(positions, patch_predictions, patch_confidences, patch_sizes)
        ):
            # 실제 패치 크기 사용 (패딩 전 크기)
            y_end = min(y_start + patch_h, h)
            x_end = min(x_start + patch_w, w)
            
            # 전체 패치에 동일한 예측값 적용
            if pred_class < len(reverse_mapping):
                original_defect_value = reverse_mapping[pred_class]
                prediction_map[y_start:y_end, x_start:x_end] = original_defect_value
                confidence_map[y_start:y_end, x_start:x_end] = confidence
        
        # 실제 마스크에서 결함 유형 추출
        unique_defects = np.unique(mask)
        defect_types_in_image = [d for d in unique_defects if d not in [0, 1, -1]]
        defect_text = ", ".join([f"결함 {int(d)}" for d in defect_types_in_image]) if defect_types_in_image else "결함 없음"
        
        # 컬러맵 생성 (각 결함 유형에 다른 색상)
        colors = plt.cm.tab20(np.linspace(0, 1, len(defect_classes)))
        color_map = {defect_class: colors[i] for i, defect_class in enumerate(defect_classes)}
        
        # 하나의 파일에 원본 결함과 모델 예측 결과를 나란히 표시
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # 왼쪽: 원본 이미지 + 실제 결함 마스크
        mask_display = mask.copy()
        mask_colored = np.zeros((*mask_display.shape, 3))
        
        for defect_class in defect_classes:
            mask_colored[mask_display == defect_class] = color_map[defect_class][:3]
        
        axes[0].imshow(original_image, cmap='gray', alpha=0.5)
        axes[0].imshow(mask_colored, alpha=0.7)
        axes[0].set_title(f'원본 이미지 + 실제 결함 마스크\n{defect_text}', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 오른쪽: 원본 이미지 + 모델 예측 결과
        prediction_colored = np.zeros((*prediction_map.shape, 3))
        
        for defect_class in defect_classes:
            if defect_class in reverse_mapping.values():
                prediction_colored[prediction_map == defect_class] = color_map[defect_class][:3]
        
        axes[1].imshow(original_image, cmap='gray', alpha=0.5)
        axes[1].imshow(prediction_colored, alpha=0.7)
        
        # 예측된 결함 유형
        predicted_defects = np.unique(prediction_map[prediction_map >= 0])
        predicted_text = ", ".join([f"결함 {int(d)}" for d in predicted_defects if d in reverse_mapping.values()]) if len(predicted_defects) > 0 else "결함 없음"
        
        axes[1].set_title(f'원본 이미지 + 모델 예측 결과 (9등분)\n예측된 결함: {predicted_text}', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 범례 추가
        all_displayed_defects = set()
        if len(defect_types_in_image) > 0:
            all_displayed_defects.update(defect_types_in_image)
        if len(predicted_defects) > 0:
            all_displayed_defects.update([d for d in predicted_defects if d in reverse_mapping.values()])
        
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[dc][:3], 
                                         label=f'결함 {int(dc)}') for dc in defect_classes if dc in all_displayed_defects]
        if legend_elements:
            axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / f'visualization_{file_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  저장 완료: {output_path}/visualization_{file_name}.png")
    
    print(f"\n모든 시각화 완료! 결과는 {output_path} 폴더에 저장되었습니다.")