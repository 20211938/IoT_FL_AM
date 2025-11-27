"""
AprilGAN 기반 제로샷 결함 분류 모델
다운로드된 레이블된 이미지 데이터를 활용하여 전처리 없이 결함을 분류합니다.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class LabeledImageDataset(Dataset):
    """다운로드된 레이블된 이미지 데이터셋"""
    
    def __init__(self, data_dir: str, transform=None, verbose=True):
        """
        Args:
            data_dir: 다운로드된 이미지가 있는 디렉토리 (data/labeled_layers)
            transform: 이미지 변환 함수
            verbose: 진행 상황 출력 여부
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.verbose = verbose
        self.image_paths = []
        self.metadata = []
        
        if self.verbose:
            print(f"\n[데이터 로딩] 디렉토리 검색 중: {data_dir}")
        
        # 모든 하위 디렉토리에서 이미지와 JSON 파일 찾기
        db_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        if self.verbose:
            print(f"[데이터 로딩] 발견된 데이터베이스 디렉토리: {len(db_dirs)}개")
        
        for idx, db_dir in enumerate(db_dirs, 1):
            if self.verbose and idx % 10 == 0:
                print(f"[데이터 로딩] 처리 중: {idx}/{len(db_dirs)} 디렉토리...")
            
            img_count = 0
            for img_file in db_dir.glob("*.jpg"):
                json_file = img_file.with_suffix(".jpg.json")
                if json_file.exists():
                    self.image_paths.append(img_file)
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            self.metadata.append(json.load(f))
                        img_count += 1
                    except Exception as e:
                        if self.verbose:
                            print(f"[경고] JSON 파일 읽기 실패: {json_file} - {e}")
            
            if self.verbose and img_count > 0:
                print(f"  - {db_dir.name}: {img_count}개 이미지 발견")
        
        if self.verbose:
            print(f"[데이터 로딩] 완료! 총 {len(self.image_paths)}개 이미지 로드됨")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        metadata = self.metadata[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, metadata, str(self.image_paths[idx])


def custom_collate_fn(batch):
    """커스텀 collate 함수 - metadata 딕셔너리 처리"""
    images = torch.stack([item[0] for item in batch])
    metadata_list = [item[1] for item in batch]
    paths = [item[2] for item in batch]
    return images, metadata_list, paths


class AprilGANGenerator(nn.Module):
    """AprilGAN Generator 네트워크"""
    
    def __init__(self, latent_dim=100, img_channels=3):
        super(AprilGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        self.model = nn.Sequential(
            # 입력: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 64x64
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 128x128
        )
    
    def forward(self, z):
        return self.model(z)


class AprilGANDiscriminator(nn.Module):
    """AprilGAN Discriminator 네트워크"""
    
    def __init__(self, img_channels=3):
        super(AprilGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 입력: img_channels x 128 x 128
            nn.Conv2d(img_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 1x1
        )
    
    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)


class AprilGANZeroShotClassifier:
    """AprilGAN 기반 제로샷 결함 분류기"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.generator = None
        self.discriminator = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.defect_types = [
            "Keyhole", "Lack of Fusion", "Balling", "Crack",
            "Porosity", "Spatter", "Surface Roughness", "Normal"
        ]
        
        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 학습 히스토리
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'epoch': 0
        }
    
    def initialize_model(self, latent_dim=100, model_path=None, lr_g=0.0002, lr_d=0.0002):
        """AprilGAN 모델 초기화"""
        print(f"\n[모델 초기화] AprilGAN 모델 생성 중...")
        print(f"  - Device: {self.device}")
        print(f"  - Latent dimension: {latent_dim}")
        
        self.generator = AprilGANGenerator(latent_dim=latent_dim).to(self.device)
        self.discriminator = AprilGANDiscriminator().to(self.device)
        
        # 옵티마이저 초기화
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        # 사전 학습된 가중치가 있다면 로드
        if model_path and os.path.exists(model_path):
            print(f"  - 체크포인트 로드 중: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.generator.load_state_dict(checkpoint.get('generator_state_dict', checkpoint.get('generator', {})))
            self.discriminator.load_state_dict(checkpoint.get('discriminator_state_dict', checkpoint.get('discriminator', {})))
            
            # 옵티마이저 상태도 로드 (이어서 학습용)
            if 'generator_optimizer' in checkpoint:
                self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer'])
            if 'discriminator_optimizer' in checkpoint:
                self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
            
            # 학습 히스토리 로드
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                print(f"  - 이전 학습 이어서: Epoch {self.training_history['epoch']}")
            
            print("  - 체크포인트 로드 완료")
        else:
            print("  - 랜덤 초기화된 모델 사용 (사전 학습된 가중치 없음)")
            if model_path:
                print(f"  - 경고: 지정된 모델 경로를 찾을 수 없음: {model_path}")
        
        print("[모델 초기화] 완료!")
    
    def load_labeled_data(self, data_dir: str, verbose: bool = True) -> LabeledImageDataset:
        """다운로드된 레이블된 이미지 데이터 로드"""
        print(f"\n[데이터 로딩 시작]")
        print(f"  - 데이터 디렉토리: {data_dir}")
        dataset = LabeledImageDataset(data_dir, transform=self.transform, verbose=verbose)
        print(f"\n[데이터 로딩 완료] 총 {len(dataset)}개 이미지 로드됨")
        return dataset
    
    def extract_defect_info_from_metadata(self, metadata: Dict) -> List[str]:
        """JSON 메타데이터에서 결함 정보 추출"""
        defect_labels = []
        
        # TagBoxes에서 결함 정보 추출
        if 'DepositionImageModel' in metadata:
            tag_boxes = metadata['DepositionImageModel'].get('TagBoxes', [])
            for tag in tag_boxes:
                name = tag.get('Name', '')
                comment = tag.get('Comment', '')
                if name or comment:
                    defect_labels.append(f"{name}: {comment}")
        
        if 'ScanningImageModel' in metadata:
            tag_boxes = metadata['ScanningImageModel'].get('TagBoxes', [])
            for tag in tag_boxes:
                name = tag.get('Name', '')
                comment = tag.get('Comment', '')
                if name or comment:
                    defect_labels.append(f"{name}: {comment}")
        
        # IsDefected 정보
        if metadata.get('IsDefected', False):
            defect_labels.append("Defected")
        
        return defect_labels if defect_labels else ["Normal"]
    
    def classify_defects(self, image: Image.Image, threshold: float = 0.5, verbose: bool = False) -> Dict:
        """
        이미지에서 결함 분류 수행 (제로샷)
        
        Args:
            image: PIL Image
            threshold: 이상 탐지 임계값
            verbose: 상세 출력 여부
        
        Returns:
            분류 결과 딕셔너리
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("모델이 초기화되지 않았습니다. initialize_model()을 먼저 호출하세요.")
        
        if verbose:
            print(f"\n[결함 분류] 이미지 분석 중...")
            print(f"  - 이미지 크기: {image.size}")
            print(f"  - 이상 탐지 임계값: {threshold}")
        
        # 이미지 전처리
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Discriminator로 이상 점수 계산
        self.discriminator.eval()
        with torch.no_grad():
            anomaly_score = self.discriminator(img_tensor).item()
        
        if verbose:
            print(f"  - 이상 점수: {anomaly_score:.4f}")
        
        # 제로샷 분류: 각 결함 유형에 대한 텍스트 프롬프트와 유사도 계산
        # 실제 AprilGAN은 사전 학습된 특징 공간에서 이상 탐지 수행
        defect_scores = {}
        if verbose:
            print(f"  - 결함 유형별 점수 계산 중...")
        
        for defect_type in self.defect_types:
            # 간단한 휴리스틱: 실제로는 사전 학습된 특징 추출기 사용
            if defect_type == "Normal":
                score = 1.0 - anomaly_score
            else:
                # 각 결함 유형에 대한 점수 (실제로는 학습된 특징 공간에서 계산)
                score = anomaly_score * np.random.uniform(0.3, 0.9)  # 예시
        
            defect_scores[defect_type] = score
            if verbose:
                print(f"    - {defect_type}: {score:.4f}")
        
        # 가장 높은 점수의 결함 유형 선택
        predicted_defect = max(defect_scores, key=defect_scores.get)
        is_anomaly = anomaly_score > threshold
        
        if verbose:
            print(f"  - 예측된 결함: {predicted_defect}")
            print(f"  - 이상 탐지 결과: {'이상 발견' if is_anomaly else '정상'}")
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'predicted_defect': predicted_defect,
            'defect_scores': defect_scores
        }
    
    def evaluate_classification(self, dataset: LabeledImageDataset, 
                                batch_size: int = 32, max_samples: Optional[int] = None) -> Dict:
        """전체 데이터셋에 대한 분류 평가"""
        print(f"\n[평가 시작]")
        print(f"  - 전체 이미지 수: {len(dataset)}")
        print(f"  - 배치 크기: {batch_size}")
        if max_samples:
            print(f"  - 최대 평가 샘플 수: {max_samples}")
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=custom_collate_fn)
        
        results = []
        correct = 0
        total = 0
        anomaly_detected = 0
        
        print(f"\n[평가 진행] 결함 분류 수행 중...")
        for batch_idx, (images, metadata_list, paths) in enumerate(tqdm(dataloader, desc="배치 처리")):
            if max_samples and total >= max_samples:
                print(f"\n[평가 중단] 최대 샘플 수({max_samples})에 도달했습니다.")
                break
            
            for i in range(len(images)):
                if max_samples and total >= max_samples:
                    break
                    
                image = images[i]
                metadata = metadata_list[i]
                path = paths[i]
                
                # PIL Image로 변환
                img_pil = transforms.ToPILImage()(image)
                
                # 결함 분류
                classification = self.classify_defects(img_pil, verbose=False)
                
                # 실제 레이블 추출
                true_labels = self.extract_defect_info_from_metadata(metadata)
                
                results.append({
                    'path': path,
                    'predicted': classification['predicted_defect'],
                    'true_labels': true_labels,
                    'anomaly_score': classification['anomaly_score'],
                    'is_anomaly': classification['is_anomaly']
                })
                
                # 정확도 계산 (간단한 매칭)
                if any(label.lower() in classification['predicted_defect'].lower() 
                       or classification['predicted_defect'].lower() in label.lower() 
                       for label in true_labels):
                    correct += 1
                
                if classification['is_anomaly']:
                    anomaly_detected += 1
                
                total += 1
                
                # 진행 상황 출력 (100개마다)
                if total % 100 == 0:
                    current_acc = correct / total if total > 0 else 0
                    print(f"  - 처리된 샘플: {total}개 | 현재 정확도: {current_acc:.3f} | 이상 탐지: {anomaly_detected}개")
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n[평가 완료]")
        print(f"  - 총 평가 샘플: {total}개")
        print(f"  - 정확한 예측: {correct}개")
        print(f"  - 전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  - 이상 탐지된 샘플: {anomaly_detected}개 ({anomaly_detected/total*100:.2f}%)")
        
        return {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'anomaly_detected': anomaly_detected,
            'results': results
        }
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        print(f"  - 체크포인트 저장: {filepath}")
    
    def train(self, dataset: LabeledImageDataset, epochs: int = 10, batch_size: int = 32,
              checkpoint_dir: str = "checkpoints", save_every: int = 5, 
              resume_from: Optional[str] = None, g_train_ratio: int = 2, d_train_ratio: int = 1):
        """
        AprilGAN 모델 학습
        
        Args:
            dataset: 학습 데이터셋
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            checkpoint_dir: 체크포인트 저장 디렉토리
            save_every: 몇 에포크마다 저장할지
            resume_from: 이어서 학습할 체크포인트 경로
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("모델이 초기화되지 않았습니다. initialize_model()을 먼저 호출하세요.")
        
        print(f"\n[학습 시작]")
        print(f"  - 총 에포크: {epochs}")
        print(f"  - 배치 크기: {batch_size}")
        print(f"  - 데이터셋 크기: {len(dataset)}")
        print(f"  - 체크포인트 디렉토리: {checkpoint_dir}")
        
        # 체크포인트 디렉토리 생성
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 이어서 학습
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"  - 이어서 학습: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            start_epoch = checkpoint.get('epoch', 0) + 1
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
            print(f"  - Epoch {start_epoch}부터 학습 재개")
        
        # 데이터로더 생성 (커스텀 collate 함수 사용)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                               collate_fn=custom_collate_fn)
        
        # 손실 함수
        criterion = nn.BCELoss()
        
        # 정상 이미지 레이블 (1.0), 이상 이미지 레이블 (0.0)
        real_label = 1.0
        fake_label = 0.0
        
        print(f"\n[학습 진행]")
        for epoch in range(start_epoch, epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (real_images, metadata_list, paths) in enumerate(progress_bar):
                batch_size_actual = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # ============================================
                # Discriminator 학습 (비율에 따라 스킵 가능)
                # ============================================
                # 실제 이미지에 대한 판별 (정상 = 1.0)
                # Label smoothing 적용 (0.9~1.0) - Discriminator가 너무 확신하지 않도록
                real_labels = torch.full((batch_size_actual,), 0.9, 
                                        dtype=torch.float32, device=self.device)
                
                # 실제 이미지 판별
                output_real = self.discriminator(real_images)
                loss_d_real = criterion(output_real, real_labels)
                
                # 가짜 이미지 생성 및 판별 (이상 = 0.0)
                # Label smoothing 적용 (0.0~0.1)
                noise = torch.randn(batch_size_actual, self.generator.latent_dim, 1, 1, device=self.device)
                fake_images = self.generator(noise)
                fake_labels = torch.full((batch_size_actual,), 0.1, 
                                        dtype=torch.float32, device=self.device)
                
                output_fake = self.discriminator(fake_images.detach())
                loss_d_fake = criterion(output_fake, fake_labels)
                
                # Discriminator 총 손실
                loss_d = (loss_d_real + loss_d_fake) / 2.0
                
                if batch_idx % (g_train_ratio + d_train_ratio) < d_train_ratio:
                    # Discriminator 학습
                    self.discriminator_optimizer.zero_grad()
                    loss_d.backward()
                    self.discriminator_optimizer.step()
                else:
                    # Discriminator 학습 스킵 시 손실만 계산 (로깅용)
                    loss_d = loss_d.detach()  # gradient 계산 안 함
                
                # ============================================
                # Generator 학습 (비율에 따라 스킵 가능)
                # ============================================
                if batch_idx % (g_train_ratio + d_train_ratio) >= d_train_ratio:
                    self.generator_optimizer.zero_grad()
                
                    # Generator는 Discriminator를 속이려고 함 (가짜를 진짜처럼 = 0.9)
                    output_fake_gen = self.discriminator(fake_images)
                    loss_g = criterion(output_fake_gen, torch.full((batch_size_actual,), 0.9, 
                                                                  dtype=torch.float32, device=self.device))
                    loss_g.backward()
                    self.generator_optimizer.step()
                else:
                    # Generator 학습 스킵 시 손실만 계산 (로깅용)
                    output_fake_gen = self.discriminator(fake_images)
                    loss_g = criterion(output_fake_gen, torch.full((batch_size_actual,), 0.9, 
                                                                  dtype=torch.float32, device=self.device))
                    loss_g = loss_g.detach()  # gradient 계산 안 함
                
                # 손실 누적
                epoch_g_loss += loss_g.item()
                epoch_d_loss += loss_d.item()
                num_batches += 1
                
                # 진행 상황 업데이트
                progress_bar.set_postfix({
                    'G_Loss': f'{loss_g.item():.4f}',
                    'D_Loss': f'{loss_d.item():.4f}'
                })
            
            # 에포크별 평균 손실
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)
            self.training_history['epoch'] = epoch + 1
            
            print(f"\n[Epoch {epoch+1}/{epochs} 완료]")
            print(f"  - Generator Loss: {avg_g_loss:.4f}")
            print(f"  - Discriminator Loss: {avg_d_loss:.4f}")
            
            # 체크포인트 저장
            if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
                checkpoint_path = os.path.join(checkpoint_dir, f"aprilgan_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path, epoch + 1)
        
        print(f"\n[학습 완료]")
        print(f"  - 총 {epochs} 에포크 학습 완료")
        print(f"  - 최종 Generator Loss: {self.training_history['g_losses'][-1]:.4f}")
        print(f"  - 최종 Discriminator Loss: {self.training_history['d_losses'][-1]:.4f}")
    
    def visualize_defects(self, image: Image.Image, classification: Dict, 
                         save_path: Optional[str] = None):
        """결함 분류 결과 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 원본 이미지
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 분류 결과
        axes[1].text(0.1, 0.9, f"Anomaly Score: {classification['anomaly_score']:.3f}", 
                    transform=axes[1].transAxes, fontsize=12, verticalalignment='top')
        axes[1].text(0.1, 0.8, f"Predicted: {classification['predicted_defect']}", 
                    transform=axes[1].transAxes, fontsize=12, verticalalignment='top')
        axes[1].text(0.1, 0.7, f"Is Anomaly: {classification['is_anomaly']}", 
                    transform=axes[1].transAxes, fontsize=12, verticalalignment='top')
        
        # 결함 점수 표시
        y_pos = 0.6
        for defect_type, score in classification['defect_scores'].items():
            axes[1].text(0.1, y_pos, f"{defect_type}: {score:.3f}", 
                        transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
            y_pos -= 0.08
        
        axes[1].axis('off')
        axes[1].set_title('Classification Results')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """학습 히스토리 시각화"""
        if not self.training_history['g_losses']:
            print("학습 히스토리가 없습니다.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.training_history['g_losses']) + 1)
        
        ax.plot(epochs, self.training_history['g_losses'], 'b-', label='Generator Loss', linewidth=2)
        ax.plot(epochs, self.training_history['d_losses'], 'r-', label='Discriminator Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('AprilGAN Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("AprilGAN 기반 제로샷 결함 분류 모델")
    print("=" * 60)
    
    # 데이터 디렉토리 설정
    data_dir = to_str(LABELED_LAYERS_DIR)
    checkpoint_dir = to_str(CHECKPOINTS_DIR)
    
    # AprilGAN 분류기 초기화
    print("\n[1단계] 모델 초기화")
    classifier = AprilGANZeroShotClassifier()
    
    # 체크포인트에서 이어서 학습할지 확인
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(Path(checkpoint_dir).glob("aprilgan_epoch_*.pth"))
        if checkpoints:
            latest_checkpoint = str(checkpoints[-1])
            print(f"  - 발견된 최신 체크포인트: {latest_checkpoint}")
    
    # 모델 초기화 (체크포인트가 있으면 로드)
    # Learning Rate 조정: Generator는 높게, Discriminator는 낮게
    classifier.initialize_model(
        model_path=latest_checkpoint,
        lr_g=0.0003,  # Generator LR 증가 (더 빠른 학습)
        lr_d=0.0001   # Discriminator LR 감소 (과도한 성장 방지)
    )
    
    # 데이터 로드
    print("\n[2단계] 데이터 로딩")
    dataset = classifier.load_labeled_data(data_dir)
    
    if len(dataset) == 0:
        print("\n[오류] 로드된 이미지가 없습니다. 데이터 디렉토리를 확인하세요.")
        return
    
    # 모델 학습
    print("\n[3단계] 모델 학습")
    print("  학습을 시작합니다...")
    classifier.train(
        dataset=dataset,
        epochs=50,      # 에포크 증가 (GAN은 더 많은 학습 필요)
        batch_size=16,  # CPU에서는 적절한 크기
        checkpoint_dir=checkpoint_dir,
        save_every=10,  # 10 에포크마다 저장
        resume_from=latest_checkpoint if latest_checkpoint else None,
        g_train_ratio=3,  # Generator를 더 자주 학습 (초기 학습 안정화)
        d_train_ratio=1   # Discriminator 학습 비율
    )
    
    # 학습 히스토리 시각화
    print("\n[4단계] 학습 히스토리 시각화")
    try:
        classifier.plot_training_history()
        print("  - 학습 히스토리 시각화 완료")
    except Exception as e:
        print(f"  - 시각화 오류: {e}")
    
    # 샘플 이미지로 테스트
    print("\n[5단계] 샘플 이미지 테스트")
    sample_image, sample_metadata, sample_path = dataset[0]
    img_pil = transforms.ToPILImage()(sample_image)
    
    print(f"\n  - 샘플 이미지 경로: {sample_path}")
    print(f"  - 레이어 번호: {sample_metadata.get('LayerNum', 'N/A')}")
    print(f"  - 결함 여부: {sample_metadata.get('IsDefected', 'N/A')}")
    
    # 실제 레이블 정보
    true_labels = classifier.extract_defect_info_from_metadata(sample_metadata)
    print(f"  - 실제 레이블: {true_labels}")
    
    # 결함 분류
    print(f"\n[6단계] 결함 분류 수행")
    classification = classifier.classify_defects(img_pil, verbose=True)
    
    print(f"\n[분류 결과 요약]")
    print(f"  - 이상 탐지: {'✓ 예' if classification['is_anomaly'] else '✗ 아니오'}")
    print(f"  - 이상 점수: {classification['anomaly_score']:.4f}")
    print(f"  - 예측된 결함: {classification['predicted_defect']}")
    print(f"\n  [결함 유형별 점수]")
    for defect_type, score in sorted(classification['defect_scores'].items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"    - {defect_type:20s}: {score:.4f}")
    
    # 시각화
    print(f"\n[7단계] 결과 시각화")
    try:
        classifier.visualize_defects(img_pil, classification)
        print("  - 시각화 완료")
    except Exception as e:
        print(f"  - 시각화 오류: {e}")
    
    print("\n" + "=" * 60)
    print("실행 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

