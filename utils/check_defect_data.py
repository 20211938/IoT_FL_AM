"""
데이터셋에서 결함이 있는 샘플을 찾아서 확인하는 스크립트
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import sys

# utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.zero_shot_defect_classifier import AprilGANZeroShotClassifier


def extract_defect_info_from_metadata(metadata: dict) -> list:
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


def check_defect_data(data_dir="data/labeled_layers", max_samples=None):
    """
    데이터셋에서 결함 샘플을 찾아서 분석합니다.
    
    Args:
        data_dir: 데이터 디렉토리
        max_samples: 최대 검사할 샘플 수 (None이면 전체)
    """
    print("=" * 60)
    print("데이터셋 결함 샘플 확인")
    print("=" * 60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    print(f"\n[데이터 검색] 디렉토리: {data_dir}")
    
    # 통계 정보
    total_images = 0
    defected_images = 0
    normal_images = 0
    images_with_tags = 0
    
    # 결함 유형별 통계
    defect_types = defaultdict(int)
    defect_samples = []
    normal_samples = []
    
    # 모든 하위 디렉토리에서 이미지와 JSON 파일 찾기
    db_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"[발견된 데이터베이스 디렉토리] {len(db_dirs)}개\n")
    
    for db_dir in db_dirs:
        img_count = 0
        defected_count = 0
        
        for img_file in db_dir.glob("*.jpg"):
            json_file = img_file.with_suffix(".jpg.json")
            if not json_file.exists():
                continue
            
            total_images += 1
            img_count += 1
            
            if max_samples and total_images > max_samples:
                break
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # IsDefected 확인
                is_defected = metadata.get('IsDefected', False)
                
                # 결함 정보 추출
                defect_info = extract_defect_info_from_metadata(metadata)
                has_defect_tags = any('Normal' not in label for label in defect_info)
                
                if is_defected or has_defect_tags:
                    defected_images += 1
                    defected_count += 1
                    defect_samples.append({
                        'path': str(img_file),
                        'metadata': metadata,
                        'defect_info': defect_info
                    })
                    
                    # 결함 유형 카운트
                    for label in defect_info:
                        if 'Normal' not in label:
                            defect_types[label] += 1
                else:
                    normal_images += 1
                    normal_samples.append({
                        'path': str(img_file),
                        'metadata': metadata
                    })
                
                if has_defect_tags:
                    images_with_tags += 1
                    
            except Exception as e:
                print(f"[경고] JSON 파일 읽기 실패: {json_file} - {e}")
        
        if img_count > 0:
            print(f"  {db_dir.name}: {img_count}개 이미지 (결함: {defected_count}개)")
        
        if max_samples and total_images >= max_samples:
            break
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("[통계 요약]")
    print("=" * 60)
    print(f"  - 전체 이미지 수: {total_images}")
    print(f"  - 결함 이미지 수: {defected_images} ({defected_images/total_images*100:.2f}%)")
    print(f"  - 정상 이미지 수: {normal_images} ({normal_images/total_images*100:.2f}%)")
    print(f"  - 태그가 있는 이미지: {images_with_tags}개")
    
    if defect_types:
        print(f"\n[결함 유형별 통계]")
        for defect_type, count in sorted(defect_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {defect_type}: {count}개")
    
    # 결함 샘플 상세 정보
    if defect_samples:
        print(f"\n[결함 샘플 상세 정보] (최대 20개)")
        print("=" * 60)
        for idx, sample in enumerate(defect_samples[:20], 1):
            print(f"\n  샘플 {idx}: {Path(sample['path']).name}")
            print(f"    - 경로: {sample['path']}")
            print(f"    - IsDefected: {sample['metadata'].get('IsDefected', False)}")
            print(f"    - LayerNum: {sample['metadata'].get('LayerNum', 'N/A')}")
            print(f"    - 결함 정보: {sample['defect_info']}")
            
            # TagBoxes 정보
            if 'DepositionImageModel' in sample['metadata']:
                tag_boxes = sample['metadata']['DepositionImageModel'].get('TagBoxes', [])
                if tag_boxes:
                    print(f"    - DepositionImageModel TagBoxes: {len(tag_boxes)}개")
                    for tag in tag_boxes[:3]:  # 최대 3개만 출력
                        print(f"      - {tag.get('Name', '')}: {tag.get('Comment', '')}")
            
            if 'ScanningImageModel' in sample['metadata']:
                tag_boxes = sample['metadata']['ScanningImageModel'].get('TagBoxes', [])
                if tag_boxes:
                    print(f"    - ScanningImageModel TagBoxes: {len(tag_boxes)}개")
                    for tag in tag_boxes[:3]:  # 최대 3개만 출력
                        print(f"      - {tag.get('Name', '')}: {tag.get('Comment', '')}")
    else:
        print("\n[결과] 결함 샘플을 찾을 수 없습니다.")
        print("\n[정상 샘플 예시] (최대 5개)")
        for idx, sample in enumerate(normal_samples[:5], 1):
            print(f"\n  샘플 {idx}: {Path(sample['path']).name}")
            print(f"    - IsDefected: {sample['metadata'].get('IsDefected', False)}")
            print(f"    - LayerNum: {sample['metadata'].get('LayerNum', 'N/A')}")
    
    # 메타데이터 구조 분석
    if total_images > 0:
        print(f"\n[메타데이터 구조 분석]")
        sample_metadata = defect_samples[0]['metadata'] if defect_samples else normal_samples[0]['metadata']
        print(f"  - 주요 키: {list(sample_metadata.keys())[:10]}")
        
        # IsDefected 필드 분포
        is_defected_count = sum(1 for s in defect_samples if s['metadata'].get('IsDefected', False))
        print(f"  - IsDefected=True인 샘플: {is_defected_count}개")
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터셋에서 결함 샘플 확인')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='데이터 디렉토리 (기본값: data/labeled_layers)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='최대 검사할 샘플 수 (기본값: 전체)')
    
    args = parser.parse_args()
    
    check_defect_data(
        data_dir=args.data_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

