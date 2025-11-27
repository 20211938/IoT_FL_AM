"""
데이터셋의 IsLabeled, TagBoxes, Comment 분포 분석 스크립트
"""

import os
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import sys

# 경로 유틸리티 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import DATA_DIR, to_str


def normalize_comment(comment: str) -> str:
    """
    Comment 문자열 정규화:
    - 공백 정규화 (여러 공백을 하나로, 앞뒤 공백 제거)
    - 일반적인 오타 수정
    """
    if not comment:
        return '(빈 값)'
    
    # 문자열로 변환하고 앞뒤 공백 제거
    normalized = str(comment).strip()
    
    # 여러 공백을 하나로 통일
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 일반적인 오타 수정
    # "Reocater" -> "Recoater" (오타 수정)
    normalized = re.sub(r'\bReocater\b', 'Recoater', normalized, flags=re.IGNORECASE)
    
    return normalized


def analyze_defect_dataset(data_dir: str = None):
    """데이터셋의 IsLabeled, TagBoxes, Comment 분포 분석"""
    if data_dir is None:
        data_dir = to_str(DATA_DIR)
    else:
        data_dir = to_str(Path(data_dir).resolve())
    
    print("=" * 60)
    print("데이터셋 IsLabeled, TagBoxes, Comment 분포 분석")
    print("=" * 60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    # 통계 정보
    total_images = 0
    is_labeled_true = 0
    is_labeled_false = 0
    is_labeled_missing = 0
    
    # IsLabeled가 true인 경우의 통계
    labeled_true_with_tagboxes = 0
    labeled_true_without_tagboxes = 0
    tagbox_count_distribution = Counter()  # TagBoxes 개수 분포
    comment_distribution = Counter()  # Comment 값 분포
    comment_by_tagbox_count = defaultdict(Counter)  # TagBoxes 개수별 Comment 분포
    
    # data 폴더에 직접 저장된 이미지 파일들을 순회
    for img_file in data_path.glob("*.jpg"):
        json_file = img_file.with_suffix(".jpg.json")
        if not json_file.exists():
            continue
        
        total_images += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # IsLabeled 값 확인
            is_labeled = metadata.get('IsLabeled')
            if is_labeled is None:
                is_labeled_missing += 1
            elif is_labeled in [True, "true", "True", 1]:
                is_labeled_true += 1
                
                # DepositionImageModel 내부의 TagBoxes 확인
                deposition_model = metadata.get('DepositionImageModel', {})
                tag_boxes = deposition_model.get('TagBoxes', [])
                tagbox_count = len(tag_boxes)
                
                # TagBoxes 존재 여부 및 개수 분포
                if tagbox_count > 0:
                    labeled_true_with_tagboxes += 1
                    tagbox_count_distribution[tagbox_count] += 1
                    
                    # 각 TagBox의 Comment 수집 및 정규화
                    for tag_box in tag_boxes:
                        comment = tag_box.get('Comment', '')
                        comment_str = normalize_comment(comment)
                        comment_distribution[comment_str] += 1
                        comment_by_tagbox_count[tagbox_count][comment_str] += 1
                else:
                    labeled_true_without_tagboxes += 1
                    tagbox_count_distribution[0] += 1
                    comment_distribution['(TagBoxes 없음)'] += 1
                    
            else:
                is_labeled_false += 1
                    
        except Exception as e:
            print(f"[경고] {json_file} 처리 중 오류 발생: {e}")
            continue
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("[전체 통계]")
    print("=" * 60)
    print(f"  - 전체 이미지 수: {total_images}")
    print(f"  - IsLabeled가 true: {is_labeled_true}개 ({is_labeled_true/total_images*100:.2f}%)")
    print(f"  - IsLabeled가 false: {is_labeled_false}개 ({is_labeled_false/total_images*100:.2f}%)")
    if is_labeled_missing > 0:
        print(f"  - IsLabeled 필드 없음: {is_labeled_missing}개 ({is_labeled_missing/total_images*100:.2f}%)")
    
    # IsLabeled가 true인 경우의 상세 통계
    print("\n" + "=" * 60)
    print("[IsLabeled=true인 경우 상세 통계]")
    print("=" * 60)
    print(f"  - 전체: {is_labeled_true}개")
    print(f"  - DepositionImageModel.TagBoxes가 있는 경우: {labeled_true_with_tagboxes}개 ({labeled_true_with_tagboxes/is_labeled_true*100:.2f}%)")
    print(f"  - DepositionImageModel.TagBoxes가 없는 경우: {labeled_true_without_tagboxes}개 ({labeled_true_without_tagboxes/is_labeled_true*100:.2f}%)")
    
    # TagBoxes 개수 분포
    print("\n" + "=" * 60)
    print("[TagBoxes 개수 분포] (IsLabeled=true인 경우)")
    print("=" * 60)
    if tagbox_count_distribution:
        sorted_counts = sorted(tagbox_count_distribution.items(), key=lambda x: x[0])
        print(f"\n{'TagBoxes 개수':<20} {'이미지 수':<15} {'비율':<15}")
        print("-" * 50)
        for count, num_images in sorted_counts:
            percentage = num_images / is_labeled_true * 100 if is_labeled_true > 0 else 0
            count_label = f"{count}개" if count > 0 else "없음"
            print(f"{count_label:<20} {num_images:<15} {percentage:>6.2f}%")
    else:
        print("  데이터가 없습니다.")
    
    # Comment 분포
    print("\n" + "=" * 60)
    print("[Comment 분포] (IsLabeled=true인 경우)")
    print("=" * 60)
    if comment_distribution:
        sorted_comments = sorted(comment_distribution.items(), key=lambda x: x[1], reverse=True)
        print(f"\n{'순위':<6} {'Comment 값':<40} {'개수':<15} {'비율':<15}")
        print("-" * 80)
        total_comments = sum(comment_distribution.values())
        for idx, (comment, count) in enumerate(sorted_comments, 1):
            percentage = count / total_comments * 100 if total_comments > 0 else 0
            # Comment가 너무 길면 잘라서 표시
            comment_display = comment[:37] + "..." if len(comment) > 40 else comment
            print(f"{idx:<6} {comment_display:<40} {count:<15} {percentage:>6.2f}%")
    else:
        print("  데이터가 없습니다.")
    
    # TagBoxes 개수별 Comment 분포
    print("\n" + "=" * 60)
    print("[TagBoxes 개수별 Comment 분포] (IsLabeled=true인 경우)")
    print("=" * 60)
    if comment_by_tagbox_count:
        sorted_tagbox_counts = sorted(comment_by_tagbox_count.keys())
        for tagbox_count in sorted_tagbox_counts:
            if tagbox_count == 0:
                continue
            print(f"\n  [TagBoxes {tagbox_count}개인 경우]")
            comments = comment_by_tagbox_count[tagbox_count]
            sorted_comments = sorted(comments.items(), key=lambda x: x[1], reverse=True)
            total_for_count = sum(comments.values())
            print(f"    {'Comment 값':<40} {'개수':<15} {'비율':<15}")
            print("    " + "-" * 70)
            for comment, count in sorted_comments[:10]:  # 상위 10개만 표시
                percentage = count / total_for_count * 100 if total_for_count > 0 else 0
                comment_display = comment[:37] + "..." if len(comment) > 40 else comment
                print(f"    {comment_display:<40} {count:<15} {percentage:>6.2f}%")
            if len(sorted_comments) > 10:
                print(f"    ... 외 {len(sorted_comments) - 10}개 Comment 값")
    else:
        print("  데이터가 없습니다.")
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    
    return {
        'total_images': total_images,
        'is_labeled_true': is_labeled_true,
        'is_labeled_false': is_labeled_false,
        'is_labeled_missing': is_labeled_missing,
        'labeled_true_with_tagboxes': labeled_true_with_tagboxes,
        'labeled_true_without_tagboxes': labeled_true_without_tagboxes,
        'tagbox_count_distribution': dict(tagbox_count_distribution),
        'comment_distribution': dict(comment_distribution),
        'comment_by_tagbox_count': {k: dict(v) for k, v in comment_by_tagbox_count.items()}
    }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터셋의 IsLabeled, TagBoxes, Comment 분포 분석')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='데이터 디렉토리 (기본값: data)')
    
    args = parser.parse_args()
    
    analyze_defect_dataset(data_dir=args.data_dir)


if __name__ == "__main__":
    main()

