"""
데이터셋 정리 및 삭제 스크립트
- TagBoxes가 2개 이상인 항목 삭제
- Comment 분포 상위 7개만 유지, 나머지 삭제
- "Laser capture timing error", "Recoater capture timing error" 삭제
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict


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


def cleanup_dataset(
    data_dir: Path,
    dry_run: bool = False,
    verbose: bool = True,
    top_n_comments: int = 7,
) -> Dict[str, int]:
    """
    데이터셋 정리:
    - TagBoxes가 2개 이상인 항목 삭제
    - Comment 분포 상위 N개만 유지, 나머지 삭제
    - "Laser capture timing error", "Recoater capture timing error" 삭제
    
    Args:
        data_dir: 데이터 디렉토리 경로
        dry_run: True이면 실제 삭제하지 않고 미리보기만 수행
        verbose: True이면 상세한 진행 상황 출력
        top_n_comments: 유지할 상위 Comment 개수 (기본값: 7)
    
    Returns:
        정리 통계 딕셔너리
    """
    if verbose:
        print("\n" + "=" * 60)
        print("데이터셋 정리 시작")
        print("=" * 60)
        print(f"\n[정리 기준]")
        print(f"  - TagBoxes가 2개 이상인 항목 삭제")
        print(f"  - Comment 분포 상위 {top_n_comments}개만 유지, 나머지 삭제")
        print(f"  - 'Laser capture timing error' 삭제")
        print(f"  - 'Recoater capture timing error' 삭제")
        print(f"  - 모드: {'DRY RUN' if dry_run else '실제 삭제'}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        if verbose:
            print(f"[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return {}
    
    # 1단계: Comment 분포 분석
    if verbose:
        print(f"\n[1단계] Comment 분포 분석 중...")
    comment_counts = Counter()
    file_metadata_info = {}  # 파일 경로 -> (tagbox_count, comments 리스트)
    
    total_files = 0
    for img_file in data_path.glob("*.jpg"):
        json_file = img_file.with_suffix(".jpg.json")
        if not json_file.exists():
            continue
        
        total_files += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # DepositionImageModel의 TagBoxes 확인
            deposition_model = metadata.get('DepositionImageModel', {})
            tag_boxes = deposition_model.get('TagBoxes', [])
            tagbox_count = len(tag_boxes)
            
            # Comment 수집 및 정규화
            comments = []
            for tag_box in tag_boxes:
                comment = tag_box.get('Comment', '')
                comment_str = normalize_comment(comment)
                comments.append(comment_str)
                comment_counts[comment_str] += 1
            
            file_metadata_info[str(img_file)] = {
                'image': img_file,
                'json': json_file,
                'tagbox_count': tagbox_count,
                'comments': comments,
                'metadata': metadata
            }
        except Exception as e:
            if verbose:
                print(f"  [경고] 파일 읽기 실패: {img_file.name} - {e}")
            continue
    
    if verbose:
        print(f"  - 검사한 파일 수: {total_files}")
        print(f"  - 발견된 Comment 종류 수: {len(comment_counts)}")
    
    if total_files == 0:
        if verbose:
            print("\n[결과] 처리할 파일이 없습니다.")
        return {'total_files': 0, 'deleted_files': 0}
    
    # 상위 N개 Comment 결정
    sorted_comments = sorted(comment_counts.items(), key=lambda x: x[1], reverse=True)
    top_comments = set([comment for comment, _ in sorted_comments[:top_n_comments]])
    
    # 삭제할 특정 Comment 목록
    comments_to_remove = {
        normalize_comment("Laser capture timing error"),
        normalize_comment("Recoater capture timing error"),
    }
    
    if verbose:
        print(f"\n[유지할 상위 {top_n_comments}개 Comment]")
        for idx, (comment, count) in enumerate(sorted_comments[:top_n_comments], 1):
            percentage = count / total_files * 100 if total_files > 0 else 0
            comment_display = comment[:50] + "..." if len(comment) > 50 else comment
            print(f"  {idx}. {comment_display}: {count}개 ({percentage:.2f}%)")
        
        # 삭제할 특정 Comment 확인
        found_remove_comments = []
        for comment in comments_to_remove:
            if comment in comment_counts:
                found_remove_comments.append((comment, comment_counts[comment]))
        
        if found_remove_comments:
            print(f"\n[삭제할 특정 Comment]")
            for comment, count in found_remove_comments:
                percentage = count / total_files * 100 if total_files > 0 else 0
                print(f"  - {comment}: {count}개 ({percentage:.2f}%)")
    
    # 2단계: 삭제 대상 파일 찾기
    if verbose:
        print(f"\n[2단계] 삭제 대상 파일 찾는 중...")
    files_to_remove = []
    
    for img_path_str, info in file_metadata_info.items():
        should_remove = False
        reasons = []
        
        # 조건 1: TagBoxes가 2개 이상인 경우
        if info['tagbox_count'] >= 2:
            should_remove = True
            reasons.append(f"TagBoxes {info['tagbox_count']}개")
        
        # 조건 2: Comment가 상위 N개에 없는 경우
        # TagBoxes가 있는 경우에만 Comment 체크
        if info['tagbox_count'] > 0:
            has_valid_comment = False
            has_remove_comment = False
            
            for comment in info['comments']:
                # 특정 삭제 대상 Comment 확인
                if comment in comments_to_remove:
                    has_remove_comment = True
                    reasons.append(f"삭제 대상 Comment: {comment}")
                    break
                # 상위 Comment 확인
                if comment in top_comments:
                    has_valid_comment = True
            
            # 특정 삭제 대상 Comment가 있으면 삭제
            if has_remove_comment:
                should_remove = True
            # 특정 삭제 대상이 없고 상위 Comment에도 없으면 삭제
            elif not has_valid_comment:
                should_remove = True
                reasons.append("상위 Comment에 없음")
        
        if should_remove:
            files_to_remove.append({
                'image': info['image'],
                'json': info['json'],
                'reasons': reasons,
                'tagbox_count': info['tagbox_count'],
                'comments': info['comments']
            })
    
    if verbose:
        print(f"  - 삭제 대상 파일 수: {len(files_to_remove)}개")
        
        # 삭제 사유별 통계
        reason_stats = Counter()
        tagbox_count_stats = Counter()
        for item in files_to_remove:
            for reason in item['reasons']:
                reason_stats[reason] += 1
            tagbox_count_stats[item['tagbox_count']] += 1
        
        print(f"\n[삭제 사유별 통계]")
        for reason, count in reason_stats.most_common():
            print(f"  - {reason}: {count}개")
        
        print(f"\n[TagBoxes 개수별 삭제 통계]")
        for count, num_files in sorted(tagbox_count_stats.items()):
            print(f"  - {count}개: {num_files}개 파일")
    
    # 3단계: 실제 삭제
    if not dry_run:
        if verbose:
            print(f"\n[3단계] 파일 삭제 중...")
        deleted_count = 0
        error_count = 0
        
        for item in files_to_remove:
            try:
                if item['image'].exists():
                    item['image'].unlink()
                if item['json'].exists():
                    item['json'].unlink()
                deleted_count += 1
                
                if verbose and deleted_count % 100 == 0:
                    print(f"  - 삭제 진행: {deleted_count}/{len(files_to_remove)}")
            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"  [오류] {item['image'].name} 삭제 실패: {e}")
        
        if verbose:
            print(f"\n[삭제 완료]")
            print(f"  - 삭제된 파일: {deleted_count}개")
            print(f"  - 삭제된 이미지+JSON: {deleted_count * 2}개 파일")
            print(f"  - 오류 발생: {error_count}개")
    else:
        if verbose:
            print(f"\n[DRY RUN 모드]")
            print(f"  - 삭제될 파일 수: {len(files_to_remove)}개 (이미지 + JSON = {len(files_to_remove) * 2}개 파일)")
    
    return {
        'total_files': total_files,
        'deleted_files': len(files_to_remove) if not dry_run else 0,
        'top_comments': list(top_comments),
        'comment_counts': dict(comment_counts)
    }


def main():
    """메인 함수 - CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="데이터셋 정리: TagBoxes 2개 이상 삭제, Comment 상위 7개만 유지"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=7,
        help='유지할 상위 Comment 개수 (기본값: 7)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 삭제하지 않고 미리보기만 수행'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='상세 출력 비활성화'
    )
    
    args = parser.parse_args()
    
    stats = cleanup_dataset(
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        top_n_comments=args.top_n,
    )
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("정리 완료!")
        print("=" * 60)
        print(f"전체 파일: {stats.get('total_files', 0)}개")
        print(f"삭제된 파일: {stats.get('deleted_files', 0)}개")


if __name__ == "__main__":
    main()

