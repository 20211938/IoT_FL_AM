"""
데이터셋 증강 스크립트
Spatter와 Humping 항목에 대해 회전, 반전, 밝기 조절을 통한 데이터 증강
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List
from PIL import Image, ImageEnhance


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


def transform_bbox(bbox: Dict, width: int, height: int, 
                   rotation: int = 0, flip_horizontal: bool = False, 
                   flip_vertical: bool = False) -> Dict:
    """
    TagBox 좌표를 변환합니다.
    
    Args:
        bbox: TagBox 딕셔너리 (StartPoint, EndPoint 포함)
        width: 원본 이미지 너비
        height: 원본 이미지 높이
        rotation: 회전 각도 (0, 90, 180, 270)
        flip_horizontal: 좌우 반전 여부
        flip_vertical: 상하 반전 여부
    
    Returns:
        변환된 TagBox 딕셔너리
    """
    start_x = bbox['StartPoint']['X']
    start_y = bbox['StartPoint']['Y']
    end_x = bbox['EndPoint']['X']
    end_y = bbox['EndPoint']['Y']
    
    new_width = width
    new_height = height
    
    # 회전 변환
    if rotation == 90:
        # 90도 시계방향 회전 (PIL rotate -90)
        # (x, y) -> (y, width - x)
        new_start_x = start_y
        new_start_y = width - end_x
        new_end_x = end_y
        new_end_y = width - start_x
        new_width, new_height = height, width
    elif rotation == 180:
        # 180도 회전
        new_start_x = width - end_x
        new_start_y = height - end_y
        new_end_x = width - start_x
        new_end_y = height - start_y
    elif rotation == 270:
        # 270도 시계방향 회전 (PIL rotate 90)
        # (x, y) -> (height - y, x)
        new_start_x = height - end_y
        new_start_y = start_x
        new_end_x = height - start_y
        new_end_y = end_x
        new_width, new_height = height, width
    else:
        new_start_x = start_x
        new_start_y = start_y
        new_end_x = end_x
        new_end_y = end_y
    
    # 좌우 반전
    if flip_horizontal:
        new_start_x, new_end_x = new_width - new_end_x, new_width - new_start_x
    
    # 상하 반전
    if flip_vertical:
        new_start_y, new_end_y = new_height - new_end_y, new_height - new_start_y
    
    # StartPoint와 EndPoint 정렬 (StartPoint가 왼쪽 위, EndPoint가 오른쪽 아래)
    min_x = min(new_start_x, new_end_x)
    max_x = max(new_start_x, new_end_x)
    min_y = min(new_start_y, new_end_y)
    max_y = max(new_start_y, new_end_y)
    
    # 경계 체크
    min_x = max(0, min(min_x, new_width - 1))
    max_x = max(0, min(max_x, new_width - 1))
    min_y = max(0, min(min_y, new_height - 1))
    max_y = max(0, min(max_y, new_height - 1))
    
    transformed_bbox = {
        'StartPoint': {'X': int(min_x), 'Y': int(min_y)},
        'EndPoint': {'X': int(max_x), 'Y': int(max_y)},
        'Name': bbox.get('Name', ''),
        'Comment': bbox.get('Comment', ''),
        'LayerNum': bbox.get('LayerNum', None)
    }
    
    return transformed_bbox


def augment_image(image: Image.Image, rotation: int = 0, 
                  flip_horizontal: bool = False, flip_vertical: bool = False,
                  brightness_factor: float = 1.0) -> Image.Image:
    """
    이미지에 증강을 적용합니다.
    
    Args:
        image: PIL Image 객체
        rotation: 회전 각도 (0, 90, 180, 270)
        flip_horizontal: 좌우 반전 여부
        flip_vertical: 상하 반전 여부
        brightness_factor: 밝기 조절 계수 (1.0 = 원본, >1.0 = 밝게, <1.0 = 어둡게)
    
    Returns:
        증강된 PIL Image 객체
    """
    # 회전
    if rotation == 90:
        image = image.rotate(-90, expand=True)
    elif rotation == 180:
        image = image.rotate(180, expand=True)
    elif rotation == 270:
        image = image.rotate(90, expand=True)
    
    # 좌우 반전
    if flip_horizontal:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 상하 반전
    if flip_vertical:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # 밝기 조절
    if brightness_factor != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_factor)
    
    return image


def augment_dataset(
    data_dir: Path,
    target_comments: List[str] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    데이터셋 증강:
    - 지정된 Comment를 가진 이미지에 대해 증강 수행
    - 회전 (90, 180, 270도)
    - 좌우 반전
    - 상하 반전
    - 밝기 조절 (밝게, 어둡게)
    
    Args:
        data_dir: 데이터 디렉토리 경로
        target_comments: 증강할 Comment 목록 (기본값: ['Spatter', 'Humping'])
        dry_run: True이면 실제 생성하지 않고 미리보기만 수행
        verbose: True이면 상세한 진행 상황 출력
    
    Note:
        이미지당 총 80개의 증강 이미지 생성:
        - 회전: 4가지 (0, 90, 180, 270도)
        - 반전 조합: 4가지 (원본, 좌우, 상하, 좌우+상하)
        - 밝기 조절: 5가지 (1.0, 1.2, 1.4, 0.8, 0.6)
        - 총: 4 × 4 × 5 = 80개
    
    Returns:
        증강 통계 딕셔너리
    """
    if target_comments is None:
        target_comments = ['Spatter', 'Humping']
    
    if verbose:
        print("\n" + "=" * 60)
        print("데이터셋 증강 시작")
        print("=" * 60)
        print(f"\n[증강 설정]")
        print(f"  - 대상 Comment: {', '.join(target_comments)}")
        print(f"  - 이미지당 증강 수: 80개")
        print(f"  - 증강 방법:")
        print(f"    1단계: 회전 4가지 (0, 90, 180, 270도)")
        print(f"    2단계: 반전 조합 4가지 (원본, 좌우, 상하, 좌우+상하) → 총 16개")
        print(f"    3단계: 밝기 조절 5가지 (1.0, 1.2, 1.4, 0.8, 0.6) → 총 80개")
        print(f"  - 모드: {'DRY RUN' if dry_run else '실제 생성'}")
    
    data_path = Path(data_dir)
    if not data_path.exists():
        if verbose:
            print(f"[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return {}
    
    # 대상 파일 찾기
    if verbose:
        print(f"\n[1단계] 대상 파일 찾는 중...")
    
    target_files = []
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
            
            # Comment 확인
            has_target_comment = False
            for tag_box in tag_boxes:
                comment = tag_box.get('Comment', '')
                comment_str = normalize_comment(comment)
                if comment_str in target_comments:
                    has_target_comment = True
                    break
            
            if has_target_comment:
                target_files.append({
                    'image': img_file,
                    'json': json_file,
                    'metadata': metadata
                })
        except Exception as e:
            if verbose:
                print(f"  [경고] 파일 읽기 실패: {img_file.name} - {e}")
            continue
    
    if verbose:
        print(f"  - 전체 파일 수: {total_files}")
        print(f"  - 대상 파일 수: {len(target_files)}")
    
    if len(target_files) == 0:
        if verbose:
            print("\n[결과] 증강할 파일이 없습니다.")
        return {'total_files': total_files, 'augmented_files': 0, 'created_images': 0}
    
    # 증강 설정 생성
    # 1단계: 회전 (0, 90, 180, 270도) → 4개
    rotations = [0, 90, 180, 270]
    
    # 2단계: 각 회전 이미지에 반전 조합 적용 → 4개 × 4가지 = 16개
    # 반전 조합: (좌우X, 상하X), (좌우O, 상하X), (좌우X, 상하O), (좌우O, 상하O)
    flip_combinations = [
        (False, False),  # 원본
        (True, False),   # 좌우 반전
        (False, True),   # 상하 반전
        (True, True),    # 좌우+상하 반전
    ]
    
    # 3단계: 각 16개 이미지에 밝기 조절 적용 → 16개 × 5가지 = 80개
    brightness_factors = [1.0, 1.2, 1.4, 0.8, 0.6]  # 원본, 밝게, 더 밝게, 어둡게, 더 어둡게
    
    # 모든 조합 생성
    augment_configs = []
    config_id = 0
    
    for rotation in rotations:
        for flip_h, flip_v in flip_combinations:
            for brightness in brightness_factors:
                config_id += 1
                
                # suffix 생성
                suffix_parts = []
                if rotation > 0:
                    suffix_parts.append(f'rot{rotation}')
                if flip_h:
                    suffix_parts.append('flipH')
                if flip_v:
                    suffix_parts.append('flipV')
                if brightness != 1.0:
                    if brightness > 1.0:
                        suffix_parts.append(f'bright{int(brightness*10)}')
                    else:
                        suffix_parts.append(f'dark{int(brightness*10)}')
                
                # suffix 생성 (모든 조합에 대해 고유한 suffix 생성)
                if not suffix_parts:
                    # 원본 (rotation=0, flip=False, brightness=1.0)도 suffix 추가
                    suffix = '_aug001'
                else:
                    suffix = '_' + '_'.join(suffix_parts)
                
                augment_configs.append({
                    'rotation': rotation,
                    'flip_horizontal': flip_h,
                    'flip_vertical': flip_v,
                    'brightness_factor': brightness,
                    'suffix': suffix,
                    'id': config_id
                })
    
    # 총 80개 (4 회전 × 4 반전조합 × 5 밝기)
    total_expected = len(rotations) * len(flip_combinations) * len(brightness_factors)
    
    if verbose:
        print(f"\n[2단계] 증강 설정")
        print(f"  - 회전: {len(rotations)}가지 (0, 90, 180, 270도)")
        print(f"  - 반전 조합: {len(flip_combinations)}가지")
        print(f"  - 밝기 조절: {len(brightness_factors)}가지")
        print(f"  - 총 증강 이미지: {len(augment_configs)}개 (이미지당)")
        print(f"\n  [증강 조합 예시]")
        # 처음 5개와 마지막 5개만 표시
        for idx in [0, 1, 2, 3, 4, len(augment_configs)-5, len(augment_configs)-4, len(augment_configs)-3, len(augment_configs)-2, len(augment_configs)-1]:
            if idx < len(augment_configs):
                config = augment_configs[idx]
                desc = []
                if config['rotation'] > 0:
                    desc.append(f"회전{config['rotation']}도")
                if config['flip_horizontal']:
                    desc.append("좌우반전")
                if config['flip_vertical']:
                    desc.append("상하반전")
                if config['brightness_factor'] != 1.0:
                    desc.append(f"밝기{config['brightness_factor']:.1f}x")
                marker = "..." if idx == len(augment_configs)-5 else ""
                print(f"    {config['id']}. {', '.join(desc) if desc else '원본'} ({config['suffix']}){marker}")
    
    # 증강 수행
    if not dry_run:
        if verbose:
            print(f"\n[3단계] 이미지 증강 중...")
        
        created_count = 0
        error_count = 0
        
        for file_info in target_files:
            try:
                # 원본 이미지 로드
                img = Image.open(file_info['image'])
                width, height = img.size
                
                # 원본 메타데이터 복사
                metadata = json.loads(json.dumps(file_info['metadata']))  # Deep copy
                
                for config in augment_configs:
                    # 증강된 이미지 생성
                    augmented_img = augment_image(
                        img.copy(),
                        rotation=config['rotation'],
                        flip_horizontal=config['flip_horizontal'],
                        flip_vertical=config['flip_vertical'],
                        brightness_factor=config['brightness_factor']
                    )
                    
                    # 새로운 이미지 크기
                    new_width, new_height = augmented_img.size
                    
                    # TagBoxes 좌표 변환
                    deposition_model = metadata.get('DepositionImageModel', {})
                    tag_boxes = deposition_model.get('TagBoxes', [])
                    
                    transformed_tag_boxes = []
                    for tag_box in tag_boxes:
                        transformed_bbox = transform_bbox(
                            tag_box,
                            width=width,
                            height=height,
                            rotation=config['rotation'],
                            flip_horizontal=config['flip_horizontal'],
                            flip_vertical=config['flip_vertical']
                        )
                        transformed_tag_boxes.append(transformed_bbox)
                    
                    # 메타데이터 업데이트
                    augmented_metadata = json.loads(json.dumps(metadata))  # Deep copy
                    augmented_metadata['DepositionImageModel']['TagBoxes'] = transformed_tag_boxes
                    
                    # 파일명 생성
                    base_name = file_info['image'].stem
                    new_img_name = f"{base_name}{config['suffix']}.jpg"
                    new_json_name = f"{base_name}{config['suffix']}.jpg.json"
                    
                    new_img_path = file_info['image'].parent / new_img_name
                    new_json_path = file_info['json'].parent / new_json_name
                    
                    # 저장
                    augmented_img.save(new_img_path, quality=95)
                    with open(new_json_path, 'w', encoding='utf-8') as f:
                        json.dump(augmented_metadata, f, indent=2, ensure_ascii=False)
                    
                    created_count += 1
                    
                    if verbose and created_count % 50 == 0:
                        print(f"  - 생성 진행: {created_count}/{len(target_files) * len(augment_configs)}")
            
            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"  [오류] {file_info['image'].name} 처리 실패: {e}")
        
        if verbose:
            print(f"\n[증강 완료]")
            print(f"  - 대상 파일: {len(target_files)}개")
            print(f"  - 생성된 이미지: {created_count}개")
            print(f"  - 오류 발생: {error_count}개")
    else:
        if verbose:
            print(f"\n[DRY RUN 모드]")
            print(f"  - 대상 파일: {len(target_files)}개")
            print(f"  - 생성될 이미지: {len(target_files) * len(augment_configs)}개")
    
    return {
        'total_files': total_files,
        'target_files': len(target_files),
        'created_images': len(target_files) * len(augment_configs) if not dry_run else 0,
        'augment_configs': len(augment_configs)
    }


def main():
    """메인 함수 - CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="데이터셋 증강: Spatter와 Humping 항목에 대해 회전, 반전, 밝기 조절"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--target-comments',
        nargs='+',
        default=['Spatter', 'Humping'],
        help='증강할 Comment 목록 (기본값: Spatter Humping)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 생성하지 않고 미리보기만 수행'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='상세 출력 비활성화'
    )
    
    args = parser.parse_args()
    
    stats = augment_dataset(
        data_dir=args.data_dir,
        target_comments=args.target_comments,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("증강 완료!")
        print("=" * 60)
        print(f"전체 파일: {stats.get('total_files', 0)}개")
        print(f"대상 파일: {stats.get('target_files', 0)}개")
        print(f"생성된 이미지: {stats.get('created_images', 0)}개")


if __name__ == "__main__":
    main()

