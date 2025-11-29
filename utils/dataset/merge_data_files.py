"""
data 폴더 내부의 모든 파일들을 하나의 폴더에 모으는 스크립트
- 0, 1, annotations 구조는 그대로 유지
- 파일 이름은 000001부터 시작해서 1씩 증가하는 형태로 저장
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict


def merge_data_files(source_dir=None, output_dir=None):
    """
    data 폴더 내부의 모든 파일들을 하나의 폴더에 모으는 함수
    
    Args:
        source_dir: 소스 데이터 폴더 경로 (기본값: 프로젝트 루트의 'data')
        output_dir: 출력 폴더 경로 (기본값: 프로젝트 루트의 'merged_data')
    """
    # 스크립트 파일의 위치를 기준으로 프로젝트 루트 찾기
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent  # new_utils의 부모 폴더 = 프로젝트 루트
    
    # 기본 경로 설정 (프로젝트 루트 기준)
    if source_dir is None:
        source_dir = project_root / 'dataset'
    if output_dir is None:
        output_dir = project_root / 'data_test'
    
    # Path 객체로 변환
    source_path = Path(source_dir) if isinstance(source_dir, (str, Path)) else source_dir
    output_path = Path(output_dir) if isinstance(output_dir, (str, Path)) else output_dir
    
    # 절대 경로로 변환
    if not source_path.is_absolute():
        source_path = project_root / source_path
    if not output_path.is_absolute():
        output_path = project_root / output_path
    
    # 경로 정보 출력
    print(f"소스 폴더: {source_path}")
    print(f"출력 폴더: {output_path}")
    
    # 소스 폴더 존재 확인
    if not source_path.exists():
        print(f"오류: 소스 폴더를 찾을 수 없습니다: {source_path}")
        return
    
    # 출력 폴더 생성
    output_path.mkdir(exist_ok=True)
    
    # 0, 1, annotations 폴더 생성
    folder_0 = output_path / '0'
    folder_1 = output_path / '1'
    folder_annotations = output_path / 'annotations'
    
    folder_0.mkdir(exist_ok=True)
    folder_1.mkdir(exist_ok=True)
    folder_annotations.mkdir(exist_ok=True)
    
    # 데이터 구조 확인: data/0/, data/1/, data/annotations/ 구조인지 확인
    folder_0_source = source_path / '0'
    folder_1_source = source_path / '1'
    annotations_folder_source = source_path / 'annotations'
    
    # 구조 1: data/0/, data/1/, data/annotations/ 직접 구조
    if folder_0_source.exists() and folder_0_source.is_dir():
        print("데이터 구조: data/0/, data/1/, data/annotations/ 직접 구조 감지")
        file_groups = []
        
        # 0 폴더의 모든 jpg 파일 찾기
        jpg_files_0 = list(folder_0_source.glob('*.jpg'))
        
        if not jpg_files_0:
            print(f"경고: {folder_0_source}에 jpg 파일을 찾을 수 없습니다.")
            return
        
        print(f"  {len(jpg_files_0)}개의 이미지 파일 발견")
        
        # 각 이미지 파일에 대해 1과 annotations 파일 찾기
        for jpg_file_0 in jpg_files_0:
            file_stem = jpg_file_0.stem  # 확장자 제거
            
            file_0_path = folder_0_source / f"{file_stem}.jpg"
            file_1_path = folder_1_source / f"{file_stem}.jpg" if folder_1_source.exists() else None
            annotation_path = annotations_folder_source / f"{file_stem}.npy" if annotations_folder_source.exists() else None
            
            # 파일 존재 확인
            if not file_0_path.exists():
                continue
            
            file_groups.append({
                'dataset': 'direct',
                'stem': file_stem,
                'file_0': file_0_path,
                'file_1': file_1_path if file_1_path and file_1_path.exists() else None,
                'annotation': annotation_path if annotation_path and annotation_path.exists() else None
            })
    
    # 구조 2: data/dataset1/0/, data/dataset1/1/, data/dataset1/annotations/ 구조
    else:
        print("데이터 구조: data/dataset/0/, data/dataset/1/, data/dataset/annotations/ 구조 감지")
        # 모든 데이터셋 폴더 찾기
        dataset_folders = [f for f in source_path.iterdir() if f.is_dir() and f.name not in ['0', '1', 'annotations']]
        
        if not dataset_folders:
            print(f"경고: {source_path} 폴더에 데이터셋 폴더를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(dataset_folders)}개의 데이터셋 폴더 발견")
        
        # 파일 그룹을 저장할 리스트
        file_groups = []
        
        # 각 데이터셋 폴더 처리
        for dataset_folder in dataset_folders:
            dataset_name = dataset_folder.name
            print(f"\n처리 중: {dataset_name}")
            
            folder_0_source = dataset_folder / '0'
            folder_1_source = dataset_folder / '1'
            annotations_folder_source = dataset_folder / 'annotations'
            
            # 0 폴더의 모든 jpg 파일 찾기
            if not folder_0_source.exists():
                print(f"  경고: {folder_0_source} 폴더를 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            jpg_files_0 = list(folder_0_source.glob('*.jpg'))
            
            if not jpg_files_0:
                print(f"  경고: {folder_0_source}에 jpg 파일을 찾을 수 없습니다.")
                continue
            
            print(f"  {len(jpg_files_0)}개의 이미지 파일 발견")
            
            # 각 이미지 파일에 대해 1과 annotations 파일 찾기
            for jpg_file_0 in jpg_files_0:
                file_stem = jpg_file_0.stem  # 확장자 제거
                
                file_0_path = folder_0_source / f"{file_stem}.jpg"
                file_1_path = folder_1_source / f"{file_stem}.jpg" if folder_1_source.exists() else None
                annotation_path = annotations_folder_source / f"{file_stem}.npy" if annotations_folder_source.exists() else None
                
                # 파일 존재 확인
                if not file_0_path.exists():
                    continue
                
                file_groups.append({
                    'dataset': dataset_name,
                    'stem': file_stem,
                    'file_0': file_0_path,
                    'file_1': file_1_path if file_1_path and file_1_path.exists() else None,
                    'annotation': annotation_path if annotation_path and annotation_path.exists() else None
                })
    
    # 파일 그룹이 비어있는지 확인
    if not file_groups:
        print(f"\n경고: 처리할 파일 그룹이 없습니다.")
        return
    
    print(f"\n총 {len(file_groups)}개의 파일 그룹 발견")
    print("파일 복사 시작...")
    
    # 파일 번호 카운터
    file_counter = 1
    
    # 각 파일 그룹을 순차적으로 복사
    for group in file_groups:
        new_name = f"{file_counter:06d}"  # 000001, 000002, ...
        
        # 0 폴더 파일 복사
        if group['file_0']:
            dest_0 = folder_0 / f"{new_name}.jpg"
            shutil.copy2(group['file_0'], dest_0)
        
        # 1 폴더 파일 복사
        if group['file_1']:
            dest_1 = folder_1 / f"{new_name}.jpg"
            shutil.copy2(group['file_1'], dest_1)
        
        # annotations 폴더 파일 복사
        if group['annotation']:
            dest_ann = folder_annotations / f"{new_name}.npy"
            shutil.copy2(group['annotation'], dest_ann)
        
        file_counter += 1
        
        if file_counter % 100 == 0:
            print(f"  진행 중: {file_counter}개 파일 처리 완료")
    
    output_dir_str = str(output_path)
    print(f"\n완료! 총 {file_counter - 1}개의 파일 그룹을 {output_dir_str} 폴더에 복사했습니다.")
    print(f"  - {output_dir_str}/0: {len(list(folder_0.glob('*.jpg')))}개 파일")
    print(f"  - {output_dir_str}/1: {len(list(folder_1.glob('*.jpg')))}개 파일")
    print(f"  - {output_dir_str}/annotations: {len(list(folder_annotations.glob('*.npy')))}개 파일")


if __name__ == '__main__':
    merge_data_files()

