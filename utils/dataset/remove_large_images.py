"""
이미지 파일 크기를 확인하여 500KB 이상인 데이터를 삭제하는 스크립트
- merged_data 폴더의 0, 1, annotations 폴더에서 작업
- 0 또는 1 폴더의 이미지가 500KB 이상이면 해당 파일 그룹 전체 삭제
"""

import os
from pathlib import Path


def remove_large_images(data_dir='merged_data', size_threshold_kb=500):
    """
    이미지 파일 크기를 확인하여 임계값 이상인 데이터를 삭제하는 함수
    
    Args:
        data_dir: 데이터 폴더 경로 (기본값: 'merged_data')
        size_threshold_kb: 삭제할 파일 크기 임계값 (KB 단위, 기본값: 500KB)
    """
    data_path = Path(data_dir)
    
    # 폴더 경로 설정
    folder_0 = data_path / '0'
    folder_1 = data_path / '1'
    folder_annotations = data_path / 'annotations'
    
    # 폴더 존재 확인
    if not folder_0.exists():
        print(f"경고: {folder_0} 폴더를 찾을 수 없습니다.")
        return
    
    if not folder_1.exists():
        print(f"경고: {folder_1} 폴더를 찾을 수 없습니다.")
        return
    
    if not folder_annotations.exists():
        print(f"경고: {folder_annotations} 폴더를 찾을 수 없습니다.")
        return
    
    # 0 폴더의 모든 jpg 파일 찾기
    jpg_files_0 = list(folder_0.glob('*.jpg'))
    
    if not jpg_files_0:
        print(f"경고: {folder_0}에 jpg 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(jpg_files_0)}개의 이미지 파일 발견")
    print(f"크기 임계값: {size_threshold_kb}KB 이상인 파일 삭제\n")
    
    # 삭제할 파일 목록
    files_to_delete = []
    size_threshold_bytes = size_threshold_kb * 1024  # KB를 바이트로 변환
    
    # 각 이미지 파일 크기 확인
    for jpg_file_0 in jpg_files_0:
        file_stem = jpg_file_0.stem  # 확장자 제거 (예: 000001)
        
        # 0 폴더 파일 크기 확인
        file_0_size = jpg_file_0.stat().st_size if jpg_file_0.exists() else 0
        
        # 1 폴더 파일 크기 확인
        file_1_path = folder_1 / f"{file_stem}.jpg"
        file_1_size = file_1_path.stat().st_size if file_1_path.exists() else 0
        
        # annotation 파일 경로
        annotation_path = folder_annotations / f"{file_stem}.npy"
        
        # 0 또는 1 폴더의 이미지가 임계값 이상이면 삭제 대상에 추가
        if file_0_size >= size_threshold_bytes or file_1_size >= size_threshold_bytes:
            files_to_delete.append({
                'stem': file_stem,
                'file_0': jpg_file_0,
                'file_1': file_1_path,
                'annotation': annotation_path,
                'size_0_kb': file_0_size / 1024,
                'size_1_kb': file_1_size / 1024
            })
    
    # 삭제할 파일이 없으면 종료
    if not files_to_delete:
        print("삭제할 파일이 없습니다. 모든 파일이 임계값 미만입니다.")
        return
    
    # 삭제할 파일 정보 출력
    print(f"삭제 대상: {len(files_to_delete)}개 파일 그룹\n")
    print("삭제 대상 파일 목록:")
    print("-" * 80)
    for item in files_to_delete[:20]:  # 처음 20개만 미리보기
        print(f"  {item['stem']}: 0폴더={item['size_0_kb']:.2f}KB, 1폴더={item['size_1_kb']:.2f}KB")
    
    if len(files_to_delete) > 20:
        print(f"  ... 외 {len(files_to_delete) - 20}개 파일")
    
    print("-" * 80)
    
    # 사용자 확인
    response = input(f"\n정말로 {len(files_to_delete)}개 파일 그룹을 삭제하시겠습니까? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("삭제 작업이 취소되었습니다.")
        return
    
    # 파일 삭제 실행
    deleted_count = 0
    failed_count = 0
    
    print("\n파일 삭제 중...")
    for item in files_to_delete:
        try:
            # 0 폴더 파일 삭제
            if item['file_0'].exists():
                item['file_0'].unlink()
            
            # 1 폴더 파일 삭제
            if item['file_1'].exists():
                item['file_1'].unlink()
            
            # annotation 파일 삭제
            if item['annotation'].exists():
                item['annotation'].unlink()
            
            deleted_count += 1
            
            if deleted_count % 50 == 0:
                print(f"  진행 중: {deleted_count}개 파일 그룹 삭제 완료")
                
        except Exception as e:
            print(f"  경고: {item['stem']} 삭제 중 오류 발생: {e}")
            failed_count += 1
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("삭제 작업 완료!")
    print(f"  - 삭제된 파일 그룹: {deleted_count}개")
    if failed_count > 0:
        print(f"  - 삭제 실패: {failed_count}개")
    
    # 남은 파일 개수 확인
    remaining_files_0 = len(list(folder_0.glob('*.jpg')))
    remaining_files_1 = len(list(folder_1.glob('*.jpg')))
    remaining_files_ann = len(list(folder_annotations.glob('*.npy')))
    
    print(f"\n남은 파일 개수:")
    print(f"  - {data_dir}/0: {remaining_files_0}개 파일")
    print(f"  - {data_dir}/1: {remaining_files_1}개 파일")
    print(f"  - {data_dir}/annotations: {remaining_files_ann}개 파일")
    print("=" * 80)


if __name__ == '__main__':
    # 기본값으로 실행 (500KB 이상 삭제)
    remove_large_images()
    
    # 다른 임계값을 사용하려면 아래처럼 호출
    # remove_large_images(size_threshold_kb=500)

