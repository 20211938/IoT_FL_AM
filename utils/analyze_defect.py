"""
데이터 분석을 위한 스크립트
- data_train 위치 자동 탐지 및 분석
- 결함 분포 분석
- 클라이언트별 데이터 분포 분석
"""

from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import os


def find_project_root():
    """
    프로젝트 루트 디렉토리를 자동으로 찾기
    utils 폴더의 부모 디렉토리를 프로젝트 루트로 간주
    """
    script_path = Path(__file__).resolve()
    # utils/analyze_defect.py -> utils -> 프로젝트 루트
    project_root = script_path.parent.parent
    return project_root


def find_data_train(data_dir=None):
    """
    data_train 디렉토리 위치를 찾기
    
    Args:
        data_dir: 직접 지정한 경로 (None이면 자동 탐지)
    
    Returns:
        data_train 경로 (Path 객체)
    """
    if data_dir is None:
        project_root = find_project_root()
        data_train_path = project_root / 'data_test'
    else:
        data_train_path = Path(data_dir)
    
    return data_train_path.resolve()


def analyze_data_train_location(data_dir=None):
    """
    data_train 위치 및 구조 분석
    
    Args:
        data_dir: data_train 경로 (None이면 자동 탐지)
    
    Returns:
        분석 결과 딕셔너리
    """
    data_train_path = find_data_train(data_dir)
    
    print("=" * 60)
    print("data_train 위치 분석")
    print("=" * 60)
    print(f"프로젝트 루트: {find_project_root()}")
    print(f"data_train 경로: {data_train_path}")
    print(f"절대 경로: {data_train_path.absolute()}")
    print(f"존재 여부: {'존재함' if data_train_path.exists() else '존재하지 않음'}")
    
    if not data_train_path.exists():
        print("\n경고: data_train 폴더를 찾을 수 없습니다!")
        return None
    
    # 폴더 구조 확인
    folder_0 = data_train_path / '0'
    folder_1 = data_train_path / '1'
    folder_annotations = data_train_path / 'annotations'
    
    print(f"\n폴더 구조:")
    print(f"  - 0/ 폴더: {'존재' if folder_0.exists() else '없음'}")
    print(f"  - 1/ 폴더: {'존재' if folder_1.exists() else '없음'}")
    print(f"  - annotations/ 폴더: {'존재' if folder_annotations.exists() else '없음'}")
    
    # 파일 개수 확인
    file_counts = {}
    if folder_0.exists():
        jpg_files_0 = list(folder_0.glob('*.jpg'))
        file_counts['0'] = len(jpg_files_0)
        print(f"  - 0/ 폴더 파일 수: {len(jpg_files_0)}개")
    
    if folder_1.exists():
        jpg_files_1 = list(folder_1.glob('*.jpg'))
        file_counts['1'] = len(jpg_files_1)
        print(f"  - 1/ 폴더 파일 수: {len(jpg_files_1)}개")
    
    if folder_annotations.exists():
        npy_files = list(folder_annotations.glob('*.npy'))
        file_counts['annotations'] = len(npy_files)
        print(f"  - annotations/ 폴더 파일 수: {len(npy_files)}개")
    
    # 디스크 사용량 확인 (선택사항)
    try:
        total_size = sum(f.stat().st_size for f in data_train_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"\n총 디스크 사용량: {size_mb:.2f} MB")
    except Exception as e:
        print(f"\n디스크 사용량 계산 실패: {e}")
    
    return {
        'path': data_train_path,
        'exists': True,
        'file_counts': file_counts,
        'folders': {
            '0': folder_0.exists(),
            '1': folder_1.exists(),
            'annotations': folder_annotations.exists()
        }
    }



def analyze_defect_distribution(data_dir=None, client_identifier_dict=None):
    """
    각 이미지에 여러 결함 유형이 함께 나타나는 비율 분석
    
    Args:
        data_dir: data_train 경로 (None이면 자동 탐지)
        client_identifier_dict: 클라이언트별 파일 딕셔너리 (None이면 전체 분석)
    """
    data_train_path = find_data_train(data_dir)
    annotations_path = data_train_path / 'annotations'
    
    if not annotations_path.exists():
        print(f"오류: annotations 폴더를 찾을 수 없습니다: {annotations_path}")
        return None
    
    print("\n" + "=" * 60)
    print("결함 분포 분석")
    print("=" * 60)
    
    multi_defect_count = 0  # 여러 결함이 있는 이미지 수
    single_defect_count = 0  # 단일 결함만 있는 이미지 수
    no_defect_count = 0  # 결함이 없는 이미지 수
    defect_distribution = []  # 결함 유형 수 분포
    defect_type_counter = Counter()  # 결함 유형별 개수
    only_01_files = []  # 0과 1만 있는 파일 목록 (디버깅용)
    
    # 분석할 파일 리스트 결정
    if client_identifier_dict is not None:
        files_to_analyze = []
        for file_list in client_identifier_dict.values():
            files_to_analyze.extend(file_list)
    else:
        # 전체 파일 분석
        npy_files = list(annotations_path.glob('*.npy'))
        files_to_analyze = [f.stem for f in npy_files]
    
    print(f"분석할 파일 수: {len(files_to_analyze)}개")
    
    for file_name in files_to_analyze:
        npy_path = annotations_path / f"{file_name}.npy"
        if not npy_path.exists():
            continue
        
        try:
            mask = np.load(str(npy_path))
            unique_values = np.unique(mask)
            defect_values = [v for v in unique_values if v not in [0, 1]]
            
            # 0과 1만 있는 파일 확인
            if len(defect_values) == 0:
                only_01_files.append({
                    'file_name': file_name,
                    'unique_values': list(unique_values)
                })
            
            # 결함 유형별 카운트
            for defect_type in defect_values:
                defect_type_counter[defect_type] += 1
            
            num_defects = len(defect_values)
            defect_distribution.append(num_defects)
            
            if num_defects > 1:
                multi_defect_count += 1
            elif num_defects == 1:
                single_defect_count += 1
            else:
                no_defect_count += 1
        except Exception as e:
            print(f"경고: {file_name}.npy 파일 처리 중 오류 발생: {e}")
            continue
    
    total = multi_defect_count + single_defect_count + no_defect_count
    
    print(f"\n전체 이미지: {total}개")
    print(f"결함 없는 이미지: {no_defect_count}개 ({no_defect_count/total*100:.1f}%)")
    print(f"단일 결함 이미지: {single_defect_count}개 ({single_defect_count/total*100:.1f}%)")
    print(f"다중 결함 이미지: {multi_defect_count}개 ({multi_defect_count/total*100:.1f}%)")
    
    # 0과 1만 있는 파일 상세 정보 출력
    if only_01_files:
        print(f"\n⚠️  0과 1만 있는 파일 발견: {len(only_01_files)}개")
        print("상세 정보:")
        for item in only_01_files[:10]:  # 처음 10개만 출력
            print(f"  - {item['file_name']}: unique_values = {item['unique_values']}")
        if len(only_01_files) > 10:
            print(f"  ... 외 {len(only_01_files) - 10}개 파일")
    else:
        print(f"\n✓ 0과 1만 있는 파일 없음 (모든 파일에 결함 유형 존재)")
    
    print(f"\n결함 유형 수 분포:")
    defect_count_dist = Counter(defect_distribution)
    for num_defects in sorted(defect_count_dist.keys()):
        count = defect_count_dist[num_defects]
        print(f"  {num_defects}개 결함: {count}개 이미지 ({count/total*100:.1f}%)")
    
    if defect_type_counter:
        print(f"\n결함 유형별 개수:")
        for defect_type in sorted(defect_type_counter.keys()):
            count = defect_type_counter[defect_type]
            print(f"  결함 유형 {defect_type}: {count}개")
    
    return {
        'total': total,
        'no_defect': no_defect_count,
        'single_defect': single_defect_count,
        'multi_defect': multi_defect_count,
        'defect_distribution': defect_count_dist,
        'defect_type_counter': defect_type_counter,
        'only_01_files': only_01_files  # 디버깅 정보 추가
    }




def analyze_client_distribution(data_dir=None, client_identifier_dict=None):
    """
    클라이언트별 데이터 분포 분석
    
    Args:
        data_dir: data_train 경로 (None이면 자동 탐지)
        client_identifier_dict: 클라이언트별 파일 딕셔너리
    """
    if client_identifier_dict is None:
        print("경고: client_identifier_dict가 제공되지 않았습니다.")
        return None
    
    data_train_path = find_data_train(data_dir)
    annotations_path = data_train_path / 'annotations'
    
    print("\n" + "=" * 60)
    print("클라이언트별 데이터 분포 분석")
    print("=" * 60)
    
    client_stats = {}
    
    for client_id, file_list in client_identifier_dict.items():
        client_defect_count = Counter()
        client_multi_defect = 0
        client_single_defect = 0
        client_no_defect = 0
        
        for file_name in file_list:
            npy_path = annotations_path / f"{file_name}.npy"
            if not npy_path.exists():
                continue
            
            try:
                mask = np.load(str(npy_path))
                unique_values = np.unique(mask)
                defect_values = [v for v in unique_values if v not in [0, 1]]
                
                for defect_type in defect_values:
                    client_defect_count[defect_type] += 1
                
                num_defects = len(defect_values)
                if num_defects > 1:
                    client_multi_defect += 1
                elif num_defects == 1:
                    client_single_defect += 1
                else:
                    client_no_defect += 1
            except Exception as e:
                continue
        
        total_files = len(file_list)
        analyzed_files = client_multi_defect + client_single_defect + client_no_defect
        
        client_stats[client_id] = {
            'total_files': total_files,
            'analyzed_files': analyzed_files,
            'no_defect': client_no_defect,
            'single_defect': client_single_defect,
            'multi_defect': client_multi_defect,
            'defect_types': dict(client_defect_count)
        }
        
        print(f"\n{client_id}:")
        print(f"  총 파일 수: {total_files}개")
        print(f"  분석된 파일 수: {analyzed_files}개")
        print(f"  결함 없는 이미지: {client_no_defect}개 ({client_no_defect/analyzed_files*100:.1f}%)" if analyzed_files > 0 else "  결함 없는 이미지: 0개")
        print(f"  단일 결함 이미지: {client_single_defect}개 ({client_single_defect/analyzed_files*100:.1f}%)" if analyzed_files > 0 else "  단일 결함 이미지: 0개")
        print(f"  다중 결함 이미지: {client_multi_defect}개 ({client_multi_defect/analyzed_files*100:.1f}%)" if analyzed_files > 0 else "  다중 결함 이미지: 0개")
        if client_defect_count:
            print(f"  결함 유형: {dict(client_defect_count)}")
    
    return client_stats


def main():
    """
    메인 함수: 전체 분석 수행
    """
    # 1. data_train 위치 분석
    location_info = analyze_data_train_location()
    
    if location_info is None:
        return
    
    # 2. 전체 결함 분포 분석
    defect_info = analyze_defect_distribution()
    
    # 3. 클라이언트별 분석은 client_identifier_dict가 필요
    # 이 부분은 호출하는 쪽에서 제공해야 함
    print("\n" + "=" * 60)
    print("분석 완료")
    print("=" * 60)


if __name__ == '__main__':
    main()