"""
프로젝트 경로 관리 유틸리티
프로젝트 루트를 자동으로 감지하고 일관된 경로를 제공합니다.
"""

import os
from pathlib import Path
from typing import Optional


# 프로젝트 루트 디렉토리 자동 감지
def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리를 자동으로 감지합니다.
    
    현재 파일의 위치를 기준으로 .git 디렉토리나 README.md 파일이 있는
    상위 디렉토리를 찾아 프로젝트 루트로 반환합니다.
    
    Returns:
        Path: 프로젝트 루트 디렉토리 경로
    """
    current_file = Path(__file__).resolve()
    current_dir = current_file.parent
    
    # utils 디렉토리에서 시작하여 프로젝트 루트 찾기
    for parent in [current_dir.parent] + list(current_dir.parents):
        # .git 디렉토리나 README.md 파일이 있으면 프로젝트 루트로 간주
        if (parent / '.git').exists() or (parent / 'README.md').exists():
            return parent
    
    # 찾지 못하면 현재 파일의 상위 2단계 디렉토리 반환 (utils의 부모)
    return current_dir.parent


# 프로젝트 루트 경로 (모듈 로드 시 한 번만 계산)
PROJECT_ROOT = get_project_root()


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    데이터 디렉토리 경로를 반환합니다.
    
    Args:
        subdir: 하위 디렉토리 이름 (예: 'labeled_layers')
    
    Returns:
        Path: 데이터 디렉토리 경로
    """
    data_dir = PROJECT_ROOT / 'data'
    if subdir:
        return data_dir / subdir
    return data_dir


def get_logs_dir() -> Path:
    """
    로그 디렉토리 경로를 반환합니다.
    
    Returns:
        Path: 로그 디렉토리 경로
    """
    return PROJECT_ROOT / 'logs'


def get_checkpoints_dir() -> Path:
    """
    체크포인트 디렉토리 경로를 반환합니다.
    
    Returns:
        Path: 체크포인트 디렉토리 경로
    """
    return PROJECT_ROOT / 'checkpoints'


def get_models_dir() -> Path:
    """
    모델 디렉토리 경로를 반환합니다.
    
    Returns:
        Path: 모델 디렉토리 경로
    """
    return PROJECT_ROOT / 'models'


def get_output_dir() -> Path:
    """
    출력 디렉토리 경로를 반환합니다.
    
    Returns:
        Path: 출력 디렉토리 경로
    """
    return PROJECT_ROOT / 'output'


def ensure_dir(path: Path) -> Path:
    """
    디렉토리가 존재하지 않으면 생성합니다.
    
    Args:
        path: 디렉토리 경로
    
    Returns:
        Path: 생성된 디렉토리 경로
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# 기본 경로 상수
DATA_DIR = get_data_dir()
LABELED_LAYERS_DIR = get_data_dir('labeled_layers')
LOGS_DIR = get_logs_dir()
CHECKPOINTS_DIR = get_checkpoints_dir()
MODELS_DIR = get_models_dir()
OUTPUT_DIR = get_output_dir()


# 경로를 문자열로 변환하는 헬퍼 함수
def to_str(path: Path) -> str:
    """
    Path 객체를 문자열로 변환합니다.
    
    Args:
        path: Path 객체
    
    Returns:
        str: 경로 문자열
    """
    return str(path)


if __name__ == "__main__":
    # 테스트: 경로 출력
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"레이블된 레이어 디렉토리: {LABELED_LAYERS_DIR}")
    print(f"로그 디렉토리: {LOGS_DIR}")
    print(f"체크포인트 디렉토리: {CHECKPOINTS_DIR}")
    print(f"모델 디렉토리: {MODELS_DIR}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")

