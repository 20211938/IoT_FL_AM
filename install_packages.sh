#!/bin/bash

# ============================================================================
# 필수 패키지 자동 설치 스크립트
# IoT FLAM 프로젝트용
# ============================================================================

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 함수 정의
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# 1단계: 시스템 환경 확인
# ============================================================================
print_header "1단계: 시스템 환경 확인"

# Python 버전 확인
echo "Python 버전 확인 중..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python 버전: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_success "Python 버전: $PYTHON_VERSION"
else
    print_error "Python을 찾을 수 없습니다. Python 3.8 이상을 설치해주세요."
    exit 1
fi

# Python 버전 체크 (3.8 이상)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 이상이 필요합니다. 현재 버전: $PYTHON_VERSION"
    exit 1
fi

# NVIDIA 드라이버 확인
echo ""
echo "NVIDIA 드라이버 확인 중..."
HAS_NVIDIA=false
CUDA_VERSION=""
DRIVER_VERSION=""

if command -v nvidia-smi &> /dev/null; then
    NVIDIA_OUTPUT=$(nvidia-smi 2>&1)
    if [ $? -eq 0 ]; then
        HAS_NVIDIA=true
        DRIVER_VERSION=$(echo "$NVIDIA_OUTPUT" | grep "Driver Version" | awk '{print $3}')
        CUDA_VERSION=$(echo "$NVIDIA_OUTPUT" | grep "CUDA Version" | awk '{print $9}')
        print_success "NVIDIA 드라이버 발견"
        echo "  - 드라이버 버전: $DRIVER_VERSION"
        echo "  - 지원 CUDA 버전: $CUDA_VERSION"
    fi
else
    print_warning "nvidia-smi를 찾을 수 없습니다. CPU 버전으로 설치합니다."
fi

# CUDA 버전 결정
PYTORCH_CUDA_INDEX=""
if [ "$HAS_NVIDIA" = true ]; then
    # 드라이버 버전에 따른 CUDA 버전 선택
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -ge 535 ]; then
        PYTORCH_CUDA_INDEX="cu121"  # CUDA 12.1
        print_success "CUDA 12.1용 PyTorch를 설치합니다."
    elif [ "$DRIVER_MAJOR" -ge 520 ]; then
        PYTORCH_CUDA_INDEX="cu118"  # CUDA 11.8
        print_success "CUDA 11.8용 PyTorch를 설치합니다."
    elif [ "$DRIVER_MAJOR" -ge 470 ]; then
        PYTORCH_CUDA_INDEX="cu117"  # CUDA 11.7
        print_success "CUDA 11.7용 PyTorch를 설치합니다."
    else
        print_warning "드라이버 버전이 낮습니다. CUDA 11.7용 PyTorch를 설치합니다."
        PYTORCH_CUDA_INDEX="cu117"
    fi
else
    print_warning "CPU 버전 PyTorch를 설치합니다."
    PYTORCH_CUDA_INDEX="cpu"
fi

# ============================================================================
# 2단계: 가상 환경 생성
# ============================================================================
print_header "2단계: 가상 환경 생성"

VENV_NAME="venv311"
if [ -d "$VENV_NAME" ]; then
    print_warning "가상 환경 '$VENV_NAME'이 이미 존재합니다."
    read -p "기존 가상 환경을 삭제하고 재생성하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_NAME"
        print_success "기존 가상 환경 삭제 완료"
    else
        print_warning "기존 가상 환경을 사용합니다."
        SKIP_VENV=true
    fi
fi

if [ "$SKIP_VENV" != true ]; then
    echo "가상 환경 생성 중: $VENV_NAME"
    $PYTHON_CMD -m venv "$VENV_NAME"
    print_success "가상 환경 생성 완료"
fi

# 가상 환경 활성화
echo "가상 환경 활성화 중..."
source "$VENV_NAME/bin/activate"
print_success "가상 환경 활성화 완료"

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip 업그레이드 완료"

# ============================================================================
# 3단계: PyTorch 설치
# ============================================================================
print_header "3단계: PyTorch 설치 (CUDA 지원)"

if [ "$PYTORCH_CUDA_INDEX" != "cpu" ]; then
    echo "PyTorch GPU 버전 설치 중 (CUDA $PYTORCH_CUDA_INDEX)..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA_INDEX"
else
    echo "PyTorch CPU 버전 설치 중..."
    pip install torch torchvision torchaudio
fi
print_success "PyTorch 설치 완료"

# PyTorch 설치 확인
echo ""
echo "PyTorch 설치 확인 중..."
PYTORCH_CHECK=$($PYTHON_CMD -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 2>&1)
PYTORCH_VER=$(echo "$PYTORCH_CHECK" | head -n 1)
CUDA_AVAILABLE=$(echo "$PYTORCH_CHECK" | tail -n 1)

print_success "PyTorch 버전: $PYTORCH_VER"
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1)
    CUDA_VER=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)" 2>&1)
    print_success "CUDA 사용 가능: True"
    echo "  - CUDA 버전: $CUDA_VER"
    echo "  - GPU 이름: $GPU_NAME"
else
    print_warning "CUDA 사용 가능: False (CPU 모드)"
fi

# ============================================================================
# 4단계: TensorFlow GPU 설치
# ============================================================================
print_header "4단계: TensorFlow GPU 설치"

echo "TensorFlow 설치 중..."
pip install tensorflow[and-cuda] 2>&1 | grep -v "WARNING" || true
print_success "TensorFlow 설치 완료"

# TensorFlow 설치 확인
echo ""
echo "TensorFlow 설치 확인 중..."
TF_CHECK=$($PYTHON_CMD -c "import tensorflow as tf; print(tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(len(gpus) > 0)" 2>&1)
TF_VER=$(echo "$TF_CHECK" | head -n 1)
TF_GPU_AVAILABLE=$(echo "$TF_CHECK" | tail -n 1)

print_success "TensorFlow 버전: $TF_VER"
if [ "$TF_GPU_AVAILABLE" = "True" ]; then
    print_success "TensorFlow GPU 사용 가능: True"
else
    print_warning "TensorFlow GPU 사용 가능: False"
fi

# ============================================================================
# 5단계: 기타 필수 패키지 설치
# ============================================================================
print_header "5단계: 기타 필수 패키지 설치"

if [ -f "requirements.txt" ]; then
    echo "requirements.txt를 사용하여 패키지 설치 중..."
    # PyTorch는 이미 설치했으므로 requirements.txt에서 제외
    pip install -r requirements.txt 2>&1 | grep -v "torch" || true
    print_success "requirements.txt 패키지 설치 완료"
else
    print_warning "requirements.txt를 찾을 수 없습니다. 기본 패키지만 설치합니다."
    echo "기본 패키지 설치 중..."
    pip install numpy Pillow matplotlib tqdm pymongo scikit-learn python-dotenv
    print_success "기본 패키지 설치 완료"
fi

# ============================================================================
# 6단계: 전체 설치 확인
# ============================================================================
print_header "6단계: 전체 설치 확인"

$PYTHON_CMD -c "
import sys
print('Python 버전:', sys.version.split()[0])
print()

# PyTorch 확인
try:
    import torch
    print('PyTorch 버전:', torch.__version__)
    print('CUDA 사용 가능:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA 버전:', torch.version.cuda)
        print('GPU 이름:', torch.cuda.get_device_name(0))
    print()
except ImportError:
    print('PyTorch: 설치되지 않음')
    print()

# TensorFlow 확인
try:
    import tensorflow as tf
    print('TensorFlow 버전:', tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print('GPU 사용 가능:', len(gpus) > 0)
    if gpus:
        print('GPU 목록:', [gpu.name for gpu in gpus])
    print()
except ImportError:
    print('TensorFlow: 설치되지 않음')
    print()

# 기타 패키지 확인
packages = ['numpy', 'PIL', 'matplotlib', 'tqdm', 'pymongo']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', '버전 정보 없음')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: 설치되지 않음')
"

# ============================================================================
# 설치 완료
# ============================================================================
print_header "설치 완료!"

echo "가상 환경 활성화 방법:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "가상 환경 비활성화:"
echo "  deactivate"
echo ""

print_success "모든 패키지 설치가 완료되었습니다!"

