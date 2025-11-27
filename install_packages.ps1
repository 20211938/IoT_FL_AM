# ============================================================================
# 필수 패키지 자동 설치 스크립트 (Windows PowerShell)
# IoT FLAM 프로젝트용
# ============================================================================

$ErrorActionPreference = "Stop"

# 색상 출력 함수
function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Header($Message) {
    Write-ColorOutput "Cyan" "================================================================================="
    Write-ColorOutput "Cyan" $Message
    Write-ColorOutput "Cyan" "================================================================================="
    Write-Output ""
}

function Write-Success($Message) {
    Write-ColorOutput "Green" "✓ $Message"
}

function Write-Warning($Message) {
    Write-ColorOutput "Yellow" "⚠ $Message"
}

function Write-Error($Message) {
    Write-ColorOutput "Red" "✗ $Message"
}

# ============================================================================
# 1단계: 시스템 환경 확인
# ============================================================================
Write-Header "1단계: 시스템 환경 확인"

# Python 버전 확인
Write-Output "Python 버전 확인 중..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $PYTHON_CMD = "python"
        Write-Success $pythonVersion
    } else {
        throw "Python을 찾을 수 없습니다."
    }
} catch {
    try {
        $pythonVersion = python3 --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PYTHON_CMD = "python3"
            Write-Success $pythonVersion
        } else {
            throw "Python을 찾을 수 없습니다."
        }
    } catch {
        Write-Error "Python을 찾을 수 없습니다. Python 3.8 이상을 설치해주세요."
        exit 1
    }
}

# Python 버전 체크
$versionMatch = $pythonVersion -match "(\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$matches[1]
    $minor = [int]$matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Error "Python 3.8 이상이 필요합니다. 현재 버전: $pythonVersion"
        exit 1
    }
}

# NVIDIA 드라이버 확인
Write-Output ""
Write-Output "NVIDIA 드라이버 확인 중..."
$HAS_NVIDIA = $false
$CUDA_VERSION = ""
$DRIVER_VERSION = ""

try {
    $nvidiaOutput = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        $HAS_NVIDIA = $true
        $driverMatch = $nvidiaOutput | Select-String "Driver Version:\s+(\d+\.\d+)"
        if ($driverMatch) {
            $DRIVER_VERSION = $driverMatch.Matches[0].Groups[1].Value
        }
        $cudaMatch = $nvidiaOutput | Select-String "CUDA Version:\s+(\d+\.\d+)"
        if ($cudaMatch) {
            $CUDA_VERSION = $cudaMatch.Matches[0].Groups[1].Value
        }
        Write-Success "NVIDIA 드라이버 발견"
        Write-Output "  - 드라이버 버전: $DRIVER_VERSION"
        Write-Output "  - 지원 CUDA 버전: $CUDA_VERSION"
    }
} catch {
    Write-Warning "nvidia-smi를 찾을 수 없습니다. CPU 버전으로 설치합니다."
}

# CUDA 버전 결정
$PYTORCH_CUDA_INDEX = ""
if ($HAS_NVIDIA) {
    $driverMajor = [int]($DRIVER_VERSION.Split('.')[0])
    if ($driverMajor -ge 535) {
        $PYTORCH_CUDA_INDEX = "cu121"
        Write-Success "CUDA 12.1용 PyTorch를 설치합니다."
    } elseif ($driverMajor -ge 520) {
        $PYTORCH_CUDA_INDEX = "cu118"
        Write-Success "CUDA 11.8용 PyTorch를 설치합니다."
    } elseif ($driverMajor -ge 470) {
        $PYTORCH_CUDA_INDEX = "cu117"
        Write-Success "CUDA 11.7용 PyTorch를 설치합니다."
    } else {
        Write-Warning "드라이버 버전이 낮습니다. CUDA 11.7용 PyTorch를 설치합니다."
        $PYTORCH_CUDA_INDEX = "cu117"
    }
} else {
    Write-Warning "CPU 버전 PyTorch를 설치합니다."
    $PYTORCH_CUDA_INDEX = "cpu"
}

# ============================================================================
# 2단계: 가상 환경 생성
# ============================================================================
Write-Header "2단계: 가상 환경 생성"

$VENV_NAME = "venv311"
$SKIP_VENV = $false

if (Test-Path $VENV_NAME) {
    Write-Warning "가상 환경 '$VENV_NAME'이 이미 존재합니다."
    $response = Read-Host "기존 가상 환경을 삭제하고 재생성하시겠습니까? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Path $VENV_NAME -Recurse -Force
        Write-Success "기존 가상 환경 삭제 완료"
    } else {
        Write-Warning "기존 가상 환경을 사용합니다."
        $SKIP_VENV = $true
    }
}

if (-not $SKIP_VENV) {
    Write-Output "가상 환경 생성 중: $VENV_NAME"
    & $PYTHON_CMD -m venv $VENV_NAME
    Write-Success "가상 환경 생성 완료"
}

# 가상 환경 활성화
Write-Output "가상 환경 활성화 중..."
& "$VENV_NAME\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Error "가상 환경 활성화 실패"
    exit 1
}
Write-Success "가상 환경 활성화 완료"

# pip 업그레이드
Write-Output ""
Write-Output "pip 업그레이드 중..."
python -m pip install --upgrade pip --quiet
Write-Success "pip 업그레이드 완료"

# ============================================================================
# 3단계: PyTorch 설치
# ============================================================================
Write-Header "3단계: PyTorch 설치 (CUDA 지원)"

if ($PYTORCH_CUDA_INDEX -ne "cpu") {
    Write-Output "PyTorch GPU 버전 설치 중 (CUDA $PYTORCH_CUDA_INDEX)..."
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA_INDEX"
} else {
    Write-Output "PyTorch CPU 버전 설치 중..."
    pip install torch torchvision torchaudio
}
Write-Success "PyTorch 설치 완료"

# PyTorch 설치 확인
Write-Output ""
Write-Output "PyTorch 설치 확인 중..."
$pytorchCheck = python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())" 2>&1
$pytorchLines = $pytorchCheck -split "`n"
$PYTORCH_VER = $pytorchLines[0]
$CUDA_AVAILABLE = $pytorchLines[-1]

Write-Success "PyTorch 버전: $PYTORCH_VER"
if ($CUDA_AVAILABLE -eq "True") {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>&1
    $cudaVer = python -c "import torch; print(torch.version.cuda)" 2>&1
    Write-Success "CUDA 사용 가능: True"
    Write-Output "  - CUDA 버전: $cudaVer"
    Write-Output "  - GPU 이름: $gpuName"
} else {
    Write-Warning "CUDA 사용 가능: False (CPU 모드)"
}

# ============================================================================
# 4단계: TensorFlow GPU 설치
# ============================================================================
Write-Header "4단계: TensorFlow GPU 설치"

Write-Output "TensorFlow 설치 중..."
pip install tensorflow[and-cuda] 2>&1 | Where-Object { $_ -notmatch "WARNING" }
Write-Success "TensorFlow 설치 완료"

# TensorFlow 설치 확인
Write-Output ""
Write-Output "TensorFlow 설치 확인 중..."
$tfCheck = python -c "import tensorflow as tf; print(tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print(len(gpus) > 0)" 2>&1
$tfLines = $tfCheck -split "`n"
$TF_VER = $tfLines[0]
$TF_GPU_AVAILABLE = $tfLines[-1]

Write-Success "TensorFlow 버전: $TF_VER"
if ($TF_GPU_AVAILABLE -eq "True") {
    Write-Success "TensorFlow GPU 사용 가능: True"
} else {
    Write-Warning "TensorFlow GPU 사용 가능: False"
}

# ============================================================================
# 5단계: 기타 필수 패키지 설치
# ============================================================================
Write-Header "5단계: 기타 필수 패키지 설치"

if (Test-Path "requirements.txt") {
    Write-Output "requirements.txt를 사용하여 패키지 설치 중..."
    pip install -r requirements.txt 2>&1 | Where-Object { $_ -notmatch "torch" }
    Write-Success "requirements.txt 패키지 설치 완료"
} else {
    Write-Warning "requirements.txt를 찾을 수 없습니다. 기본 패키지만 설치합니다."
    Write-Output "기본 패키지 설치 중..."
    pip install numpy Pillow matplotlib tqdm pymongo scikit-learn python-dotenv
    Write-Success "기본 패키지 설치 완료"
}

# ============================================================================
# 6단계: 전체 설치 확인
# ============================================================================
Write-Header "6단계: 전체 설치 확인"

python -c @"
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
"@

# ============================================================================
# 설치 완료
# ============================================================================
Write-Header "설치 완료!"

Write-Output "가상 환경 활성화 방법:"
Write-Output "  .\$VENV_NAME\Scripts\Activate.ps1"
Write-Output ""
Write-Output "가상 환경 비활성화:"
Write-Output "  deactivate"
Write-Output ""

Write-Success "모든 패키지 설치가 완료되었습니다!"

