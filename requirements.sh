# 원격 서버 이용 시

# 1. 가상환경 디렉토리 설정
python3.10 -m venv /data/ephemeral/home/testvenv --system-site-packages

VENV_DIR="/data/ephemeral/home/testvenv" # 경로 설정 필요

# 가상 환경 활성화
source $VENV_DIR/bin/activate

# 임시 디렉토리 설정 (디스크 용량 문제 방지)
export TMPDIR=/data/ephemeral/tmp
export TEMP=/data/ephemeral/tmp
export TMP=/data/ephemeral/tmp
mkdir -p $TMPDIR

# C 컴파일러 설치
apt update
apt install -y gcc

# 파이썬 패키지 설치
pip install --upgrade pip
pip install --no-cache-dir torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd ./nnUNet
pip install -e .

echo "------패키지 변동 사항------"

pip install --no-cache-dir TotalSegmentator
