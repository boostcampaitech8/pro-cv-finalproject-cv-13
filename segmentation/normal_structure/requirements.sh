# 1. 가상환경 디렉토리 설정
python3.10 -m venv /data/ephemeral/home/testvenv --system-site-packages 

VENV_DIR="/data/ephemeral/home/testvenv"

# 2. 가상 환경 활성화
source $VENV_DIR/bin/activate

# 3. 임시 디렉토리 설정 (디스크 용량 문제 방지)
export TMPDIR=/data/ephemeral/tmp
export TEMP=/data/ephemeral/tmp
export TMP=/data/ephemeral/tmp
mkdir -p $TMPDIR

# 4. C 컴파일러 설치
echo "------nnUNetv2------"
apt update
apt install -y gcc

# 5. 파이썬 패키지 설치
pip install --upgrade pip
pip install --no-cache-dir torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd ./nnUNet
pip install -e .

echo "------TSV2------"

pip install --no-cache-dir TotalSegmentator
