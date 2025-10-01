#!/bin/bash
# 라즈베리파이 LiveKit 설치 스크립트
# 이 스크립트는 라즈베리파이에서 LiveKit 기반 음성통신을 설정합니다.

set -e  # 오류 발생 시 스크립트 중단

echo "=========================================="
echo "라즈베리파이 LiveKit 설치 시작"
echo "=========================================="

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 시스템 업데이트
log_info "시스템 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 2. 필수 패키지 설치
log_info "필수 패키지 설치 중..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    curl \
    wget

# 3. 오디오 관련 패키지 설치
log_info "오디오 관련 패키지 설치 중..."
sudo apt install -y \
    libglib2.0-dev \
    libsndfile1 \
    portaudio19-dev \
    alsa-utils \
    pulseaudio

# 4. 오디오 그룹에 사용자 추가
log_info "오디오 권한 설정 중..."
sudo usermod -a -G audio $USER

# 5. 프로젝트 디렉토리 확인
if [ ! -d "safety-factory-sinknode-2401" ]; then
    log_info "프로젝트 클론 중..."
    git clone https://github.com/bssoft-dev/safety-factory-sinknode-2401.git
    cd safety-factory-sinknode-2401
else
    log_info "기존 프로젝트 디렉토리 사용"
    cd safety-factory-sinknode-2401
fi

# 6. 가상환경 생성 및 활성화
log_info "가상환경 설정 중..."
if [ -d "venv" ]; then
    log_warn "기존 가상환경 제거 중..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# 7. Python 패키지 설치
log_info "Python 패키지 설치 중..."
pip install --upgrade pip

# 기존 requirements.txt 설치
if [ -f "requirements.txt" ]; then
    log_info "기존 의존성 설치 중..."
    pip install -r requirements.txt
fi

# LiveKit 관련 패키지 설치
log_info "LiveKit 관련 패키지 설치 중..."
pip install \
    livekit==1.0.13 \
    PyJWT \
    aiohttp \
    sounddevice \
    numpy \
    soundfile \
    protobuf \
    aiofiles

# 8. 오디오 장치 확인
log_info "오디오 장치 확인 중..."
python3 -c "
import sounddevice as sd
print('사용 가능한 오디오 장치:')
print(sd.query_devices())
"

# 9. 설정 파일 생성
log_info "설정 파일 생성 중..."
if [ ! -f "config.txt" ]; then
    cat > config.txt << EOF
# 라즈베리파이 LiveKit 설정
DEVICE_ID = "0"
IN_DEVICE_NUM = 0
OUT_DEVICE_NUM = 0
CHUNK = 1024
RATE = 16000
WARN_DISTANCE = 3
TARGET_BEACONS = ["yo:ur:ma:ca:dr:es"]

# LiveKit 설정
communication_mode = "livekit"
room_name = "safety-room"
livekit_url = "wss://livekit.bs-soft.co.kr"
api_base_url = "https://safety-server.bs-soft.co.kr"
participant_name = "RaspberryPi-001"
participant_identity = "raspberry-001"
EOF
    log_info "config.txt 파일이 생성되었습니다. 실제 운영 서버 도메인이 설정되었습니다."
else
    log_info "기존 config.txt 파일이 있습니다."
fi

# 10. 서비스 파일 생성 (선택사항)
log_info "시스템 서비스 파일 생성 중..."
sudo tee /etc/systemd/system/safety-factory-livekit.service > /dev/null << EOF
[Unit]
Description=Safety Factory LiveKit Client
After=network.target sound.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 11. 권한 설정
log_info "권한 설정 중..."
chmod +x run.sh
chmod +x stop.sh

# 12. 설치 완료 메시지
echo ""
echo "=========================================="
log_info "LiveKit 설치가 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 단계를 수행하세요:"
echo ""
echo "1. config.txt 파일에서 서버 IP 주소를 수정하세요:"
echo "   nano config.txt"
echo ""
echo "2. 오디오 장치 번호를 확인하고 수정하세요:"
echo "   python3 -c \"import sounddevice as sd; print(sd.query_devices())\""
echo ""
echo "3. 테스트 실행:"
echo "   source venv/bin/activate"
echo "   python main.py"
echo ""
echo "4. 서비스로 등록 (선택사항):"
echo "   sudo systemctl enable safety-factory-livekit"
echo "   sudo systemctl start safety-factory-livekit"
echo ""
echo "5. 서비스 상태 확인:"
echo "   sudo systemctl status safety-factory-livekit"
echo ""
log_warn "설치 후 재부팅을 권장합니다: sudo reboot"
echo ""

# 13. 네트워크 연결 테스트 함수
test_connection() {
    local server_ip=$(grep "api_base_url" config.txt | cut -d'=' -f2 | tr -d ' ' | sed 's|http://||' | cut -d':' -f1)
    if [ "$server_ip" != "YOUR_SERVER_IP" ]; then
        log_info "서버 연결 테스트 중..."
        if ping -c 1 $server_ip > /dev/null 2>&1; then
            log_info "서버 연결 성공: $server_ip"
        else
            log_warn "서버 연결 실패: $server_ip"
        fi
    else
        log_warn "config.txt에서 서버 IP를 먼저 설정하세요"
    fi
}

# 연결 테스트 실행
test_connection

echo ""
echo "=========================================="
log_info "실제 운영 서버 도메인이 설정되었습니다!"
echo "=========================================="
echo ""
echo "설정된 서버 정보:"
echo "- LiveKit 서버: wss://livekit.bs-soft.co.kr:7880"
echo "- API 서버: https://safety-server.bs-soft.co.kr"
echo ""
echo "설치 스크립트가 완료되었습니다."
