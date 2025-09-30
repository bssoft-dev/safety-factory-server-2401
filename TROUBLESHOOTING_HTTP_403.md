# HTTP 403 오류 해결 가이드

## 🚨 문제 상황
```
2025-10-01 07:29:11,887[ERROR|audio_client.py,84] Connection failed: server rejected WebSocket connection: HTTP 403
```

## 🔍 문제 분석

### 1. 오류 원인
- **기존 WebSocket 클라이언트 실행**: `audio_client.py`가 여전히 실행 중
- **잘못된 연결 URL**: 기존 WebSocket URL로 연결 시도
- **서버 설정 문제**: WebSocket 엔드포인트 접근 권한 문제

### 2. 현재 상태 확인
- ✅ LiveKit 서버: 정상 실행 중 (포트 7880, 7881)
- ✅ API 서버: 정상 실행 중 (포트 24015)
- ✅ LiveKit API: 정상 작동 (토큰 생성 성공)
- ❌ 클라이언트: 기존 WebSocket 연결 시도 중

## 🛠️ 해결 방법

### 방법 1: 기존 프로세스 종료 및 LiveKit 클라이언트 실행

#### 1단계: 기존 프로세스 확인 및 종료
```bash
# 실행 중인 프로세스 확인
ps aux | grep python
ps aux | grep audio_client

# 기존 프로세스 종료
pkill -f "audio_client.py"
pkill -f "main.py"
```

#### 2단계: LiveKit 클라이언트 실행
```bash
# 라즈베리파이에서 실행
cd safety-factory-sinknode-2401
source venv/bin/activate

# LiveKit 클라이언트 실행 (새로운 파일)
python raspberry_pi_livekit_client.py
```

### 방법 2: 설정 파일 수정으로 통신 모드 변경

#### 1단계: config.txt 파일 수정
```bash
nano config.txt
```

다음 내용으로 수정:
```ini
# 통신 모드 변경
communication_mode = "livekit"

# LiveKit 설정 추가
room_name = "safety-room"
livekit_url = "wss://livekit.bs-soft.co.kr"
api_base_url = "https://safety-server.bs-soft.co.kr"
participant_name = "RaspberryPi-001"
participant_identity = "raspberry-001"

# 기존 WebSocket 설정 비활성화
# ws_host = 'wss://localhost:24015/ws/room'
# host = 'https://localhost:24015'
```

#### 2단계: main.py 파일 수정
`main.py` 파일에 모드 선택 로직 추가:

```python
import asyncio
from utils.config import parse_config
from utils.setLogger import Logger

async def main():
    logger = Logger(name='sink', logdir='./logs', level='debug')
    config = parse_config()
    
    # 통신 모드에 따라 클라이언트 선택
    if config.get('communication_mode') == 'livekit':
        from livekit_audio_client import LiveKitAudioClient
        client = LiveKitAudioClient(logger)
    else:
        from audio_client import AudioClient
        client = AudioClient(logger)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("클라이언트를 종료합니다.")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 방법 3: 서버 설정 확인 및 수정

#### 1단계: WebSocket 엔드포인트 확인
기존 WebSocket 엔드포인트가 여전히 활성화되어 있는지 확인:

```bash
# 서버에서 WebSocket 엔드포인트 확인
curl -I http://localhost:24015/ws/room/test-room/16000/int16/0
```

#### 2단계: CORS 설정 확인
`app.py`에서 CORS 설정이 올바른지 확인:

```python
origins = ["*"]  # 모든 오리진 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🧪 테스트 및 검증

### 1. LiveKit 연결 테스트
```bash
# 운영 서버에서 테스트
curl -k -X POST https://safety-server.bs-soft.co.kr/api/livekit/join \
  -H "Content-Type: application/json" \
  -d '{"room_name": "test-room", "participant_name": "Test User"}'
```

### 2. 클라이언트 연결 테스트
```bash
# 라즈베리파이에서 테스트
python simple_livekit_test.py
```

### 3. 네트워크 연결 확인
```bash
# 서버 도메인 확인
ping safety-server.bs-soft.co.kr
ping livekit.bs-soft.co.kr

# 포트 연결 확인
telnet safety-server.bs-soft.co.kr 443
telnet livekit.bs-soft.co.kr 7880
```

## 🔧 추가 문제 해결

### 문제 1: 토큰 생성 실패
```bash
# API 서버 상태 확인
curl http://YOUR_SERVER_IP:24015/api/livekit/health

# 토큰 생성 테스트
curl -X POST http://YOUR_SERVER_IP:24015/api/livekit/join \
  -H "Content-Type: application/json" \
  -d '{"room_name": "test-room", "participant_name": "Test User"}'
```

### 문제 2: LiveKit 서버 연결 실패
```bash
# LiveKit 서버 상태 확인
docker-compose -f docker-compose-livekit.yml ps

# LiveKit 서버 로그 확인
docker-compose -f docker-compose-livekit.yml logs livekit
```

### 문제 3: 오디오 장치 문제
```bash
# 오디오 장치 확인
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# 오디오 권한 확인
groups $USER
```

## 📋 체크리스트

### 서버 측 확인사항
- [ ] LiveKit 서버 실행 중 (포트 7880, 7881)
- [ ] API 서버 실행 중 (포트 24015)
- [ ] LiveKit API 엔드포인트 정상 작동
- [ ] JWT 토큰 생성 성공
- [ ] 네트워크 방화벽 설정 확인

### 클라이언트 측 확인사항
- [ ] 기존 WebSocket 프로세스 종료
- [ ] LiveKit 클라이언트 실행
- [ ] config.txt 설정 올바름
- [ ] 네트워크 연결 정상
- [ ] 오디오 장치 설정 정상

## 🚀 권장 해결 순서

1. **기존 프로세스 종료**
   ```bash
   pkill -f "audio_client.py"
   pkill -f "main.py"
   ```

2. **LiveKit 클라이언트 직접 실행**
   ```bash
   python raspberry_pi_livekit_client.py
   ```

3. **설정 파일 수정 후 재실행**
   ```bash
   # config.txt 수정
   nano config.txt
   
   # main.py 수정
   nano main.py
   
   # 재실행
   python main.py
   ```

4. **서비스 재시작 (필요시)**
   ```bash
   sudo systemctl restart safety-factory-livekit
   ```

## 📞 추가 지원

문제가 지속되면 다음 정보를 확인해주세요:

1. **서버 로그**: `docker-compose logs livekit`
2. **클라이언트 로그**: `./logs/` 디렉토리
3. **네트워크 설정**: `ip addr show`
4. **방화벽 상태**: `sudo ufw status`

---

**HTTP 403 오류는 대부분 기존 WebSocket 클라이언트가 실행 중이거나 잘못된 연결 URL 사용으로 발생합니다. 위 해결 방법을 순서대로 시도해보세요!** 🔧
