# 🎵 Safety Server 클라이언트 샘플

Safety Server와 실시간 오디오 스트리밍을 위한 다양한 클라이언트 구현

## 📁 파일 구성

| 파일 | 설명 | 언어/기술 |
|------|------|-----------|
| `python_client_basic.py` | 기본 Python 클라이언트 | Python + PyAudio |
| `python_client_advanced.py` | 고급 Python 클라이언트 | Python + 파일재생/녹음 |
| `web_client.html` | 웹 브라우저 클라이언트 | HTML5 + JavaScript |
| `nodejs_client.js` | Node.js 서버용 클라이언트 | Node.js + WebSocket |

## 🔧 설치 요구사항

### Python 클라이언트
```bash
pip install asyncio websockets numpy pyaudio aiohttp wave
```

### Node.js 클라이언트  
```bash
npm install ws node-fetch
```

### 웹 클라이언트
- 모던 웹 브라우저 (Chrome, Firefox, Safari, Edge)
- 마이크 접근 권한 필요

## 🚀 사용법

### 1. Python 기본 클라이언트
```bash
python python_client_basic.py
```

### 2. Python 고급 클라이언트
```bash
# 테스트 톤 모드 (기본)
python python_client_advanced.py --mode test --room 테스트방

# 라이브 마이크 모드
python python_client_advanced.py --mode live --device my-device

# 파일 재생 모드
python python_client_advanced.py --mode file --file audio.wav --room 파일테스트

# 원격 서버 연결
python python_client_advanced.py --host 192.168.1.100 --port 24015
```

### 3. 웹 클라이언트
```bash
# 웹 서버에서 호스팅하거나 직접 브라우저에서 열기
open web_client.html
```

### 4. Node.js 클라이언트
```bash
# 테스트 톤 30초 재생
node nodejs_client.js --mode test --duration 30

# WAV 파일 재생
node nodejs_client.js --mode file --file sample.wav

# 원격 서버 + 커스텀 설정
node nodejs_client.js --host 192.168.1.100 --room nodejs방 --device node-001
```

## 📊 Safety Server 연결 정보

### WebSocket URL 형식
```
ws://<host>:<port>/ws/room/<room_name>/<sample_rate>/<dtype>/<device_id>
```

### 파라미터 설명
- **host**: Safety Server IP 주소 (기본: localhost)
- **port**: Safety Server 포트 (기본: 24015)
- **room_name**: 방 이름 (미리 생성되어야 함)
- **sample_rate**: 샘플레이트 (16000 권장, 48000 지원)
- **dtype**: 데이터 타입 (`int16` 또는 `float32`)
- **device_id**: 고유 디바이스 식별자

### 방 생성 API
```bash
curl -X POST http://localhost:24015/v1/create_room \
  -H "Content-Type: application/json" \
  -d '{"room_name":"테스트방"}'
```

## 🎯 주요 기능

### ✅ 지원 기능
- [x] 실시간 양방향 오디오 스트리밍
- [x] 다중 클라이언트 동시 연결
- [x] 16kHz/48kHz 샘플레이트 지원
- [x] int16/float32 데이터 타입 자동 변환
- [x] 파일 재생 및 녹음
- [x] 테스트 톤 생성
- [x] 실시간 통계 모니터링
- [x] 자동 재연결 (일부 클라이언트)

### 🎵 오디오 설정
- **샘플레이트**: 16000Hz (권장), 48000Hz
- **채널**: 모노 (1채널)
- **비트깊이**: 16비트
- **프레임크기**: 1024 샘플 (64ms @ 16kHz)

## 🐛 문제 해결

### 연결 오류
1. Safety Server가 실행 중인지 확인
2. 방이 미리 생성되었는지 확인
3. 방화벽/포트 차단 확인

### 오디오 오류 (Python)
```bash
# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pyaudio
```

### 웹 클라이언트 마이크 오류
- HTTPS 환경에서 실행 권장
- 브라우저에서 마이크 권한 허용
- 디바이스 설정에서 기본 마이크 확인

## 📈 성능 최적화

### 지연 최소화
- 프레임 크기: 1024 샘플 (64ms)
- 버퍼링: 최소한으로 유지
- 네트워크: 유선 LAN 권장

### 품질 향상
- 16kHz보다 48kHz 사용 (대역폭 허용 시)
- 조용한 환경에서 테스트
- 마이크-스피커 거리 충분히 확보

## 🔧 개발자 정보

### 커스터마이징
각 클라이언트는 다음과 같이 확장 가능:
- 오디오 필터링/이퀄라이저
- 암호화/보안 강화
- 파일 포맷 지원 확장
- UI/UX 개선

### API 통합
Safety Server의 다른 API들:
- `/v1/rooms`: 방 목록 조회
- `/v1/option_settings`: 서버 설정 조회
- `/v1/events`: 이벤트 로그 조회

## 📝 라이센스

이 샘플 코드들은 Safety Server와 함께 사용하기 위한 예제입니다.
실제 운영 환경에서는 보안, 에러 처리, 로깅 등을 강화하여 사용하세요.

---
**🎵 Safety Server Client Samples - BS Soft**
