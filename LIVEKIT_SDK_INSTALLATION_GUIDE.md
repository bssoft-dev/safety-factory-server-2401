# LiveKit Python SDK 설치 가이드

## 🎯 개요

디바이스(라즈베리파이, PC 등)에서 LiveKit에 연결하기 위해 필요한 Python SDK 설치 가이드입니다.

## 📦 필요한 패키지

### 1. LiveKit Python SDK
- **패키지명**: `livekit`
- **권장 버전**: `1.0.13`
- **용도**: LiveKit 서버에 참가자로 연결

### 2. 의존성 패키지
- **`protobuf`**: Protocol Buffers 지원
- **`aiofiles`**: 비동기 파일 처리
- **`numpy`**: 수치 계산
- **`aiohttp`**: HTTP 클라이언트
- **`PyJWT`**: JWT 토큰 처리
- **`sounddevice`**: 오디오 입출력
- **`soundfile`**: 오디오 파일 처리

## 🔧 설치 방법

### 방법 1: pip를 사용한 설치 (권장)

```bash
# LiveKit Python SDK 설치 (최신 버전)
pip install livekit==1.0.13

# 의존성 패키지 설치
pip install protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

### 방법 2: 한 번에 설치

```bash
pip install livekit==1.0.13 protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

### 방법 3: requirements.txt 사용

`requirements.txt` 파일 생성:
```txt
livekit==1.0.13
protobuf
aiofiles
numpy
aiohttp
PyJWT
sounddevice
soundfile
```

설치:
```bash
pip install -r requirements.txt
```

## 🐍 Python 버전 호환성

### 지원되는 Python 버전
- **Python 3.8+** (권장: Python 3.9 이상)
- **Python 3.11** (현재 서버 환경)

### 버전 확인
```bash
python --version
# 또는
python3 --version
```

## 🖥️ 플랫폼별 설치

### 라즈베리파이 (Raspberry Pi)

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# Python 및 pip 설치
sudo apt install python3 python3-pip python3-venv -y

# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# LiveKit SDK 설치
pip install livekit==1.0.13 protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

### Ubuntu/Debian

```bash
# Python 및 pip 설치
sudo apt install python3 python3-pip python3-venv -y

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# LiveKit SDK 설치
pip install livekit==1.0.13 protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

### Windows

```cmd
# Python 설치 (python.org에서 다운로드)
# pip는 Python과 함께 설치됨

# 가상환경 생성
python -m venv venv
venv\Scripts\activate

# LiveKit SDK 설치
pip install livekit==1.0.13 protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

### macOS

```bash
# Homebrew를 사용한 Python 설치
brew install python3

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# LiveKit SDK 설치
pip install livekit==1.0.13 protobuf aiofiles numpy aiohttp PyJWT sounddevice soundfile
```

## ✅ 설치 확인

### 1. 패키지 설치 확인
```bash
pip list | grep livekit
```

예상 출력:
```
livekit                   1.0.13
```

### 2. 버전 확인
```bash
pip show livekit
```

예상 출력:
```
Name: livekit
Version: 1.0.13
Summary: Python Real-time SDK for LiveKit
Home-page: https://github.com/livekit/python-sdks
License: Apache-2.0
```

### 3. Python에서 import 테스트
```python
# Python 인터프리터에서 테스트
python3 -c "import livekit; print('LiveKit SDK 설치 성공!')"
```

## 🔍 문제 해결

### 일반적인 문제

#### 1. `ModuleNotFoundError: No module named 'livekit'`
```bash
# 가상환경이 활성화되어 있는지 확인
which python
which pip

# 가상환경 활성화
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate     # Windows

# 다시 설치
pip install livekit==1.0.13
```

#### 2. `pip install` 권한 오류
```bash
# 사용자 디렉토리에 설치
pip install --user livekit==1.0.13

# 또는 가상환경 사용
python3 -m venv venv
source venv/bin/activate
pip install livekit==1.0.13
```

#### 3. 의존성 패키지 설치 오류
```bash
# 시스템 패키지 업데이트
sudo apt update  # Ubuntu/Debian
brew update      # macOS

# pip 업그레이드
pip install --upgrade pip

# 개별 패키지 설치
pip install protobuf
pip install aiofiles
pip install numpy
```

#### 4. 오디오 관련 패키지 설치 오류 (라즈베리파이)
```bash
# 시스템 오디오 라이브러리 설치
sudo apt install portaudio19-dev python3-pyaudio

# sounddevice 설치
pip install sounddevice
```

### 버전 호환성 문제

#### 1. Python 버전이 너무 낮음
```bash
# Python 3.8 이상 필요
python3 --version

# Python 업그레이드 (Ubuntu/Debian)
sudo apt install python3.9 python3.9-pip
```

#### 2. LiveKit 버전 불일치
```bash
# 특정 버전 설치
pip install livekit==1.0.13

# 기존 버전 제거 후 재설치
pip uninstall livekit
pip install livekit==1.0.13
```

## 🚀 빠른 시작

### 1. 기본 연결 테스트
```python
import asyncio
from livekit import rtc

async def test_connection():
    room = rtc.Room()
    
    # LiveKit 서버에 연결 (토큰 필요)
    await room.connect("wss://livekit.bs-soft.co.kr", "your_token_here")
    
    print("LiveKit 연결 성공!")
    
    # 연결 해제
    await room.disconnect()

# 테스트 실행
asyncio.run(test_connection())
```

### 2. 오디오 스트림 테스트
```python
import asyncio
import sounddevice as sd
from livekit import rtc

async def audio_test():
    room = rtc.Room()
    
    # 오디오 장치 확인
    print("사용 가능한 오디오 장치:")
    print(sd.query_devices())
    
    # LiveKit 연결
    await room.connect("wss://livekit.bs-soft.co.kr", "your_token_here")
    
    # 오디오 트랙 발행
    audio_track = rtc.LocalAudioTrack.create_audio_track("microphone")
    await room.local_participant.publish_track(audio_track)
    
    print("오디오 스트림 시작!")
    
    # 10초 대기
    await asyncio.sleep(10)
    
    await room.disconnect()

asyncio.run(audio_test())
```

## 📚 추가 리소스

### 공식 문서
- [LiveKit Python SDK 문서](https://docs.livekit.io/reference/python/)
- [LiveKit GitHub 저장소](https://github.com/livekit/python-sdks)
- [LiveKit WebRTC 가이드](https://docs.livekit.io/guides/webrtc/)

### 예제 코드
- [LiveKit Python 예제](https://github.com/livekit/python-sdks/tree/main/examples)
- [오디오 스트리밍 예제](https://github.com/livekit/python-sdks/tree/main/examples/audio)

### 커뮤니티
- [LiveKit Discord](https://discord.gg/livekit)
- [LiveKit 포럼](https://github.com/livekit/livekit/discussions)

## ⚠️ 주의사항

### 1. 버전 고정
- **권장**: `livekit==1.0.13` 사용
- **이유**: 현재 서버와 호환성 검증됨

### 2. 가상환경 사용
- **권장**: 프로젝트별 가상환경 사용
- **이유**: 패키지 충돌 방지

### 3. 의존성 관리
- **필수**: 모든 의존성 패키지 설치
- **확인**: `pip list`로 설치 확인

### 4. 네트워크 설정
- **방화벽**: LiveKit 서버 포트 열기
- **프록시**: 기업 환경에서 프록시 설정 확인

---

**이제 LiveKit Python SDK가 설치되어 디바이스에서 LiveKit 서버에 연결할 수 있습니다!** 🎵

- **버전**: `livekit==1.0.13` (권장)
- **호환성**: Python 3.8+ 지원
- **플랫폼**: Windows, macOS, Linux, Raspberry Pi 지원
- **기능**: 실시간 오디오/비디오 통신, WebRTC 기반
