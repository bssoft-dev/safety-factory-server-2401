# 라즈베리파이 디바이스 LiveKit 음성통신 가이드

이 가이드는 라즈베리파이 디바이스(작업반장용 단말)에서 기존 WebSocket 기반 음성통신을 LiveKit으로 전환하는 방법을 설명합니다.

## 📋 개요

현재 라즈베리파이 디바이스는 WebSocket을 통해 음성통신을 수행하고 있습니다. 이를 LiveKit 기반으로 전환하여 더 안정적이고 확장 가능한 음성통신을 구현합니다.

### 기존 시스템
- **통신 방식**: WebSocket (`wss://localhost:24015/ws/room`)
- **오디오 처리**: sounddevice + numpy
- **설정 파일**: `config.txt`

### LiveKit 시스템
- **통신 방식**: LiveKit WebRTC (`ws://localhost:7880`)
- **오디오 처리**: LiveKit Python SDK + sounddevice
- **인증**: JWT 토큰 기반

## 🚀 설치 및 설정

### 1. 라즈베리파이 환경 준비

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y python3 python3-pip python3-venv git

# 오디오 관련 패키지 설치
sudo apt install -y libglib2.0-dev libsndfile1 python3-dev portaudio19-dev
```

### 2. 프로젝트 클론 및 설정

```bash
# 프로젝트 클론
git clone https://github.com/bssoft-dev/safety-factory-sinknode-2401.git
cd safety-factory-sinknode-2401

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 기존 의존성 설치
pip install -r requirements.txt

# LiveKit 의존성 추가 설치
pip install livekit PyJWT aiohttp
```

### 3. 설정 파일 수정

기존 `config.txt` 파일을 LiveKit용으로 수정합니다:

```bash
# config.txt 파일 생성/수정
nano config.txt
```

다음 내용을 추가합니다:

```ini
# 기존 설정 (유지)
DEVICE_ID = "0"
IN_DEVICE_NUM = 0
OUT_DEVICE_NUM = 0
CHUNK = 2048*3
RATE = 16000*3
WARN_DISTANCE = 3
TARGET_BEACONS = ["yo:ur:ma:ca:dr:es"]

# LiveKit 설정 (추가)
room_name = "safety-room"
livekit_url = "ws://YOUR_SERVER_IP:7880"
api_base_url = "http://YOUR_SERVER_IP:24015"
participant_name = "RaspberryPi-001"
participant_identity = "raspberry-001"

# 기존 WebSocket 설정 (비활성화)
# ws_host = 'wss://localhost:24015/ws/room'
# host = 'https://localhost:24015'
```

**중요**: `YOUR_SERVER_IP`를 실제 서버 IP 주소로 변경하세요.

## 🔧 LiveKit 클라이언트 구현

### 1. LiveKit 오디오 클라이언트 생성

새로운 파일 `livekit_audio_client.py`를 생성합니다:

```python
import asyncio
import aiohttp
import sounddevice as sd
import numpy as np
import soundfile as sf
from collections import deque
import os
import jwt
from livekit import rtc
from utils.config import parse_config
from utils.setLogger import Logger

# 오디오 설정
CHANNELS = 1
DTYPE = np.int16
RECONNECT_DELAY = 3

class LiveKitAudioClient:
    def __init__(self, logger):
        config = parse_config()
        
        # LiveKit 설정
        self.livekit_url = config['livekit_url']
        self.api_base_url = config['api_base_url']
        self.room_name = config['room_name']
        self.participant_name = config['participant_name']
        self.participant_identity = config['participant_identity']
        
        # 오디오 설정
        self.RATE = config['RATE']
        self.CHUNK = config['CHUNK']
        self.IN_DEVICE_NUM = config['IN_DEVICE_NUM']
        self.OUT_DEVICE_NUM = config['OUT_DEVICE_NUM']
        self.BUFFER_SIZE = int(self.RATE / self.CHUNK) * 2
        
        # LiveKit 객체
        self.room = rtc.Room()
        self.local_audio_track = None
        
        # 오디오 스트림
        self.input_stream = None
        self.output_stream = None
        self.input_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.output_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # 상태 관리
        self.logger = logger
        self.is_running = True
        self.loop = asyncio.get_event_loop()
        
        # 알림음 로드
        self.dangerwav = sf.read(f'./wav/danger{self.RATE}.wav', dtype="int16")[0].reshape(-1, CHANNELS)
        self.disconnectwav = sf.read(f'./wav/disconnect{self.RATE}.wav', dtype="int16")[0].reshape(-1, CHANNELS)

    async def get_token(self):
        """API 서버에서 JWT 토큰 가져오기"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/api/livekit/join"
                data = {
                    "room_name": self.room_name,
                    "participant_name": self.participant_name,
                    "participant_identity": self.participant_identity
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"토큰 요청 실패: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"토큰 요청 오류: {e}")
            return None

    async def connect(self):
        """LiveKit 서버에 연결"""
        while self.is_running:
            try:
                self.logger.info("LiveKit 서버에 연결 시도 중...")
                
                # 토큰 가져오기
                token_info = await self.get_token()
                if not token_info:
                    self.logger.error("토큰 가져오기 실패")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue
                
                # LiveKit 방에 연결
                await self.room.connect(token_info["url"], token_info["token"])
                self.logger.info(f"LiveKit 서버에 연결되었습니다: {self.participant_name}")
                
                # 이벤트 핸들러 설정
                self.room.on("participant_connected", self.on_participant_connected)
                self.room.on("participant_disconnected", self.on_participant_disconnected)
                self.room.on("track_subscribed", self.on_track_subscribed)
                self.room.on("track_unsubscribed", self.on_track_unsubscribed)
                
                return True
                
            except Exception as e:
                self.logger.error(f"LiveKit 연결 실패: {e}")
                await asyncio.sleep(RECONNECT_DELAY)
        
        return False

    async def publish_microphone(self):
        """마이크 오디오 발행"""
        try:
            # LiveKit 오디오 소스 생성
            audio_source = rtc.AudioSource(self.RATE, CHANNELS)
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "microphone", audio_source
            )
            
            # 트랙 발행
            await self.room.local_participant.publish_track(
                self.local_audio_track, rtc.TrackPublishOptions()
            )
            self.logger.info("마이크 오디오가 발행되었습니다.")
            
        except Exception as e:
            self.logger.error(f"마이크 발행 오류: {e}")

    def input_callback(self, indata, frames, time, status):
        """마이크 입력 콜백"""
        try:
            if status:
                self.logger.warn(f"Input status: {status}")
            
            # LiveKit으로 오디오 데이터 전송
            if self.local_audio_track:
                audio_data = indata.flatten().astype(np.float32) / 32768.0
                self.local_audio_track.push_frame(audio_data)
            
            # 로컬 버퍼에도 저장 (필요시)
            self.input_buffer.append(indata.copy())
                
        except Exception as e:
            self.logger.warn(f"Error in input callback: {e}")

    def audio_callback(self, outdata, frames, time, status):
        """스피커 출력 콜백"""
        if status:
            self.logger.warn(f"Output status: {status}")
        
        try:
            if len(self.output_buffer) > 0:
                np_audio_data = self.output_buffer.popleft()
                if len(np_audio_data) < len(outdata):
                    outdata[:len(np_audio_data)] = np_audio_data
                    outdata[len(np_audio_data):] = np.zeros((len(outdata) - len(np_audio_data), CHANNELS), dtype=DTYPE)
                else:
                    outdata[:] = np_audio_data[:len(outdata)]
            else:
                outdata[:] = np.zeros((len(outdata), CHANNELS), dtype=DTYPE)
                
        except Exception as e:
            self.logger.warn(f"Error in audio callback: {e}")
            outdata[:] = np.zeros((len(outdata), CHANNELS), dtype=DTYPE)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 이벤트"""
        self.logger.info(f"참가자가 연결되었습니다: {participant.identity}")

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 해제 이벤트"""
        self.logger.info(f"참가자가 연결 해제되었습니다: {participant.identity}")

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 이벤트"""
        self.logger.info(f"트랙이 구독되었습니다: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self.handle_audio_track(track, participant)

    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 해제 이벤트"""
        self.logger.info(f"트랙 구독이 해제되었습니다: {track.kind} from {participant.identity}")

    def handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """오디오 트랙 처리"""
        self.logger.info(f"오디오 트랙을 처리합니다: {participant.identity}")
        
        @track.on("data_received")
        def on_audio_data(frame: rtc.AudioFrame):
            try:
                # 오디오 프레임을 numpy 배열로 변환
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                audio_array = audio_data.reshape(-1, CHANNELS)
                
                # 출력 버퍼에 추가
                self.write_audio(audio_array)
                
            except Exception as e:
                self.logger.warn(f"오디오 데이터 처리 오류: {e}")

    def write_audio(self, audio_array):
        """오디오 데이터를 출력 버퍼에 쓰기"""
        for i in range(0, len(audio_array), self.CHUNK):
            self.output_buffer.append(audio_array[i:i+self.CHUNK])

    def write_np_audio(self, audio_array, start_idx, write_len):
        """numpy 오디오 배열을 출력 버퍼에 쓰기"""
        if start_idx + write_len < audio_array.shape[0]:
            for i in range(0, write_len, self.CHUNK):
                self.output_buffer.append(audio_array[start_idx+i:start_idx+i+self.CHUNK])
            return False
        else:
            self.output_buffer.append(audio_array[start_idx:])
            return True
        
    async def play_np_audio(self, np_audio):
        """numpy 오디오 배열 재생"""
        play_idx = 0
        audio_end = False
        while not audio_end:
            play_len = self.CHUNK * self.BUFFER_SIZE // 2
            audio_end = self.write_np_audio(np_audio, play_idx, play_len)
            play_idx = play_idx + play_len
            while len(self.output_buffer) > 3:  # 버퍼 크기 충분할 때까지 대기
                await asyncio.sleep(0.001)

    async def danger_monitor(self):
        """위험 신호 모니터링"""
        while self.is_running:
            if os.path.exists("./danger.sig"):
                os.remove("./danger.sig")
                # 버퍼 클리어
                for _ in range(len(self.output_buffer)):
                    if self.output_buffer:
                        self.output_buffer.popleft()
                # 경보음 재생
                await self.play_np_audio(self.dangerwav)
            await asyncio.sleep(0.1)

    async def run(self):
        """메인 실행 함수"""
        if not await self.connect():
            return
        
        # 오디오 스트림 시작
        self.input_stream = sd.InputStream(
            device=self.IN_DEVICE_NUM,
            samplerate=self.RATE, blocksize=self.CHUNK, channels=CHANNELS,
            dtype=DTYPE, callback=self.input_callback
        )
        self.output_stream = sd.OutputStream(
            device=self.OUT_DEVICE_NUM,
            samplerate=self.RATE, blocksize=self.CHUNK, channels=CHANNELS,
            dtype=DTYPE, callback=self.audio_callback
        )
        
        self.input_stream.start()
        self.output_stream.start()
        
        # 마이크 발행
        await self.publish_microphone()
        
        # 태스크 실행
        await asyncio.gather(
            self.danger_monitor(),
        )
        
    def close(self):
        """리소스 정리"""
        self.logger.info("LiveKit Audio Client 종료")
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        if self.room:
            asyncio.create_task(self.room.disconnect())
```

### 2. 메인 파일 수정

`main.py` 파일을 수정하여 LiveKit 클라이언트를 사용하도록 합니다:

```python
import asyncio
from livekit_audio_client import LiveKitAudioClient  # 기존 AudioClient 대신
from utils.setLogger import Logger

async def main():
    logger = Logger(name='sink', logdir='./logs', level='debug')
    client = LiveKitAudioClient(logger)  # LiveKit 클라이언트 사용
    try:
        await client.run()
    except KeyboardInterrupt:
        print("클라이언트를 종료합니다.")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 테스트 및 실행

### 1. 서버 준비

먼저 서버에서 LiveKit 서버가 실행 중인지 확인합니다:

```bash
# 서버에서 실행
docker-compose -f docker-compose-livekit.yml up -d
python app.py
```

### 2. 라즈베리파이에서 테스트

```bash
# 라즈베리파이에서 실행
cd safety-factory-sinknode-2401
source venv/bin/activate
python main.py
```

### 3. 연결 확인

서버 로그에서 다음과 같은 메시지를 확인할 수 있습니다:

```
INFO:uvicorn.access:127.0.0.1:xxxxx - "POST /api/livekit/join HTTP/1.1" 200
```

라즈베리파이 로그에서:

```
[INFO] LiveKit 서버에 연결되었습니다: RaspberryPi-001
[INFO] 마이크 오디오가 발행되었습니다.
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 연결 실패
```bash
# 네트워크 연결 확인
ping YOUR_SERVER_IP

# 포트 확인
telnet YOUR_SERVER_IP 7880
telnet YOUR_SERVER_IP 24015
```

#### 2. 오디오 장치 문제
```bash
# 오디오 장치 목록 확인
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# 설정 파일의 IN_DEVICE_NUM, OUT_DEVICE_NUM 확인
```

#### 3. 권한 문제
```bash
# 오디오 그룹에 사용자 추가
sudo usermod -a -G audio $USER

# 재부팅 후 다시 시도
sudo reboot
```

#### 4. 의존성 문제
```bash
# 가상환경 재생성
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install livekit PyJWT aiohttp
```

## 📊 성능 최적화

### 1. 오디오 버퍼 크기 조정

`config.txt`에서 버퍼 크기를 조정할 수 있습니다:

```ini
# 낮은 지연시간 (더 많은 CPU 사용)
CHUNK = 1024
RATE = 16000

# 높은 안정성 (더 많은 지연시간)
CHUNK = 4096
RATE = 16000
```

### 2. 네트워크 최적화

```ini
# 로컬 네트워크 사용 시
livekit_url = "ws://192.168.1.100:7880"
api_base_url = "http://192.168.1.100:24015"

# 인터넷 사용 시 (더 안정적인 연결 필요)
livekit_url = "wss://your-domain.com:7880"
api_base_url = "https://your-domain.com:24015"
```

## 🔄 기존 시스템과의 호환성

### 점진적 전환

기존 WebSocket 시스템과 LiveKit 시스템을 동시에 지원할 수 있습니다:

1. **설정 파일에 모드 선택 추가**:
```ini
# config.txt
communication_mode = "livekit"  # "websocket" 또는 "livekit"
```

2. **메인 파일에서 모드에 따라 클라이언트 선택**:
```python
# main.py
if config.get('communication_mode') == 'livekit':
    from livekit_audio_client import LiveKitAudioClient
    client = LiveKitAudioClient(logger)
else:
    from audio_client import AudioClient
    client = AudioClient(logger)
```

## 📚 추가 리소스

- [LiveKit Python SDK 문서](https://docs.livekit.io/reference/python/)
- [LiveKit WebRTC 가이드](https://docs.livekit.io/guides/webrtc/)
- [라즈베리파이 오디오 설정](https://www.raspberrypi.org/documentation/configuration/audio-config.md)

## 🆘 지원

문제가 발생하거나 질문이 있으시면:

1. 로그 파일 확인: `./logs/` 디렉토리
2. 서버 상태 확인: `curl http://YOUR_SERVER_IP:24015/api/livekit/health`
3. 네트워크 연결 확인: `ping YOUR_SERVER_IP`

---

**LiveKit 기반 안정적인 음성통신을 경험해보세요! 🎵**
