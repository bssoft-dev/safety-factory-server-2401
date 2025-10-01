# LiveKit 방 상태 조회 가이드

## 🎯 개요

LiveKit 방의 상태 정보(접속자 수, 접속자 목록, 소음제거 설정 등)를 실시간으로 조회할 수 있는 API입니다.

## ⚠️ 중요 안내

**참가자 등록은 실제 LiveKit 연결 시점에만 발생합니다.**
- 토큰 생성만으로는 참가자로 등록되지 않습니다
- 클라이언트가 실제로 LiveKit 서버에 연결해야 방 상태에 반영됩니다
- 이는 실제 운영 환경에서 정확한 참가자 수를 추적하기 위함입니다

## 📋 API 엔드포인트

### 1. 특정 방 상태 조회
```http
GET /api/livekit/rooms/{room_name}/status
```

**응답 예시:**
```json
{
  "room_name": "safety-room",
  "total_participants": 3,
  "participants": [
    {
      "identity": "user-001",
      "name": "User1",
      "device_id": "device-001",
      "sample_rate": 16000,
      "data_type": "int16",
      "enhancement": {
        "enabled": true,
        "type": "light"
      },
      "is_connected": true,
      "audio_tracks": 1
    },
    {
      "identity": "user-002",
      "name": "User2",
      "device_id": "device-002",
      "sample_rate": 16000,
      "data_type": "int16",
      "enhancement": {
        "enabled": true,
        "type": "full"
      },
      "is_connected": true,
      "audio_tracks": 1
    }
  ],
  "is_active": true,
  "has_audio_buffers": true,
  "audio_buffer_count": 2
}
```

### 2. 전체 방 상태 조회
```http
GET /api/livekit/rooms/status
```

**응답 예시:**
```json
{
  "total_rooms": 2,
  "total_participants": 5,
  "rooms": {
    "safety-room": {
      "room_name": "safety-room",
      "total_participants": 3,
      "participants": [...],
      "is_active": true,
      "has_audio_buffers": true,
      "audio_buffer_count": 3
    },
    "test-room": {
      "room_name": "test-room",
      "total_participants": 2,
      "participants": [...],
      "is_active": true,
      "has_audio_buffers": true,
      "audio_buffer_count": 2
    }
  }
}
```

## 🔄 참가자 등록 프로세스

### 1. 토큰 생성 단계
```bash
# 참가자 토큰 생성
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/join \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "safety-room",
    "participant_name": "User1",
    "participant_identity": "user-001"
  }'
```

**이 시점에서는:**
- ✅ 토큰이 생성됨
- ✅ 클라이언트 정보가 저장됨
- ❌ 방 상태에 참가자로 등록되지 않음

### 2. 실제 연결 단계
```python
# 클라이언트에서 LiveKit 서버에 연결
await room.connect(token_info["url"], token_info["token"])
```

**이 시점에서:**
- ✅ 실제 LiveKit 연결 발생
- ✅ `participant_connected` 이벤트 발생
- ✅ 방 상태에 참가자로 등록됨
- ✅ 소음제거 스트리머 초기화됨

### 3. 연결 해제 단계
```python
# 클라이언트에서 연결 해제
await room.disconnect()
```

**이 시점에서:**
- ✅ `participant_disconnected` 이벤트 발생
- ✅ 방 상태에서 참가자 제거됨
- ✅ 소음제거 스트리머 정리됨

## 🧪 테스트 시나리오

### 시나리오 1: 토큰 생성만으로는 참가자 등록 안됨
```bash
# 1. 방 생성
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/rooms \
  -H "Content-Type: application/json" \
  -d '{"room_name": "test-room"}'

# 2. 토큰 생성
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/join \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "test-room",
    "participant_name": "TestUser",
    "participant_identity": "test-user-001"
  }'

# 3. 방 상태 조회 (여전히 빈 방)
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/test-room/status
# 결과: total_participants: 0
```

### 시나리오 2: 실제 연결 후 참가자 등록됨
```python
# 클라이언트 코드에서 실제 연결
import asyncio
from livekit import rtc

async def connect_to_room():
    room = rtc.Room()
    
    # 토큰으로 연결
    await room.connect("wss://livekit.bs-soft.co.kr", token)
    
    # 이 시점에서 방 상태에 참가자로 등록됨
    print("연결 완료 - 방 상태에 반영됨")

asyncio.run(connect_to_room())
```

## 📊 응답 필드 설명

### 방 정보
- `room_name`: 방 이름
- `total_participants`: 실제 연결된 참가자 수
- `is_active`: 방 활성 상태
- `has_audio_buffers`: 오디오 버퍼 존재 여부
- `audio_buffer_count`: 오디오 버퍼 개수

### 참가자 정보
- `identity`: 참가자 고유 식별자
- `name`: 참가자 이름
- `device_id`: 디바이스 ID (선택사항)
- `sample_rate`: 오디오 샘플레이트
- `data_type`: 오디오 데이터 타입
- `enhancement`: 소음제거 설정
  - `enabled`: 소음제거 활성화 여부
  - `type`: 소음제거 타입 ("light" 또는 "full")
- `is_connected`: 연결 상태
- `audio_tracks`: 오디오 트랙 개수

## 🔧 사용 예시

### Python 클라이언트 예시
```python
import aiohttp
import asyncio

async def monitor_room_status(room_name):
    """방 상태 모니터링"""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    f"https://safety-server.bs-soft.co.kr/api/livekit/rooms/{room_name}/status"
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"방: {data['room_name']}")
                        print(f"참가자 수: {data['total_participants']}")
                        
                        for participant in data['participants']:
                            print(f"  - {participant['name']} ({participant['identity']})")
                            print(f"    소음제거: {participant['enhancement']['type']}")
                    else:
                        print(f"상태 조회 실패: {response.status}")
                        
            except Exception as e:
                print(f"오류: {e}")
            
            await asyncio.sleep(5)  # 5초마다 조회

# 사용
asyncio.run(monitor_room_status("safety-room"))
```

### cURL 예시
```bash
# 특정 방 상태 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/safety-room/status

# 전체 방 상태 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/status

# JSON 형태로 보기 좋게 출력
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/status | jq .
```

## ⚠️ 주의사항

### 1. 실제 연결과 토큰 생성의 차이
- **토큰 생성**: 클라이언트가 연결할 수 있는 권한 부여
- **실제 연결**: LiveKit 서버와의 WebRTC 연결 수립
- 방 상태는 **실제 연결**된 참가자만 반영

### 2. 연결 상태 추적
- 참가자가 네트워크 문제로 일시적으로 연결이 끊어져도 즉시 반영됨
- 재연결 시 자동으로 다시 등록됨
- 소음제거 스트리머도 연결/해제에 따라 자동 관리됨

### 3. 성능 고려사항
- 방 상태 조회는 메모리 기반으로 빠름
- 대량의 방과 참가자가 있어도 실시간 조회 가능
- 서버 재시작 시 메모리 기반 상태는 초기화됨

## 🚀 실시간 모니터링

### 웹 대시보드 예시
```html
<!DOCTYPE html>
<html>
<head>
    <title>LiveKit 방 상태 모니터</title>
    <script>
        async function updateRoomStatus() {
            try {
                const response = await fetch('/api/livekit/rooms/status');
                const data = await response.json();
                
                document.getElementById('total-rooms').textContent = data.total_rooms;
                document.getElementById('total-participants').textContent = data.total_participants;
                
                const roomsList = document.getElementById('rooms-list');
                roomsList.innerHTML = '';
                
                for (const [roomName, roomInfo] of Object.entries(data.rooms)) {
                    const roomDiv = document.createElement('div');
                    roomDiv.innerHTML = `
                        <h3>${roomName}</h3>
                        <p>참가자: ${roomInfo.total_participants}명</p>
                        <ul>
                            ${roomInfo.participants.map(p => 
                                `<li>${p.name} (${p.identity}) - 소음제거: ${p.enhancement.type}</li>`
                            ).join('')}
                        </ul>
                    `;
                    roomsList.appendChild(roomDiv);
                }
            } catch (error) {
                console.error('상태 업데이트 실패:', error);
            }
        }
        
        // 5초마다 상태 업데이트
        setInterval(updateRoomStatus, 5000);
        updateRoomStatus(); // 초기 로드
    </script>
</head>
<body>
    <h1>LiveKit 방 상태 모니터</h1>
    <p>총 방 수: <span id="total-rooms">0</span></p>
    <p>총 참가자 수: <span id="total-participants">0</span></p>
    <div id="rooms-list"></div>
</body>
</html>
```

## 🔍 문제 해결

### 일반적인 문제

1. **토큰 생성했는데 방 상태에 참가자가 없음**
   - 정상 동작입니다. 실제 LiveKit 연결이 필요합니다
   - 클라이언트에서 `room.connect()` 호출 확인

2. **참가자가 연결했는데 상태에 반영 안됨**
   - LiveKit 서버 연결 상태 확인
   - 네트워크 연결 상태 확인
   - 서버 로그에서 `participant_connected` 이벤트 확인

3. **참가자 수가 부정확함**
   - 네트워크 불안정으로 인한 일시적 연결 해제
   - 클라이언트 재연결 시 자동으로 복구됨
   - 서버 재시작 시 메모리 기반 상태 초기화됨

### 디버깅

```bash
# 서버 로그에서 참가자 연결 이벤트 확인
tail -f log.txt | grep "참가자"

# 방 상태 실시간 모니터링
watch -n 1 'curl -s https://safety-server.bs-soft.co.kr/api/livekit/rooms/status | jq .'
```

---

**이제 LiveKit 방의 실제 연결 상태를 정확하게 추적할 수 있습니다!** 🎵
