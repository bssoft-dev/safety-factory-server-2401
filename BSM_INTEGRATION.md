# Safety-Server ↔ BSM 연동 가이드

## 개요

이 문서는 Safety-Server가 BSM(BS Studio Manager)과 연동하여 '음성 처리 허브' 역할을 수행하는 방법을 설명합니다.

## 아키텍처

```
[BSM] ←→ [Safety-Server] ←→ [디바이스들]
  │           │                    │
  └─ 관리     └─ 음성분석           └─ 음성스트림
```

- **BSM**: 오케스트레이터 역할, 디바이스 관리 및 이벤트 처리
- **Safety-Server**: 음성 분석 전문 서비스 (이벤트 감지, STT)
- **디바이스**: 음성 데이터 송신

## 설정 방법

### 1. 환경 변수 설정

`env.py` 파일에 다음 설정을 추가하세요:

```python
# BSM 연동 설정
BSM_URL = 'https://bsm.bs-soft.co.kr'
BSM_API_KEY = 'your-bsm-api-key-here'  # BSM에서 발급받은 API 키
SAFETY_SERVER_ID = 'safety-server-main'

# 선택적: IP 화이트리스트 (환경변수)
ALLOWED_BSM_IPS = 'BSM서버IP1,BSM서버IP2'
```

### 2. BSM 전용 API 엔드포인트

Safety-Server는 BSM과의 연동을 위한 전용 API를 제공합니다:

#### 방 생성 (BSM → Safety-Server)
```http
POST /api/v1/bsm/create-room
Authorization: Bearer your-api-key
Content-Type: application/json

{
  "room_name": "bsm-room-{device_id}-{uuid}",
  "device_id": "device-001",
  "audio_settings": {
    "sample_rate": 16000,
    "channels": 1
  }
}
```

**응답:**
```json
{
  "status": "success",
  "message": "Room created successfully",
  "data": {
    "room_name": "bsm-room-device-001-abc123",
    "websocket_url": "ws://safety-server.bs-soft.co.kr:24015/ws/room/bsm-room-device-001-abc123/16000/int16/device-001",
    "device_id": "device-001"
  }
}
```

#### 방 삭제
```http
DELETE /api/v1/bsm/room/{room_name}
Authorization: Bearer your-api-key
```

#### 헬스체크
```http
GET /api/v1/bsm/health
```

#### 방 상태 조회
```http
GET /api/v1/bsm/rooms/status
Authorization: Bearer your-api-key
```

### 3. 이벤트 보고 (Safety-Server → BSM)

Safety-Server는 다음 이벤트들을 BSM에 자동으로 보고합니다:

```http
POST /api/v1/private/ingress/safety-server
Authorization: Bearer bsm-api-key
Content-Type: application/json

{
  "device_id": "device-001",
  "room_name": "bsm-room-device-001-abc123",
  "event_type": "audio_analysis",
  "event_name": "scream_detected" | "alarm_detected" | "shock_detected" | "stt_result",
  "severity": "critical" | "info",
  "payload": {
    "confidence": 0.95,
    "description": "비명 소리가 감지되었습니다",
    "audio_file_path": "path/to/audio.wav"
  },
  "occurred_at": "2024-01-15T10:30:00Z"
}
```

## 연동 흐름

### 1. 디바이스 음성 분석 활성화

1. **BSM**: 디바이스에 음성 분석 기능 할당
2. **BSM**: Safety-Server에 방 생성 요청
   ```bash
   curl -X POST https://safety-server.bs-soft.co.kr/api/v1/bsm/create-room \
     -H "Authorization: Bearer $BSM_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"room_name": "bsm-room-device001-uuid123", "device_id": "device001"}'
   ```
3. **BSM**: 응답받은 WebSocket URL을 디바이스 설정에 저장
4. **BSM**: Supabase Realtime으로 디바이스에 설정 전달

### 2. 디바이스 음성 스트리밍

1. **디바이스**: BSM으로부터 새로운 `audio_processing` 설정 수신
2. **디바이스**: WebSocket 연결 수립
   ```javascript
   const ws = new WebSocket('ws://safety-server.bs-soft.co.kr:24015/ws/room/bsm-room-device001-uuid123/16000/int16/device001');
   ```
3. **디바이스**: 마이크 데이터를 실시간으로 전송

### 3. 이벤트 감지 및 보고

1. **Safety-Server**: 음성 분석 수행 (이벤트 분류, STT)
2. **Safety-Server**: 이벤트 감지 시 BSM에 자동 보고
3. **BSM**: 이벤트 수신 후 `device_events` 테이블에 저장
4. **BSM**: 파이프라인 트리거 및 후속 처리

## 보안 고려사항

### API 키 인증
- BSM과 Safety-Server 간 통신은 Bearer 토큰으로 인증
- `BSM_API_KEY` 환경변수 설정 필수

### IP 화이트리스트
```bash
export ALLOWED_BSM_IPS="192.168.1.100,10.0.0.50"
```

### 방 이름 검증
- BSM이 생성하는 방은 반드시 `bsm-room-` 접두사 사용
- 무분별한 방 생성 방지

## 테스트 방법

### 1. 연결 테스트
```bash
curl -X POST https://safety-server.bs-soft.co.kr/api/v1/bsm/test-connection \
  -H "Authorization: Bearer $BSM_API_KEY"
```

### 2. 헬스체크
```bash
curl https://safety-server.bs-soft.co.kr/api/v1/bsm/health
```

### 3. 설정 확인
```bash
curl https://safety-server.bs-soft.co.kr/api/v1/bsm/config
```

## 모니터링

### Safety-Server 로그
- BSM 이벤트 보고: `async.log`에 기록
- API 호출 로그: uvicorn 로그에 기록

### BSM 측 확인사항
- `external_apis` 테이블에서 Safety-Server health_status 모니터링
- `device_events` 테이블에서 수신된 이벤트 확인

## 문제 해결

### 1. BSM 이벤트 보고 실패
```bash
# 환경변수 확인
echo $BSM_URL
echo $BSM_API_KEY

# BSM 서버 연결 테스트
curl -I $BSM_URL/api/v1/health
```

### 2. 방 생성 실패
- API 키 확인
- 방 이름 패턴 확인 (`bsm-room-` 접두사)
- IP 화이트리스트 설정 확인

### 3. 음성 스트리밍 연결 실패
- WebSocket URL 정확성 확인
- 방이 사전에 생성되었는지 확인
- 네트워크 방화벽 설정 확인

## 코드 구조

```
services/
├── bsm_api_agent.py      # BSM과의 HTTP 통신 담당
├── event_classification.py  # 이벤트 감지 + BSM 보고
├── stt.py               # STT 처리 + BSM 보고
└── voice_chat.py        # 기존 음성 채팅 기능

routers/
├── bsm_integration.py   # BSM 전용 API 엔드포인트
├── chat.py             # 기존 채팅 API
└── log_mon.py          # 로그 모니터링

app.py                  # 메인 FastAPI 앱 (BSM 라우터 추가)
```

## 업그레이드 경로

이 연동 구조는 기존 코드를 전혀 수정하지 않고 추가된 것이므로:

1. **기존 기능 유지**: 레거시 클라이언트들은 기존 방식대로 동작
2. **점진적 마이그레이션**: 디바이스별로 순차적으로 BSM 연동 활성화 가능
3. **롤백 용이**: BSM 연동 비활성화 시 기존 방식으로 즉시 복귀 가능

## 향후 확장 계획

1. **실시간 상태 동기화**: Safety-Server → BSM 주기적 상태 보고
2. **고급 이벤트 필터링**: BSM에서 설정한 조건에 따른 선택적 이벤트 보고
3. **다중 Safety-Server 지원**: BSM이 여러 Safety-Server 인스턴스 관리 