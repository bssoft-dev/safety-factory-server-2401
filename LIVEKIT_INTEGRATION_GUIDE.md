# LiveKit 통합 가이드

이 문서는 기존 음성통신 시스템을 LiveKit으로 전환하는 방법을 설명합니다.

## 🎯 개요

LiveKit은 WebRTC 기반의 실시간 오디오/비디오 통신 플랫폼으로, 기존 WebSocket 기반 음성통신을 더 안정적이고 확장 가능한 방식으로 대체합니다.

## 📁 추가된 파일들

### 서비스 파일
- `services/livekit_voice_chat.py` - LiveKit 기반 음성통신 서비스
- `routers/livekit_chat.py` - LiveKit API 라우터

### 설정 파일
- `livekit.yaml` - LiveKit 서버 설정
- `docker-compose-livekit.yml` - LiveKit 서버 Docker Compose 설정

### 예제 파일
- `livekit_client_example.py` - LiveKit 클라이언트 테스트 예제

## 🚀 시작하기

### 1. LiveKit 서버 시작

```bash
# Docker Compose로 LiveKit 서버 시작
docker-compose -f docker-compose-livekit.yml up -d

# 서버 상태 확인
docker-compose -f docker-compose-livekit.yml ps
```

### 2. API 서버 시작

```bash
# 기존 FastAPI 서버 시작
python app.py
```

### 3. 클라이언트 테스트

```bash
# LiveKit 클라이언트 예제 실행
python livekit_client_example.py
```

## 📡 API 엔드포인트

### 방 관리
- `GET /api/livekit/rooms` - 방 목록 조회
- `POST /api/livekit/rooms` - 방 생성
- `DELETE /api/livekit/rooms` - 방 삭제

### 참가자 관리
- `POST /api/livekit/join` - 방 참가
- `GET /api/livekit/rooms/{room_name}/participants` - 참가자 목록 조회

### 토큰 관리
- `POST /api/livekit/token` - JWT 토큰 생성

### 설정 관리
- `GET /api/livekit/rooms/{room_name}/settings` - 방 설정 조회
- `PUT /api/livekit/rooms/{room_name}/settings` - 방 설정 업데이트

### 상태 확인
- `GET /api/livekit/health` - 서비스 상태 확인

## 🔧 설정

### LiveKit 서버 설정 (`livekit.yaml`)

```yaml
port: 7880                    # HTTP API 포트
rtc:
  udp_port: 50000            # RTP UDP 포트
  tcp_port: 7881             # WebSocket 포트
redis:
  address: redis:6379        # Redis 주소
keys:
  devkey: secret             # API 키와 시크릿
```

### API 서버 설정

LiveKit 서비스는 다음 설정으로 초기화됩니다:
- LiveKit URL: `ws://localhost:7880`
- API Key: `devkey`
- API Secret: `secret`

## 🎵 음성통신 기능

### 기존 기능 유지
- 음성 향상 (Voice Enhancement)
- STT (Speech-to-Text)
- 이벤트 분류
- 오디오 녹음
- 방별 설정 관리

### LiveKit 추가 기능
- WebRTC 기반 안정적인 연결
- 자동 재연결
- 네트워크 품질 모니터링
- 참가자 상태 실시간 추적

## 🔄 기존 시스템과의 차이점

### 기존 WebSocket 방식
```python
# WebSocket 직접 연결
websocket = await websocket.accept()
await websocket.send_bytes(audio_data)
```

### LiveKit 방식
```python
# LiveKit Room을 통한 연결
room = rtc.Room()
await room.connect(livekit_url, token)
await room.local_participant.publish_track(audio_track)
```

## 🧪 테스트

### 1. 서버 상태 확인
```bash
curl http://localhost:8000/api/livekit/health
```

### 2. 방 생성
```bash
curl -X POST http://localhost:8000/api/livekit/rooms \
  -H "Content-Type: application/json" \
  -d '{"room_name": "test-room"}'
```

### 3. 방 참가
```bash
curl -X POST http://localhost:8000/api/livekit/join \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "test-room",
    "participant_name": "Test User",
    "participant_identity": "test-user-001"
  }'
```

## 🐛 문제 해결

### 일반적인 문제들

#### 1. LiveKit 서버 연결 실패
```bash
# 서버 상태 확인
docker-compose -f docker-compose-livekit.yml ps

# 로그 확인
docker-compose -f docker-compose-livekit.yml logs livekit
```

#### 2. 포트 충돌
```bash
# 포트 사용 확인
netstat -tulpn | grep :7880
netstat -tulpn | grep :7881
```

#### 3. Redis 연결 오류
```bash
# Redis 상태 확인
docker-compose -f docker-compose-livekit.yml logs redis
```

## 🔐 보안 설정

### 개발 환경
현재 설정은 개발용으로 구성되어 있습니다:
- API 키: `devkey`
- API 시크릿: `secret`

### 프로덕션 환경
프로덕션 환경에서는 다음을 변경해야 합니다:

1. **강력한 API 키와 시크릿 사용**
2. **HTTPS 사용**
3. **방화벽 설정**
4. **토큰 만료 시간 조정**

## 📚 추가 리소스

- [LiveKit 공식 문서](https://docs.livekit.io/)
- [LiveKit Python SDK](https://docs.livekit.io/reference/python/)
- [LiveKit JavaScript SDK](https://docs.livekit.io/reference/client-sdk-js/)
- [WebRTC 문서](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)

## 🤝 기여하기

1. 이 저장소를 포크합니다
2. 새로운 기능 브랜치를 생성합니다 (`git checkout -b feature/livekit-enhancement`)
3. 변경사항을 커밋합니다 (`git commit -m 'Add LiveKit enhancement'`)
4. 브랜치에 푸시합니다 (`git push origin feature/livekit-enhancement`)
5. Pull Request를 생성합니다

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**LiveKit과 함께하는 안정적인 실시간 음성통신을 경험해보세요! 🎉**
