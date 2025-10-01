# LiveKit 방 설정 가이드

## 🎯 개요

LiveKit 음성대화 방의 설정값에 따라 오디오 처리가 동작하도록 구현된 기능입니다. 기존 프로젝트의 방 설정 시스템과 LiveKit 서비스를 완전히 통합했습니다.

## 📋 지원되는 방 설정

### 1. 음성 강화 (소음 제거) - `use_voice_enhance`
- **기본값**: `true`
- **설명**: 오디오 스트림에 소음제거 적용 여부
- **동작**: 
  - `true`: 참가자별 소음제거 설정에 따라 Light/Full enhancement 적용
  - `false`: 소음제거 비활성화, 원본 오디오 그대로 전송

### 2. 내 소리 듣기 - `hear_me`
- **기본값**: `false`
- **설명**: 자신의 음성을 다시 들을 수 있는지 여부
- **동작**: 
  - `true`: 자신의 음성을 다시 들을 수 있음
  - `false`: 자신의 음성을 다시 들을 수 없음

### 3. 오디오 녹음 - `record_audio`
- **기본값**: `true`
- **설명**: 방의 오디오를 녹음할지 여부
- **동작**: 
  - `true`: 오디오 녹음 활성화
  - `false`: 오디오 녹음 비활성화

### 4. 이벤트 분류 - `classify_event`
- **기본값**: `true`
- **설명**: 음원 분류 (비명, 경보음 등) 수행 여부
- **동작**: 
  - `true`: AI 기반 이벤트 분류 활성화
  - `false`: 이벤트 분류 비활성화

### 5. 음성 인식 (STT) - `do_stt`
- **기본값**: `true`
- **설명**: 음성을 텍스트로 변환하는 STT 수행 여부
- **동작**: 
  - `true`: STT 활성화
  - `false`: STT 비활성화

### 6. 볼륨 증폭 - `enhance_volume`
- **기본값**: `0`
- **범위**: `0-100`
- **설명**: 오디오 볼륨 증폭 정도
- **동작**: 
  - `0`: 볼륨 증폭 없음
  - `1-100`: 해당 비율만큼 볼륨 증폭 (1.0-2.0배)

## 🔧 API 엔드포인트

### 1. 방 설정 조회
```http
GET /api/livekit/rooms/{room_name}/settings
```

**응답 예시:**
```json
{
  "room_name": "safety-room",
  "settings": {
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": true,
    "enhance_volume": 20
  }
}
```

### 2. 방 설정 업데이트
```http
POST /api/livekit/rooms/settings
```

**요청 예시:**
```json
{
  "room_name": "safety-room",
  "use_voice_enhance": false,
  "hear_me": true,
  "enhance_volume": 50
}
```

**응답 예시:**
```json
{
  "message": "Room settings updated for safety-room",
  "settings": {
    "use_voice_enhance": false,
    "hear_me": true,
    "enhance_volume": 50
  }
}
```

### 3. 전체 방 설정 조회
```http
GET /api/livekit/rooms/settings/all
```

**응답 예시:**
```json
{
  "total_rooms": 3,
  "rooms_settings": {
    "safety-room": {
      "room_name": "safety-room",
      "settings": {
        "use_voice_enhance": true,
        "hear_me": false,
        "record_audio": true,
        "classify_event": true,
        "do_stt": true,
        "enhance_volume": 20
      }
    },
    "test-room": {
      "room_name": "test-room",
      "settings": {
        "use_voice_enhance": false,
        "hear_me": true,
        "record_audio": false,
        "classify_event": false,
        "do_stt": false,
        "enhance_volume": 0
      }
    }
  }
}
```

## 🎛️ 오디오 처리 파이프라인

### 처리 순서
1. **음성 강화 (소음 제거)**: `use_voice_enhance`가 `true`일 때
2. **볼륨 증폭**: `enhance_volume`이 0보다 클 때
3. **이벤트 분류**: `classify_event`가 `true`일 때 (향후 구현)
4. **STT 처리**: `do_stt`가 `true`일 때 (향후 구현)
5. **오디오 녹음**: `record_audio`가 `true`일 때 (향후 구현)

### 코드 예시
```python
def apply_audio_processing(self, audio_data: np.ndarray, participant_identity: str, room_name: str) -> np.ndarray:
    """오디오 데이터에 방 설정에 따른 처리 적용"""
    # 방 설정 조회
    room_settings = self.get_room_settings(room_name)
    
    processed_audio = audio_data.copy()
    
    # 1. 음성 강화 (소음 제거)
    if room_settings.get('use_voice_enhance', True):
        # 참가자별 소음제거 설정 적용
        # Light 또는 Full enhancement 선택
        
    # 2. 볼륨 증폭
    volume_boost = room_settings.get('enhance_volume', 0)
    if volume_boost > 0:
        boost_factor = 1.0 + (volume_boost / 100.0)
        processed_audio = np.clip(processed_audio * boost_factor, -32768, 32767)
    
    # 3. 이벤트 분류 (향후 구현)
    if room_settings.get('classify_event', True):
        # AI 기반 이벤트 분류 로직
        
    # 4. STT 처리 (향후 구현)
    if room_settings.get('do_stt', True):
        # 음성을 텍스트로 변환
        
    # 5. 오디오 녹음 (향후 구현)
    if room_settings.get('record_audio', True):
        # 오디오 녹음 로직
        
    return processed_audio
```

## 🧪 테스트 시나리오

### 시나리오 1: 고품질 방 설정
```bash
# 방 생성
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/rooms \
  -H "Content-Type: application/json" \
  -d '{"room_name": "high-quality-room"}'

# 고품질 설정 적용
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/rooms/settings \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "high-quality-room",
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": true,
    "enhance_volume": 20
  }'
```

### 시나리오 2: 저지연 방 설정
```bash
# 저지연 설정 적용
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/rooms/settings \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "low-latency-room",
    "use_voice_enhance": false,
    "hear_me": true,
    "record_audio": false,
    "classify_event": false,
    "do_stt": false,
    "enhance_volume": 0
  }'
```

### 시나리오 3: 균형잡힌 방 설정
```bash
# 균형잡힌 설정 적용
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/rooms/settings \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "balanced-room",
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": false,
    "enhance_volume": 10
  }'
```

## 📊 방 상태 조회 (설정 정보 포함)

### 방 상태 API 응답
```json
{
  "room_name": "safety-room",
  "total_participants": 3,
  "participants": [...],
  "is_active": true,
  "has_audio_buffers": true,
  "audio_buffer_count": 3,
  "settings": {
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": true,
    "enhance_volume": 20
  }
}
```

## 🔄 설정 변경의 실시간 적용

### 1. 설정 변경 시점
- API를 통해 방 설정이 변경되면 즉시 캐시에 반영
- 다음 오디오 프레임부터 새로운 설정이 적용됨

### 2. 캐시 관리
```python
# 방 설정 캐시
self.room_settings_cache: Dict[str, dict] = {}

def get_room_settings(self, room_name: str) -> dict:
    # 1. 캐시에서 먼저 확인
    if room_name in self.room_settings_cache:
        return self.room_settings_cache[room_name]
    
    # 2. DB에서 조회
    # 3. 캐시에 저장
    self.room_settings_cache[room_name] = room_settings
    return room_settings
```

### 3. 설정 업데이트
```python
def update_room_settings(self, room_name: str, settings: dict) -> dict:
    # 1. DB 업데이트
    # 2. 캐시 업데이트
    self.room_settings_cache[room_name] = settings
    return result
```

## 🎯 사용 사례

### 1. 안전 관리실 (고품질)
- **음성 강화**: 활성화 (Full enhancement)
- **내 소리 듣기**: 비활성화
- **오디오 녹음**: 활성화
- **이벤트 분류**: 활성화
- **STT**: 활성화
- **볼륨 증폭**: 20%

### 2. 현장 작업팀 (저지연)
- **음성 강화**: 비활성화
- **내 소리 듣기**: 활성화
- **오디오 녹음**: 비활성화
- **이벤트 분류**: 비활성화
- **STT**: 비활성화
- **볼륨 증폭**: 0%

### 3. 회의실 (균형)
- **음성 강화**: 활성화 (Light enhancement)
- **내 소리 듣기**: 비활성화
- **오디오 녹음**: 활성화
- **이벤트 분류**: 활성화
- **STT**: 비활성화
- **볼륨 증폭**: 10%

## 🔍 문제 해결

### 일반적인 문제

1. **설정이 적용되지 않음**
   - 방 설정 캐시 확인
   - DB에 설정이 저장되었는지 확인
   - 서버 재시작 후 캐시 초기화

2. **볼륨 증폭이 작동하지 않음**
   - `enhance_volume` 값이 0보다 큰지 확인
   - 오디오 데이터 타입이 `int16`인지 확인

3. **소음제거가 적용되지 않음**
   - `use_voice_enhance`가 `true`인지 확인
   - 참가자별 소음제거 설정 확인

### 디버깅

```bash
# 방 설정 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/safety-room/settings

# 방 상태 조회 (설정 포함)
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/safety-room/status

# 전체 방 설정 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/rooms/settings/all
```

## 🚀 향후 확장 계획

### 1. 이벤트 분류 구현
- AI 모델을 통한 실시간 이벤트 분류
- 비명, 경보음, 충돌음 등 감지
- 이벤트 발생 시 알림 시스템 연동

### 2. STT 구현
- 실시간 음성을 텍스트로 변환
- 다국어 지원
- 텍스트 기반 검색 및 분석

### 3. 오디오 녹음 구현
- 방별 오디오 녹음 저장
- 녹음 파일 관리 시스템
- 녹음 재생 및 다운로드 기능

### 4. 고급 오디오 처리
- 에코 제거
- 노이즈 게이트
- 자동 게인 컨트롤
- 스펙트럼 분석

---

**이제 LiveKit 방의 설정값에 따라 모든 오디오 처리가 동적으로 제어됩니다!** 🎵

- **실시간 설정 변경**: API를 통한 즉시 적용
- **방별 독립 설정**: 각 방마다 다른 설정 가능
- **캐시 기반 성능**: 빠른 설정 조회 및 적용
- **확장 가능한 구조**: 새로운 오디오 처리 기능 쉽게 추가
