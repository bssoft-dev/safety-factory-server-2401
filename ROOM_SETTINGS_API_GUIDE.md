# 🛠️ Safety Server 방별 설정 API 가이드

## 📋 개요

Safety Server에서 이제 방(Room)별로 개별적인 오디오 처리 설정을 관리할 수 있습니다.
각 방마다 소음제거, STT, 음원분류 등의 기능을 독립적으로 설정할 수 있습니다.

## 🎯 방별 설정 항목

| 설정 | 타입 | 기본값 | 설명 |
|------|------|---------|------|
| `use_voice_enhance` | boolean | true | 소음제거 (노이즈 리덕션) |
| `hear_me` | boolean | false | 자신의 목소리 듣기 |
| `record_audio` | boolean | true | 오디오 녹음 |
| `classify_event` | boolean | true | 음원분류 (비명, 경보음, 충격음) |
| `do_stt` | boolean | true | STT (음성인식) |
| `enhance_volume` | integer | 0 | 볼륨 증폭 (0-10) |

## 🔗 API 엔드포인트

### 1. 방 설정 조회
```http
GET /v1/room/{room_name}/settings
```

**응답 예시:**
```json
{
  "room_name": "테스트방",
  "settings": {
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": true,
    "enhance_volume": 0
  }
}
```

### 2. 방 설정 전체 업데이트
```http
PUT /v1/room/{room_name}/settings
Content-Type: application/json

{
  "room_name": "테스트방",
  "use_voice_enhance": false,
  "hear_me": true,
  "record_audio": true,
  "classify_event": false,
  "do_stt": true,
  "enhance_volume": 2
}
```

**응답:**
```json
{
  "message": "방 '테스트방' 설정이 업데이트되었습니다.",
  "updated_settings": {
    "use_voice_enhance": false,
    "hear_me": true,
    "record_audio": true,
    "classify_event": false,
    "do_stt": true,
    "enhance_volume": 2
  }
}
```

### 3. 개별 설정 조회
```http
GET /v1/room/{room_name}/settings/{setting_name}
```

**응답 예시:**
```json
{
  "room_name": "테스트방",
  "setting_name": "use_voice_enhance",
  "value": true
}
```

### 4. 개별 설정 업데이트
```http
PUT /v1/room/{room_name}/settings/{setting_name}
Content-Type: application/json

true
```

**응답:**
```json
{
  "message": "방 '테스트방'의 'use_voice_enhance' 설정이 업데이트되었습니다.",
  "room_name": "테스트방",
  "setting_name": "use_voice_enhance",
  "new_value": true
}
```

### 5. 방 설정 리셋 (기본값으로)
```http
POST /v1/room/{room_name}/settings/reset
```

**응답:**
```json
{
  "message": "방 '테스트방' 설정이 기본값으로 리셋되었습니다.",
  "default_settings": {
    "use_voice_enhance": true,
    "hear_me": false,
    "record_audio": true,
    "classify_event": true,
    "do_stt": true,
    "enhance_volume": 0
  }
}
```

## 🌐 웹 클라이언트 사용법

### 1. 방 설정 로드
1. 방 이름 입력: `테스트방`
2. **⚙️ 방 설정 로드** 버튼 클릭
3. 방별 설정 UI가 표시됨
4. 현재 설정값이 체크박스에 반영됨

### 2. 설정 변경
- **소음제거**: 체크박스 클릭으로 즉시 적용
- **STT**: 음성인식 기능 켜기/끄기
- **음원분류**: 비명, 경보음 감지 켜기/끄기
- **볼륨 증폭**: 슬라이더로 0-10 범위 조절

### 3. 기본값 리셋
- **🔄 기본값으로 리셋** 버튼 클릭
- 확인 대화상자 후 모든 설정이 기본값으로 복원

## 📝 cURL 사용 예시

### 방 설정 조회
```bash
curl -X GET "http://localhost:24015/v1/room/테스트방/settings"
```

### 소음제거 끄기
```bash
curl -X PUT "http://localhost:24015/v1/room/테스트방/settings/use_voice_enhance" \
  -H "Content-Type: application/json" \
  -d "false"
```

### STT 켜기
```bash
curl -X PUT "http://localhost:24015/v1/room/테스트방/settings/do_stt" \
  -H "Content-Type: application/json" \
  -d "true"
```

### 볼륨 증폭 설정
```bash
curl -X PUT "http://localhost:24015/v1/room/테스트방/settings/enhance_volume" \
  -H "Content-Type: application/json" \
  -d "3"
```

### 전체 설정 업데이트
```bash
curl -X PUT "http://localhost:24015/v1/room/테스트방/settings" \
  -H "Content-Type: application/json" \
  -d '{
    "room_name": "테스트방",
    "use_voice_enhance": false,
    "hear_me": true,
    "record_audio": true,
    "classify_event": false,
    "do_stt": true,
    "enhance_volume": 2
  }'
```

## 🔄 기존 전역 설정과의 호환성

기존 전역 설정 API들은 여전히 작동합니다 (호환성 유지):

```http
GET /v1/hear_me/true           # 전역 설정
GET /v1/noise_remove/false     # 전역 설정
GET /v1/do_stt/true           # 전역 설정
GET /v1/classify_event/false   # 전역 설정
```

하지만 **방별 설정이 우선순위**를 가집니다:
- 방별 설정이 있으면: 방별 설정 사용
- 방별 설정이 없으면: 전역 설정 사용

## 💾 데이터베이스 구조

### RoomSettings 테이블
```sql
CREATE TABLE safety_factory_2401.room_settings (
    id SERIAL PRIMARY KEY,
    room_name VARCHAR UNIQUE NOT NULL,
    use_voice_enhance BOOLEAN DEFAULT true,
    hear_me BOOLEAN DEFAULT false,
    record_audio BOOLEAN DEFAULT true,
    classify_event BOOLEAN DEFAULT true,
    do_stt BOOLEAN DEFAULT true,
    enhance_volume INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## 🎯 활용 시나리오

### 시나리오 1: 회의실 설정
```json
{
  "room_name": "회의실A",
  "use_voice_enhance": true,    // 소음제거 켜기
  "hear_me": false,             // 에코 방지
  "record_audio": true,         // 회의 녹음
  "classify_event": false,      // 비명감지 끄기
  "do_stt": true,              // 회의록 자동 생성
  "enhance_volume": 1          // 약간의 볼륨 증폭
}
```

### 시나리오 2: 안전 모니터링 구역
```json
{
  "room_name": "작업장B",
  "use_voice_enhance": true,    // 소음제거 켜기
  "hear_me": false,             // 모니터링 전용
  "record_audio": true,         // 사고 기록용
  "classify_event": true,       // 위험음 감지 켜기
  "do_stt": false,             // STT 불필요
  "enhance_volume": 3          // 높은 볼륨으로 모니터링
}
```

### 시나리오 3: 테스트 환경
```json
{
  "room_name": "테스트방",
  "use_voice_enhance": false,   // 원음 확인
  "hear_me": true,              // 자신 목소리 확인
  "record_audio": false,        // 녹음 불필요
  "classify_event": false,      // 분류 테스트 별도
  "do_stt": false,             // STT 테스트 별도
  "enhance_volume": 0          // 원래 볼륨
}
```

## 🚀 시작하기

1. **웹 클라이언트 접속**: `http://safety-server.bs-soft.co.kr:24015/webclient`
2. **방 이름 입력**: 예) "테스트방"
3. **방 설정 로드**: ⚙️ 버튼 클릭
4. **설정 조정**: 체크박스와 슬라이더로 원하는 설정
5. **연결 테스트**: 🔗 연결 버튼으로 오디오 스트리밍 시작

---
**💡 참고**: 방별 설정은 실시간으로 적용되며, 데이터베이스에 영구 저장됩니다.
