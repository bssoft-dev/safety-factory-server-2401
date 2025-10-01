# LiveKit 소음제거 기능 가이드

## 🎯 개요

LiveKit을 통한 음성통신에서 기존 프로젝트의 소음제거 모델을 채널별로 선택적으로 적용할 수 있는 기능입니다.

## 🔧 기능 특징

### 지원하는 소음제거 모델
1. **Light Enhancement** (`light`): 가벼운 스펙트럼 기반 노이즈 게이팅
2. **Full Enhancement** (`full`): AI 기반 고성능 소음제거 (Demucs 모델)

### 채널별 선택적 적용
- 각 참가자(채널)별로 독립적인 소음제거 설정
- 실시간으로 소음제거 활성화/비활성화 가능
- 소음제거 모델 타입 변경 가능

## 📋 API 엔드포인트

### 1. 소음제거 설정
```http
POST /api/livekit/enhancement/set
Content-Type: application/json

{
    "participant_identity": "user-001",
    "enabled": true,
    "enhancement_type": "light"  // "light" 또는 "full"
}
```

**응답:**
```json
{
    "message": "Enhancement settings updated for user-001",
    "enabled": true,
    "type": "light"
}
```

### 2. 소음제거 설정 조회
```http
GET /api/livekit/enhancement/{participant_identity}
```

**응답:**
```json
{
    "participant_identity": "user-001",
    "enabled": true,
    "type": "light"
}
```

### 3. 전체 소음제거 상태 조회
```http
GET /api/livekit/enhancement/status
```

**응답:**
```json
{
    "total_participants": 3,
    "enhancement_status": {
        "user-001": {
            "participant_identity": "user-001",
            "enabled": true,
            "type": "light"
        },
        "user-002": {
            "participant_identity": "user-002",
            "enabled": true,
            "type": "full"
        },
        "user-003": {
            "participant_identity": "user-003",
            "enabled": false,
            "type": "light"
        }
    }
}
```

## 🚀 사용 예시

### Python 클라이언트 예시

```python
import aiohttp
import asyncio

async def set_enhancement(participant_id, enabled=True, enhancement_type="light"):
    """소음제거 설정"""
    async with aiohttp.ClientSession() as session:
        data = {
            "participant_identity": participant_id,
            "enabled": enabled,
            "enhancement_type": enhancement_type
        }
        
        async with session.post(
            "https://safety-server.bs-soft.co.kr/api/livekit/enhancement/set",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"소음제거 설정 완료: {result}")
                return result
            else:
                print(f"설정 실패: {response.status}")
                return None

async def get_enhancement_status(participant_id):
    """소음제거 상태 조회"""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://safety-server.bs-soft.co.kr/api/livekit/enhancement/{participant_id}"
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"현재 설정: {result}")
                return result
            else:
                print(f"조회 실패: {response.status}")
                return None

# 사용 예시
async def main():
    # Light 소음제거 활성화
    await set_enhancement("user-001", True, "light")
    
    # Full 소음제거 활성화
    await set_enhancement("user-002", True, "full")
    
    # 소음제거 비활성화
    await set_enhancement("user-003", False)
    
    # 상태 조회
    await get_enhancement_status("user-001")

asyncio.run(main())
```

### cURL 예시

```bash
# Light 소음제거 활성화
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/enhancement/set \
  -H "Content-Type: application/json" \
  -d '{
    "participant_identity": "user-001",
    "enabled": true,
    "enhancement_type": "light"
  }'

# Full 소음제거 활성화
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/enhancement/set \
  -H "Content-Type: application/json" \
  -d '{
    "participant_identity": "user-002",
    "enabled": true,
    "enhancement_type": "full"
  }'

# 소음제거 비활성화
curl -X POST https://safety-server.bs-soft.co.kr/api/livekit/enhancement/set \
  -H "Content-Type: application/json" \
  -d '{
    "participant_identity": "user-003",
    "enabled": false,
    "enhancement_type": "light"
  }'

# 상태 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/enhancement/user-001

# 전체 상태 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/enhancement/status
```

## 🎛️ 소음제거 모델 비교

| 모델 | CPU 사용량 | 품질 | 지연시간 | 용도 |
|------|------------|------|----------|------|
| **Light** | 낮음 | 보통 | 매우 낮음 | 실시간 통신, 저사양 디바이스 |
| **Full** | 높음 | 높음 | 낮음 | 고품질 음성, 서버 환경 |

### Light Enhancement 특징
- 스펙트럼 기반 노이즈 게이팅
- CPU 사용량 최소화
- 실시간 처리에 최적화
- 기본 소음 제거 효과

### Full Enhancement 특징
- AI 기반 고성능 소음제거
- Demucs 모델 사용
- 높은 품질의 음성 향상
- 서버 환경에서 권장

## 🔄 실시간 설정 변경

소음제거 설정은 실시간으로 변경 가능합니다:

```python
# 통화 중에 소음제거 모델 변경
await set_enhancement("user-001", True, "full")  # Light → Full

# 통화 중에 소음제거 비활성화
await set_enhancement("user-001", False)  # 활성화 → 비활성화

# 통화 중에 소음제거 재활성화
await set_enhancement("user-001", True, "light")  # 비활성화 → Light
```

## ⚠️ 주의사항

### 1. 성능 고려사항
- **Full Enhancement**: GPU 사용 시 최적 성능
- **Light Enhancement**: CPU만으로도 충분한 성능
- 다중 참가자 시 서버 리소스 모니터링 필요

### 2. 네트워크 고려사항
- 소음제거 처리로 인한 약간의 지연시간 증가
- 대역폭 사용량은 동일 (오디오 데이터 크기 변화 없음)

### 3. 호환성
- 기존 LiveKit 클라이언트와 완전 호환
- 추가 설정 없이 자동으로 적용
- 클라이언트 측 코드 변경 불필요

## 🧪 테스트

테스트 스크립트를 실행하여 기능을 확인할 수 있습니다:

```bash
python test_livekit_enhancement.py
```

테스트 항목:
- ✅ 기본 API 동작 확인
- ✅ Light/Full 모델 설정 테스트
- ✅ 실시간 설정 변경 테스트
- ✅ 다중 참가자 설정 테스트
- ✅ 잘못된 설정 거부 테스트

## 📊 모니터링

### 로그 확인
```bash
# 소음제거 관련 로그 확인
tail -f log.txt | grep "소음제거"

# 참가자별 설정 로그
tail -f log.txt | grep "참가자.*소음제거"
```

### API 상태 확인
```bash
# 전체 소음제거 상태 조회
curl https://safety-server.bs-soft.co.kr/api/livekit/enhancement/status
```

## 🔧 문제 해결

### 일반적인 문제

1. **소음제거가 적용되지 않음**
   - 참가자가 방에 참가했는지 확인
   - `participant_identity`가 정확한지 확인
   - API 응답에서 오류 메시지 확인

2. **성능 문제**
   - Full Enhancement 사용 시 GPU 가용성 확인
   - 동시 참가자 수와 서버 리소스 확인
   - Light Enhancement로 변경 고려

3. **설정이 저장되지 않음**
   - 서버 재시작 시 메모리 기반 설정 초기화됨
   - 참가자 재참가 시 설정 재적용 필요

### 디버깅

```python
# 현재 모든 참가자의 소음제거 상태 확인
async def debug_enhancement_status():
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://safety-server.bs-soft.co.kr/api/livekit/enhancement/status"
        ) as response:
            if response.status == 200:
                data = await response.json()
                print("현재 소음제거 상태:")
                for participant_id, settings in data["enhancement_status"].items():
                    print(f"  {participant_id}: {settings}")
            else:
                print(f"상태 조회 실패: {response.status}")
```

## 🚀 향후 계획

- [ ] 데이터베이스 기반 설정 영속성
- [ ] 추가 소음제거 모델 지원
- [ ] 실시간 품질 모니터링
- [ ] 자동 소음제거 모델 선택
- [ ] 클라이언트 측 소음제거 옵션

---

**문의사항이나 문제가 있으시면 개발팀에 연락해주세요.**
