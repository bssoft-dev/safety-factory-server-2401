# 웹소켓 스트리밍 전송 문제 진단 가이드

## 🎯 목적
웹소켓을 통한 실시간 음성통신에서 음원이 끊어지는 문제의 원인을 파악하기 위한 테스트 도구입니다.

## ⚠️ 중요 안내
**모든 테스트는 '테스트' 방에서만 작동합니다.**
- 다른 방의 정상적인 음성통신에는 영향을 주지 않습니다.
- 테스트 전에 반드시 '테스트' 방을 생성하고 해당 방에서 테스트를 진행하세요.

## 🧪 테스트 시나리오

### 1. 무음 테스트 (silence)
**목적**: 기본 송신 로직과 네트워크 문제 확인
```bash
curl -X POST http://localhost:24015/v1/test/start/silence
```
- **예상 결과**: 무음이 끊어지지 않고 지속적으로 전송
- **문제 시**: 기본 송신 로직이나 네트워크 문제

### 2. 사인파 테스트 (sine)
**목적**: 일정한 주파수 신호 전송으로 타이밍 문제 확인
```bash
curl -X POST http://localhost:24015/v1/test/start/sine
```
- **예상 결과**: 440Hz 톤이 끊어지지 않고 재생
- **문제 시**: 프레임 타이밍이나 버퍼링 문제

### 3. 원본 데이터 테스트 (raw)
**목적**: 믹싱 없이 원본 데이터만 전송
```bash
curl -X POST http://localhost:24015/v1/test/start/raw
```
- **예상 결과**: 첫 번째 클라이언트의 원본 음성이 그대로 전송
- **문제 시**: 믹싱 로직 문제

### 4. 음성 향상 제외 테스트 (no_enhance)
**목적**: AI 음성 향상 처리 없이 믹싱만 수행
```bash
curl -X POST http://localhost:24015/v1/test/start/no_enhance
```
- **예상 결과**: 믹싱된 음성이 끊어지지 않고 전송
- **문제 시**: 음성 향상(AI 모델) 처리 문제

### 5. 믹싱 제외 테스트 (no_mixing)
**목적**: 모든 처리 없이 단순 전송
```bash
curl -X POST http://localhost:24015/v1/test/start/no_mixing
```
- **예상 결과**: 첫 번째 클라이언트 음성만 단순 전송
- **문제 시**: 기본 송신 로직 문제

### 6. 프레임 드랍 비활성화 테스트 (no_drop)
**목적**: 프레임 드랍 로직 비활성화
```bash
curl -X POST http://localhost:24015/v1/test/start/no_drop
```
- **예상 결과**: 오래된 프레임도 드랍하지 않고 전송
- **문제 시**: 프레임 드랍 로직 문제

### 7. 연속 사인파 테스트 (continuous_sine)
**목적**: 시간 오프셋으로 연속적인 사인파 전송
```bash
curl -X POST http://localhost:24015/v1/test/start/continuous_sine
```
- **예상 결과**: 지속적인 440Hz 톤
- **문제 시**: 시간 동기화 문제

### 8. 긴 사인파 테스트 (long_sine)
**목적**: 1초 분량의 긴 사인파 전송
```bash
curl -X POST http://localhost:24015/v1/test/start/long_sine
```
- **예상 결과**: 1초 동안 끊어지지 않는 톤
- **문제 시**: 긴 프레임 처리 문제

## 📊 테스트 통계 확인

### 실시간 통계 조회
```bash
curl http://localhost:24015/v1/test/stats
```

**응답 예시**:
```json
{
  "frame_count": 150,
  "send_count": 148,
  "drop_count": 2,
  "drop_rate": 1.33
}
```

### 통계 초기화
```bash
curl -X POST http://localhost:24015/v1/test/reset
```

## 🛑 테스트 중지
```bash
curl -X POST http://localhost:24015/v1/test/stop
```

## 🔍 문제 진단 가이드

### 단계별 진단 순서

1. **무음 테스트** → 기본 송신 로직 확인
2. **사인파 테스트** → 타이밍 및 버퍼링 확인
3. **원본 데이터 테스트** → 믹싱 로직 확인
4. **음성 향상 제외 테스트** → AI 처리 확인
5. **믹싱 제외 테스트** → 최종 송신 로직 확인

### 예상 문제 원인

| 테스트 결과 | 문제 원인 | 해결 방향 |
|------------|----------|----------|
| 무음/사인파에서 끊어짐 | 세마포어, 네트워크 | 송신 로직 최적화 |
| 원본 데이터에서 끊어짐 | 믹싱 로직 | 믹싱 알고리즘 개선 |
| 음성 향상 제외에서 끊어짐 | AI 모델 처리 | 음성 향상 최적화 |
| 모든 테스트에서 끊어짐 | 기본 송신 로직 | 전체 송신 파이프라인 재검토 |

## 🚀 테스트 실행 예시

### 1. 기본 테스트 시나리오
```bash
# 1. 테스트 방 생성
curl -X POST http://localhost:24015/v1/create_room -H "Content-Type: application/json" -d '{"room_name":"테스트"}'

# 2. 무음 테스트 시작
curl -X POST http://localhost:24015/v1/test/start/silence

# 3. 웹 클라이언트에서 '테스트' 방에 접속
# http://localhost:24015/webclient

# 4. 통계 확인
curl http://localhost:24015/v1/test/stats

# 5. 테스트 중지
curl -X POST http://localhost:24015/v1/test/stop
```

### 2. 연속 테스트 시나리오
```bash
# 각 테스트를 순차적으로 실행
for test_type in silence sine raw no_enhance no_mixing; do
    echo "=== $test_type 테스트 시작 ==="
    curl -X POST http://localhost:24015/v1/test/start/$test_type
    sleep 5
    curl http://localhost:24015/v1/test/stats
    curl -X POST http://localhost:24015/v1/test/stop
    sleep 2
done
```

## 📝 로그 모니터링

### 실시간 로그 확인
```bash
tail -f /home/bssoft/safety-factory-server-2401/log.txt | grep "\[TEST\]"
```

### 로그 메시지 예시
```
[TEST] 무음 전송: frame=1
[TEST] 사인파 전송: frame=2
[TEST] 원본 전송: frame=3, size=1024
[TEST] 믹싱만 전송: frame=4
[TEST] 단순 전송: frame=5
```

## ⚠️ 주의사항

1. **테스트 중에는 실제 음성통신이 중단됩니다**
2. **테스트 완료 후 반드시 `stop` 명령으로 정상 모드로 복귀**
3. **통계는 누적되므로 테스트 간 초기화 권장**
4. **프로덕션 환경에서는 테스트 모드 사용 금지**

## 🔧 문제 해결 후 조치

테스트 결과에 따라 다음 조치를 수행하세요:

1. **세마포어 문제**: `_safe_send_audio_bounded` 로직 개선
2. **믹싱 문제**: `mix_audio` 함수 최적화
3. **AI 처리 문제**: `voice_enhance` 함수 성능 개선
4. **전체 송신 문제**: 송신 파이프라인 전면 재검토
