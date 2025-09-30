#!/usr/bin/env python3
"""
WebSocket 연결 문제 진단 스크립트
"""
import asyncio
import websockets
import json
import sys

async def test_websocket_connection():
    """WebSocket 연결 테스트"""
    
    # 테스트할 URL들
    test_urls = [
        # ❌ 잘못된 URL (현재 클라이언트가 사용하는 것)
        "ws://localhost:24015/ws/room/보온팀",
        
        # ✅ 올바른 URL들
        "ws://localhost:24015/ws/room/보온팀/16000/float32/default",
        "ws://localhost:24015/ws/room/보온팀/16000/int16/default",
        "ws://localhost:24015/ws/room/보온팀/48000/float32/default",
        
        # URL 인코딩 테스트
        "ws://localhost:24015/ws/room/%EB%B3%B4%EC%98%A8%ED%8C%80/16000/float32/default",
    ]
    
    print("🔍 WebSocket 연결 테스트 시작")
    print("=" * 50)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n{i}. 테스트 URL: {url}")
        print("-" * 30)
        
        try:
            # 연결 시도
            async with websockets.connect(url, timeout=5) as websocket:
                print("✅ 연결 성공!")
                
                # 간단한 메시지 전송 테스트
                test_message = b"test_audio_data" * 100  # 1400 bytes
                await websocket.send(test_message)
                print("✅ 메시지 전송 성공")
                
                # 응답 대기 (짧게)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    print(f"✅ 응답 수신: {len(response)} bytes")
                except asyncio.TimeoutError:
                    print("⚠️ 응답 타임아웃 (정상 - 서버가 오디오만 처리)")
                
        except websockets.exceptions.InvalidURI as e:
            print(f"❌ 잘못된 URI: {e}")
        except websockets.exceptions.ConnectionClosed as e:
            print(f"❌ 연결 종료: {e}")
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"❌ 잘못된 상태 코드: {e}")
        except asyncio.TimeoutError:
            print("❌ 연결 타임아웃")
        except Exception as e:
            print(f"❌ 기타 오류: {e}")
    
    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약")
    print("=" * 50)
    print("1. 잘못된 URL (파라미터 누락) → 연결 실패 예상")
    print("2. 올바른 URL (모든 파라미터 포함) → 연결 성공 예상")
    print("3. URL 인코딩 → 연결 성공 예상")

async def test_server_status():
    """서버 상태 확인"""
    print("\n🔍 서버 상태 확인")
    print("=" * 30)
    
    try:
        import aiohttp
        
        # 기본 서버 상태 확인
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:24015/") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ 서버 실행 중: {data}")
                else:
                    print(f"❌ 서버 응답 오류: {response.status}")
            
            # 방 목록 확인
            async with session.get("http://localhost:24015/v1/rooms") as response:
                if response.status == 200:
                    rooms = await response.json()
                    print(f"✅ 방 목록: {len(rooms)}개 방")
                    for room in rooms:
                        print(f"   - {room['room_name']} (ID: {room['id']}, 인원: {room['num_person']})")
                else:
                    print(f"❌ 방 목록 조회 실패: {response.status}")
                    
    except Exception as e:
        print(f"❌ 서버 상태 확인 실패: {e}")

def print_fix_guide():
    """수정 가이드 출력"""
    print("\n" + "=" * 50)
    print("🛠️ 클라이언트 수정 가이드")
    print("=" * 50)
    
    print("\n❌ 현재 잘못된 URL:")
    print('   "wss://safety-server.bs-soft.co.kr/ws/room/보온팀"')
    
    print("\n✅ 올바른 URL 형식:")
    print('   "wss://safety-server.bs-soft.co.kr/ws/room/보온팀/16000/float32/default"')
    
    print("\n📝 필수 파라미터:")
    print("   1. room_name: 방 이름 (예: '보온팀')")
    print("   2. sr: 샘플레이트 (16000 또는 48000)")
    print("   3. dtype: 데이터 타입 ('int16' 또는 'float32')")
    print("   4. device_id: 디바이스 ID (예: 'default', 'webclient-001')")
    
    print("\n🔧 클라이언트 코드 수정 예시:")
    print("""
    // 기존 (잘못된 코드)
    const wsUrl = `wss://safety-server.bs-soft.co.kr/ws/room/${roomName}`;
    
    // 수정 후 (올바른 코드)
    const wsUrl = `wss://safety-server.bs-soft.co.kr/ws/room/${encodeURIComponent(roomName)}/${sampleRate}/${dataType}/${deviceId}`;
    
    // 예시
    const wsUrl = `wss://safety-server.bs-soft.co.kr/ws/room/보온팀/16000/float32/default`;
    """)
    
    print("\n⚠️ 주의사항:")
    print("   1. 한글 방 이름은 URL 인코딩 필요")
    print("   2. 모든 파라미터가 필수")
    print("   3. 샘플레이트는 16000 또는 48000만 지원")
    print("   4. 데이터 타입은 'int16' 또는 'float32'만 지원")

async def main():
    """메인 함수"""
    print("🚀 Safety Server WebSocket 연결 진단")
    print("=" * 60)
    
    # 1. 서버 상태 확인
    await test_server_status()
    
    # 2. WebSocket 연결 테스트
    await test_websocket_connection()
    
    # 3. 수정 가이드 출력
    print_fix_guide()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ 테스트 중단됨")
    except Exception as e:
        print(f"\n❌ 테스트 실행 오류: {e}")
        sys.exit(1) 