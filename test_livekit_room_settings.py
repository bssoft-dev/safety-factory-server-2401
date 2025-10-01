#!/usr/bin/env python3
"""
LiveKit 방 설정 기능 테스트 스크립트
"""

import asyncio
import aiohttp
import json
import ssl

# 서버 설정
API_BASE_URL = "https://safety-server.bs-soft.co.kr"

def print_test(msg):
    print(f"[TEST] {msg}")

async def test_room_settings_apis():
    """방 설정 API 테스트"""
    print_test("=== LiveKit 방 설정 API 테스트 시작 ===")
    
    # SSL 컨텍스트 설정 (인증서 검증 비활성화)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            # 1. Health check
            print_test("1. Health check 테스트")
            async with session.get(f"{API_BASE_URL}/api/livekit/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ Health check 성공: {data}")
                else:
                    print_test(f"❌ Health check 실패: {response.status}")
                    return
            
            # 2. 테스트 방 생성
            print_test("2. 테스트 방 생성")
            room_data = {"room_name": "settings-test-room"}
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms", json=room_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 방 생성 성공: {data}")
                else:
                    print_test(f"❌ 방 생성 실패: {response.status}")
                    return
            
            # 3. 기본 방 설정 조회
            print_test("3. 기본 방 설정 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/settings-test-room/settings") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 기본 방 설정 조회 성공:")
                    print_test(f"  - 방 이름: {data.get('room_name', 'N/A')}")
                    print_test(f"  - 음성 강화: {data['settings']['use_voice_enhance']}")
                    print_test(f"  - 내 소리 듣기: {data['settings']['hear_me']}")
                    print_test(f"  - 오디오 녹음: {data['settings']['record_audio']}")
                    print_test(f"  - 이벤트 분류: {data['settings']['classify_event']}")
                    print_test(f"  - 음성 인식: {data['settings']['do_stt']}")
                    print_test(f"  - 볼륨 증폭: {data['settings']['enhance_volume']}")
                else:
                    print_test(f"❌ 방 설정 조회 실패: {response.status}")
            
            # 4. 방 설정 업데이트 (음성 강화 비활성화)
            print_test("4. 방 설정 업데이트 (음성 강화 비활성화)")
            settings_data = {
                "room_name": "settings-test-room",
                "use_voice_enhance": False
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 음성 강화 비활성화 성공: {data}")
                else:
                    print_test(f"❌ 음성 강화 비활성화 실패: {response.status}")
            
            # 5. 방 설정 업데이트 (내 소리 듣기 활성화)
            print_test("5. 방 설정 업데이트 (내 소리 듣기 활성화)")
            settings_data = {
                "room_name": "settings-test-room",
                "hear_me": True
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 내 소리 듣기 활성화 성공: {data}")
                else:
                    print_test(f"❌ 내 소리 듣기 활성화 실패: {response.status}")
            
            # 6. 방 설정 업데이트 (볼륨 증폭 50%)
            print_test("6. 방 설정 업데이트 (볼륨 증폭 50%)")
            settings_data = {
                "room_name": "settings-test-room",
                "enhance_volume": 50
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 볼륨 증폭 50% 설정 성공: {data}")
                else:
                    print_test(f"❌ 볼륨 증폭 설정 실패: {response.status}")
            
            # 7. 방 설정 업데이트 (AI 기능 비활성화)
            print_test("7. 방 설정 업데이트 (AI 기능 비활성화)")
            settings_data = {
                "room_name": "settings-test-room",
                "classify_event": False,
                "do_stt": False
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ AI 기능 비활성화 성공: {data}")
                else:
                    print_test(f"❌ AI 기능 비활성화 실패: {response.status}")
            
            # 8. 업데이트된 방 설정 조회
            print_test("8. 업데이트된 방 설정 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/settings-test-room/settings") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 업데이트된 방 설정 조회 성공:")
                    print_test(f"  - 음성 강화: {data['settings'].get('use_voice_enhance', 'N/A')} (False로 변경됨)")
                    print_test(f"  - 내 소리 듣기: {data['settings'].get('hear_me', 'N/A')} (True로 변경됨)")
                    print_test(f"  - 오디오 녹음: {data['settings'].get('record_audio', 'N/A')}")
                    print_test(f"  - 이벤트 분류: {data['settings'].get('classify_event', 'N/A')} (False로 변경됨)")
                    print_test(f"  - 음성 인식: {data['settings'].get('do_stt', 'N/A')} (False로 변경됨)")
                    print_test(f"  - 볼륨 증폭: {data['settings'].get('enhance_volume', 'N/A')} (50으로 변경됨)")
                else:
                    print_test(f"❌ 업데이트된 방 설정 조회 실패: {response.status}")
            
            # 9. 방 상태 조회 (설정 정보 포함)
            print_test("9. 방 상태 조회 (설정 정보 포함)")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/settings-test-room/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 방 상태 조회 성공 (설정 정보 포함):")
                    print_test(f"  - 방 이름: {data['room_name']}")
                    print_test(f"  - 참가자 수: {data['total_participants']}")
                    print_test(f"  - 설정 정보:")
                    for key, value in data['settings'].items():
                        print_test(f"    * {key}: {value}")
                else:
                    print_test(f"❌ 방 상태 조회 실패: {response.status}")
            
            # 10. 잘못된 설정 테스트
            print_test("10. 잘못된 설정 테스트 (볼륨 증폭 150%)")
            settings_data = {
                "room_name": "settings-test-room",
                "enhance_volume": 150  # 100을 초과하는 값
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 볼륨 증폭 150% 설정 성공 (서버에서 처리): {data}")
                else:
                    print_test(f"❌ 볼륨 증폭 150% 설정 실패: {response.status}")
            
            print_test("=== 모든 테스트 완료 ===")
            
        except Exception as e:
            print_test(f"❌ 테스트 중 오류 발생: {e}")

async def test_multiple_rooms_settings():
    """다중 방 설정 테스트"""
    print_test("=== 다중 방 설정 테스트 ===")
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            # 여러 방 생성 및 각각 다른 설정 적용
            rooms_config = [
                {
                    "name": "room-high-quality",
                    "settings": {
                        "use_voice_enhance": True,
                        "hear_me": False,
                        "record_audio": True,
                        "classify_event": True,
                        "do_stt": True,
                        "enhance_volume": 20
                    }
                },
                {
                    "name": "room-low-latency",
                    "settings": {
                        "use_voice_enhance": False,
                        "hear_me": True,
                        "record_audio": False,
                        "classify_event": False,
                        "do_stt": False,
                        "enhance_volume": 0
                    }
                },
                {
                    "name": "room-balanced",
                    "settings": {
                        "use_voice_enhance": True,
                        "hear_me": False,
                        "record_audio": True,
                        "classify_event": True,
                        "do_stt": False,
                        "enhance_volume": 10
                    }
                }
            ]
            
            for room_config in rooms_config:
                print_test(f"방 생성 및 설정: {room_config['name']}")
                
                # 방 생성
                room_data = {"room_name": room_config["name"]}
                async with session.post(f"{API_BASE_URL}/api/livekit/rooms", json=room_data) as response:
                    if response.status == 200:
                        print_test(f"✅ {room_config['name']} 생성 성공")
                    else:
                        print_test(f"❌ {room_config['name']} 생성 실패")
                        continue
                
                # 설정 적용
                settings_data = {"room_name": room_config["name"], **room_config["settings"]}
                async with session.post(f"{API_BASE_URL}/api/livekit/rooms/settings", json=settings_data) as response:
                    if response.status == 200:
                        print_test(f"✅ {room_config['name']} 설정 적용 성공")
                    else:
                        print_test(f"❌ {room_config['name']} 설정 적용 실패")
            
            # 전체 방 설정 조회
            print_test("전체 방 설정 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/settings/all") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 전체 방 설정 조회 성공:")
                    print_test(f"  - 총 방 수: {data['total_rooms']}")
                    print_test(f"  - 방별 설정:")
                    for room_name, room_info in data['rooms_settings'].items():
                        print_test(f"    * {room_name}:")
                        for setting_key, setting_value in room_info['settings'].items():
                            print_test(f"      - {setting_key}: {setting_value}")
                else:
                    print_test(f"❌ 전체 방 설정 조회 실패: {response.status}")
            
        except Exception as e:
            print_test(f"❌ 다중 방 설정 테스트 중 오류: {e}")

async def main():
    """메인 테스트 함수"""
    print_test("LiveKit 방 설정 기능 테스트 시작")
    
    # 기본 API 테스트
    await test_room_settings_apis()
    
    print_test("\n" + "="*50 + "\n")
    
    # 다중 방 설정 테스트
    await test_multiple_rooms_settings()
    
    print_test("\n모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())
