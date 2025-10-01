#!/usr/bin/env python3
"""
LiveKit 방 상태 조회 기능 테스트 스크립트
"""

import asyncio
import aiohttp
import json
import ssl

# 서버 설정
API_BASE_URL = "https://safety-server.bs-soft.co.kr"

def print_test(msg):
    print(f"[TEST] {msg}")

async def test_room_status_apis():
    """방 상태 조회 API 테스트"""
    print_test("=== LiveKit 방 상태 조회 API 테스트 시작 ===")
    
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
            room_data = {"room_name": "status-test-room"}
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms", json=room_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 방 생성 성공: {data}")
                else:
                    print_test(f"❌ 방 생성 실패: {response.status}")
                    return
            
            # 3. 빈 방 상태 조회 (토큰 생성 전)
            print_test("3. 빈 방 상태 조회 (토큰 생성 전)")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/status-test-room/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 빈 방 상태 조회 성공: {json.dumps(data, indent=2, ensure_ascii=False)}")
                else:
                    print_test(f"❌ 빈 방 상태 조회 실패: {response.status}")
            
            # 4. 참가자 토큰 생성 (실제 연결은 아님)
            print_test("4. 참가자 토큰 생성 (실제 연결은 아님)")
            participants = [
                {"name": "User1", "identity": "user-001"},
                {"name": "User2", "identity": "user-002"},
                {"name": "User3", "identity": "user-003"}
            ]
            
            for participant in participants:
                join_data = {
                    "room_name": "status-test-room",
                    "participant_name": participant["name"],
                    "participant_identity": participant["identity"]
                }
                async with session.post(f"{API_BASE_URL}/api/livekit/join", json=join_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        print_test(f"✅ {participant['name']} 토큰 생성 성공 (실제 연결 전)")
                    else:
                        print_test(f"❌ {participant['name']} 토큰 생성 실패")
            
            # 5. 토큰 생성 후 방 상태 조회 (여전히 빈 방)
            print_test("5. 토큰 생성 후 방 상태 조회 (여전히 빈 방)")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/status-test-room/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 방 상태 조회 성공 (토큰만 생성됨):")
                    print_test(f"  - 방 이름: {data['room_name']}")
                    print_test(f"  - 총 참가자 수: {data['total_participants']} (실제 연결 없음)")
                    print_test(f"  - 활성 상태: {data['is_active']}")
                    print_test(f"  - 참가자 목록: {len(data['participants'])}명")
                    if data['total_participants'] == 0:
                        print_test("  ✅ 올바름: 토큰 생성만으로는 참가자로 등록되지 않음")
                    else:
                        print_test("  ❌ 예상과 다름: 토큰 생성만으로 참가자가 등록됨")
                else:
                    print_test(f"❌ 방 상태 조회 실패: {response.status}")
            
            # 6. 전체 방 상태 조회
            print_test("6. 전체 방 상태 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 전체 방 상태 조회 성공:")
                    print_test(f"  - 총 방 수: {data['total_rooms']}")
                    print_test(f"  - 총 참가자 수: {data['total_participants']}")
                    print_test(f"  - 방 목록:")
                    for room_name, room_info in data['rooms'].items():
                        print_test(f"    * {room_name}: {room_info['total_participants']}명 참가")
                else:
                    print_test(f"❌ 전체 방 상태 조회 실패: {response.status}")
            
            # 7. 존재하지 않는 방 조회
            print_test("7. 존재하지 않는 방 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/nonexistent-room/status") as response:
                if response.status == 404:
                    data = await response.json()
                    print_test(f"✅ 존재하지 않는 방 조회 거부 성공: {data}")
                else:
                    print_test(f"❌ 존재하지 않는 방 조회 거부 실패: {response.status}")
            
            # 8. 참가자별 소음제거 설정 변경 후 상태 확인
            print_test("8. 참가자별 소음제거 설정 변경 후 상태 확인")
            enhancement_data = {
                "participant_identity": "user-001",
                "enabled": True,
                "enhancement_type": "full"
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=enhancement_data) as response:
                if response.status == 200:
                    print_test("✅ user-001 소음제거 설정 변경 성공")
            
            # 상태 재조회
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/status-test-room/status") as response:
                if response.status == 200:
                    data = await response.json()
                    user001 = next((p for p in data['participants'] if p['identity'] == 'user-001'), None)
                    if user001:
                        print_test(f"✅ user-001 소음제거 설정 확인: {user001['enhancement']['type']} ({'활성화' if user001['enhancement']['enabled'] else '비활성화'})")
                else:
                    print_test(f"❌ 상태 재조회 실패: {response.status}")
            
            print_test("=== 모든 테스트 완료 ===")
            
        except Exception as e:
            print_test(f"❌ 테스트 중 오류 발생: {e}")

async def test_multiple_rooms():
    """다중 방 상태 테스트"""
    print_test("=== 다중 방 상태 테스트 ===")
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            # 여러 방 생성
            rooms = ["room-1", "room-2", "room-3"]
            
            for room_name in rooms:
                print_test(f"방 생성: {room_name}")
                room_data = {"room_name": room_name}
                async with session.post(f"{API_BASE_URL}/api/livekit/rooms", json=room_data) as response:
                    if response.status == 200:
                        print_test(f"✅ {room_name} 생성 성공")
                    else:
                        print_test(f"❌ {room_name} 생성 실패")
                        continue
                
                # 각 방에 참가자 추가
                for i in range(1, 4):  # 각 방에 3명씩
                    participant_name = f"User-{i}"
                    participant_identity = f"{room_name}-user-{i}"
                    
                    join_data = {
                        "room_name": room_name,
                        "participant_name": participant_name,
                        "participant_identity": participant_identity
                    }
                    async with session.post(f"{API_BASE_URL}/api/livekit/join", json=join_data) as response:
                        if response.status == 200:
                            print_test(f"✅ {participant_name}가 {room_name}에 참가")
                        else:
                            print_test(f"❌ {participant_name} 참가 실패")
            
            # 전체 방 상태 조회
            print_test("전체 방 상태 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/rooms/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 전체 방 상태:")
                    print_test(f"  - 총 방 수: {data['total_rooms']}")
                    print_test(f"  - 총 참가자 수: {data['total_participants']}")
                    print_test(f"  - 방별 상세 정보:")
                    for room_name, room_info in data['rooms'].items():
                        print_test(f"    * {room_name}:")
                        print_test(f"      - 참가자 수: {room_info['total_participants']}")
                        print_test(f"      - 활성 상태: {room_info['is_active']}")
                        print_test(f"      - 오디오 버퍼: {room_info['audio_buffer_count']}개")
                        for participant in room_info['participants']:
                            print_test(f"        - {participant['name']} ({participant['identity']})")
                else:
                    print_test(f"❌ 전체 방 상태 조회 실패: {response.status}")
            
        except Exception as e:
            print_test(f"❌ 다중 방 테스트 중 오류: {e}")

async def main():
    """메인 테스트 함수"""
    print_test("LiveKit 방 상태 조회 기능 테스트 시작")
    
    # 기본 API 테스트
    await test_room_status_apis()
    
    print_test("\n" + "="*50 + "\n")
    
    # 다중 방 테스트
    await test_multiple_rooms()
    
    print_test("\n모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())
