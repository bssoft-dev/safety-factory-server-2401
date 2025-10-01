#!/usr/bin/env python3
"""
LiveKit 소음제거 기능 테스트 스크립트
"""

import asyncio
import aiohttp
import json
import ssl

# 서버 설정
API_BASE_URL = "https://safety-server.bs-soft.co.kr"

def print_test(msg):
    print(f"[TEST] {msg}")

async def test_enhancement_apis():
    """소음제거 API 테스트"""
    print_test("=== LiveKit 소음제거 API 테스트 시작 ===")
    
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
            
            # 2. 방 생성
            print_test("2. 테스트 방 생성")
            room_data = {"room_name": "enhancement-test"}
            async with session.post(f"{API_BASE_URL}/api/livekit/rooms", json=room_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 방 생성 성공: {data}")
                else:
                    print_test(f"❌ 방 생성 실패: {response.status}")
                    return
            
            # 3. 참가자 참가 (토큰 생성)
            print_test("3. 참가자 참가 및 토큰 생성")
            join_data = {
                "room_name": "enhancement-test",
                "participant_name": "TestUser",
                "participant_identity": "test-user-001"
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/join", json=join_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 참가자 참가 성공: 토큰 길이 {len(data.get('token', ''))}")
                else:
                    print_test(f"❌ 참가자 참가 실패: {response.status}")
                    return
            
            # 4. 소음제거 설정 (Light 모드)
            print_test("4. Light 소음제거 설정")
            enhancement_data = {
                "participant_identity": "test-user-001",
                "enabled": True,
                "enhancement_type": "light"
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=enhancement_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ Light 소음제거 설정 성공: {data}")
                else:
                    print_test(f"❌ Light 소음제거 설정 실패: {response.status}")
            
            # 5. 소음제거 설정 조회
            print_test("5. 소음제거 설정 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/enhancement/participant/test-user-001") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 소음제거 설정 조회 성공: {data}")
                else:
                    print_test(f"❌ 소음제거 설정 조회 실패: {response.status}")
            
            # 6. Full 소음제거 설정
            print_test("6. Full 소음제거 설정")
            enhancement_data["enhancement_type"] = "full"
            async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=enhancement_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ Full 소음제거 설정 성공: {data}")
                else:
                    print_test(f"❌ Full 소음제거 설정 실패: {response.status}")
            
            # 7. 소음제거 비활성화
            print_test("7. 소음제거 비활성화")
            enhancement_data["enabled"] = False
            async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=enhancement_data) as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 소음제거 비활성화 성공: {data}")
                else:
                    print_test(f"❌ 소음제거 비활성화 실패: {response.status}")
            
            # 8. 전체 상태 조회
            print_test("8. 전체 소음제거 상태 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/enhancement/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 전체 상태 조회 성공: {data}")
                else:
                    print_test(f"❌ 전체 상태 조회 실패: {response.status}")
            
            # 9. 잘못된 설정 테스트
            print_test("9. 잘못된 설정 테스트")
            invalid_data = {
                "participant_identity": "test-user-001",
                "enabled": True,
                "enhancement_type": "invalid_type"
            }
            async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=invalid_data) as response:
                if response.status == 400:
                    data = await response.json()
                    print_test(f"✅ 잘못된 설정 거부 성공: {data}")
                else:
                    print_test(f"❌ 잘못된 설정 거부 실패: {response.status}")
            
            print_test("=== 모든 테스트 완료 ===")
            
        except Exception as e:
            print_test(f"❌ 테스트 중 오류 발생: {e}")

async def test_multiple_participants():
    """여러 참가자 소음제거 설정 테스트"""
    print_test("=== 다중 참가자 소음제거 테스트 ===")
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            # 여러 참가자 추가
            participants = [
                {"identity": "user-001", "name": "User1", "type": "light"},
                {"identity": "user-002", "name": "User2", "type": "full"},
                {"identity": "user-003", "name": "User3", "type": "light"},
                {"identity": "user-004", "name": "User4", "enabled": False}
            ]
            
            for i, participant in enumerate(participants):
                print_test(f"참가자 {i+1} 설정: {participant['identity']}")
                
                # 참가자 참가
                join_data = {
                    "room_name": "enhancement-test",
                    "participant_name": participant["name"],
                    "participant_identity": participant["identity"]
                }
                async with session.post(f"{API_BASE_URL}/api/livekit/join", json=join_data) as response:
                    if response.status == 200:
                        print_test(f"✅ {participant['identity']} 참가 성공")
                    else:
                        print_test(f"❌ {participant['identity']} 참가 실패")
                        continue
                
                # 소음제거 설정
                if participant.get("enabled", True):
                    enhancement_data = {
                        "participant_identity": participant["identity"],
                        "enabled": True,
                        "enhancement_type": participant["type"]
                    }
                else:
                    enhancement_data = {
                        "participant_identity": participant["identity"],
                        "enabled": False,
                        "enhancement_type": "light"
                    }
                
                async with session.post(f"{API_BASE_URL}/api/livekit/enhancement/set", json=enhancement_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        print_test(f"✅ {participant['identity']} 소음제거 설정 성공: {data}")
                    else:
                        print_test(f"❌ {participant['identity']} 소음제거 설정 실패")
            
            # 전체 상태 조회
            print_test("전체 참가자 상태 조회")
            async with session.get(f"{API_BASE_URL}/api/livekit/enhancement/status") as response:
                if response.status == 200:
                    data = await response.json()
                    print_test(f"✅ 전체 상태: {json.dumps(data, indent=2, ensure_ascii=False)}")
                else:
                    print_test(f"❌ 전체 상태 조회 실패")
            
        except Exception as e:
            print_test(f"❌ 다중 참가자 테스트 중 오류: {e}")

async def main():
    """메인 테스트 함수"""
    print_test("LiveKit 소음제거 기능 테스트 시작")
    
    # 기본 API 테스트
    await test_enhancement_apis()
    
    print_test("\n" + "="*50 + "\n")
    
    # 다중 참가자 테스트
    await test_multiple_participants()
    
    print_test("\n모든 테스트 완료!")

if __name__ == "__main__":
    asyncio.run(main())
