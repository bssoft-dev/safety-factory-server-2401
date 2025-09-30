#!/usr/bin/env python3
"""
실제 운영 서버 연결 테스트 스크립트
"""

import asyncio
import aiohttp
import json
import ssl

# 운영 서버 설정
LIVEKIT_URL = "wss://livekit.bs-soft.co.kr"
API_BASE_URL = "https://safety-server.bs-soft.co.kr"

def print_test(msg):
    print(f"[TEST] {msg}")

async def test_production_server():
    """실제 운영 서버 테스트"""
    try:
        # SSL 컨텍스트 설정 (자체 서명 인증서 허용)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # 1. 헬스 체크
            print_test("1. 헬스 체크...")
            try:
                async with session.get(f"{API_BASE_URL}/api/livekit/health") as response:
                    if response.status == 200:
                        result = await response.json()
                        print_test(f"헬스 체크 성공: {result}")
                    else:
                        print_test(f"헬스 체크 실패: {response.status}")
                        return
            except Exception as e:
                print_test(f"헬스 체크 오류: {e}")
                return
            
            # 2. 방 생성
            print_test("2. 방 생성...")
            try:
                async with session.post(f"{API_BASE_URL}/api/livekit/rooms", 
                                      json={"room_name": "test-room"}) as response:
                    if response.status == 200:
                        result = await response.json()
                        print_test(f"방 생성 성공: {result}")
                    else:
                        print_test(f"방 생성 실패: {response.status}")
                        return
            except Exception as e:
                print_test(f"방 생성 오류: {e}")
                return
            
            # 3. 방 참가 (토큰 생성)
            print_test("3. 방 참가...")
            try:
                async with session.post(f"{API_BASE_URL}/api/livekit/join", 
                                      json={
                                          "room_name": "test-room",
                                          "participant_name": "Test User",
                                          "participant_identity": "test-user-001"
                                      }) as response:
                    if response.status == 200:
                        result = await response.json()
                        print_test(f"방 참가 성공: 토큰 길이 = {len(result.get('token', ''))}")
                        print_test(f"LiveKit URL: {result.get('url')}")
                        print_test(f"API Base URL: {API_BASE_URL}")
                    else:
                        print_test(f"방 참가 실패: {response.status}")
                        return
            except Exception as e:
                print_test(f"방 참가 오류: {e}")
                return
            
            # 4. 방 목록 조회
            print_test("4. 방 목록 조회...")
            try:
                async with session.get(f"{API_BASE_URL}/api/livekit/rooms") as response:
                    if response.status == 200:
                        result = await response.json()
                        print_test(f"방 목록: {result}")
                    else:
                        print_test(f"방 목록 조회 실패: {response.status}")
            except Exception as e:
                print_test(f"방 목록 조회 오류: {e}")
            
            print_test("모든 API 테스트 완료!")
            print_test("")
            print_test("라즈베리파이 설정 파일 (config.txt) 내용:")
            print_test("=" * 50)
            print_test("communication_mode = \"livekit\"")
            print_test("room_name = \"safety-room\"")
            print_test("livekit_url = \"wss://livekit.bs-soft.co.kr:7880\"")
            print_test("api_base_url = \"https://safety-server.bs-soft.co.kr\"")
            print_test("participant_name = \"RaspberryPi-001\"")
            print_test("participant_identity = \"raspberry-001\"")
            print_test("=" * 50)
            
    except Exception as e:
        print_test(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print_test("실제 운영 서버 연결 테스트 시작")
    print_test(f"LiveKit 서버: {LIVEKIT_URL}")
    print_test(f"API 서버: {API_BASE_URL}")
    print_test("")
    
    asyncio.run(test_production_server())
