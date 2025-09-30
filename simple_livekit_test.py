#!/usr/bin/env python3
"""
간단한 LiveKit 테스트
"""

import asyncio
import aiohttp
import json
# from utils.sys import aprint
def aprint(msg):
    print(f"[TEST] {msg}")

async def test_livekit_api():
    """LiveKit API 테스트"""
    try:
        async with aiohttp.ClientSession() as session:
            # 1. 헬스 체크
            aprint("1. 헬스 체크...")
            async with session.get("http://localhost:24015/api/livekit/health") as response:
                if response.status == 200:
                    result = await response.json()
                    aprint(f"헬스 체크 성공: {result}")
                else:
                    aprint(f"헬스 체크 실패: {response.status}")
                    return
            
            # 2. 방 생성
            aprint("2. 방 생성...")
            async with session.post("http://localhost:24015/api/livekit/rooms", 
                                  json={"room_name": "test-room"}) as response:
                if response.status == 200:
                    result = await response.json()
                    aprint(f"방 생성 성공: {result}")
                else:
                    aprint(f"방 생성 실패: {response.status}")
                    return
            
            # 3. 방 참가 (토큰 생성)
            aprint("3. 방 참가...")
            async with session.post("http://localhost:24015/api/livekit/join", 
                                  json={
                                      "room_name": "test-room",
                                      "participant_name": "Test User",
                                      "participant_identity": "test-user-001"
                                  }) as response:
                if response.status == 200:
                    result = await response.json()
                    aprint(f"방 참가 성공: 토큰 길이 = {len(result.get('token', ''))}")
                    aprint(f"LiveKit URL: {result.get('url')}")
                else:
                    aprint(f"방 참가 실패: {response.status}")
                    return
            
            # 4. 방 목록 조회
            aprint("4. 방 목록 조회...")
            async with session.get("http://localhost:24015/api/livekit/rooms") as response:
                if response.status == 200:
                    result = await response.json()
                    aprint(f"방 목록: {result}")
                else:
                    aprint(f"방 목록 조회 실패: {response.status}")
            
            aprint("모든 API 테스트 완료!")
            
    except Exception as e:
        aprint(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_livekit_api())
