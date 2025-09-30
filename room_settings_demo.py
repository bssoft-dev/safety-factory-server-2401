#!/usr/bin/env python3
"""
Safety Server 방별 설정 API 데모
"""
import asyncio
import aiohttp
import json

async def test_room_settings():
    """방별 설정 API 테스트"""
    
    base_url = "http://localhost:24015"
    room_name = "테스트방"
    
    async with aiohttp.ClientSession() as session:
        print("🚀 Safety Server 방별 설정 API 테스트")
        
        # 1. 방 생성
        try:
            async with session.post(f"{base_url}/v1/create_room", 
                                   json={"room_name": room_name}) as resp:
                result = await resp.json()
                print(f"✅ 방 생성: {result.get('message', 'OK')}")
        except Exception as e:
            print(f"⚠️ 방 생성 오류 (이미 존재할 수 있음): {e}")
        
        # 2. 방 설정 조회
        try:
            async with session.get(f"{base_url}/v1/room/{room_name}/settings") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    settings = data['settings']
                    print(f"📋 현재 방 설정:")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
                else:
                    error = await resp.json()
                    print(f"❌ 설정 조회 실패: {error}")
        except Exception as e:
            print(f"❌ 설정 조회 오류: {e}")
        
        # 3. 개별 설정 업데이트 테스트
        settings_to_test = [
            ("use_voice_enhance", False),
            ("do_stt", True),
            ("classify_event", False),
            ("enhance_volume", 3)
        ]
        
        for setting_name, new_value in settings_to_test:
            try:
                async with session.put(f"{base_url}/v1/room/{room_name}/settings/{setting_name}",
                                     json=new_value) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"✅ {setting_name} = {new_value} 업데이트 성공")
                    else:
                        error = await resp.json()
                        print(f"❌ {setting_name} 업데이트 실패: {error}")
            except Exception as e:
                print(f"❌ {setting_name} 업데이트 오류: {e}")
        
        # 4. 전체 설정 조회 (업데이트 후)
        try:
            async with session.get(f"{base_url}/v1/room/{room_name}/settings") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    settings = data['settings']
                    print(f"📋 업데이트 후 설정:")
                    for key, value in settings.items():
                        print(f"   {key}: {value}")
        except Exception as e:
            print(f"❌ 업데이트 후 조회 오류: {e}")
        
        # 5. 설정 리셋 테스트
        try:
            async with session.post(f"{base_url}/v1/room/{room_name}/settings/reset") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print(f"✅ 설정 리셋 성공")
                    print(f"📋 기본 설정:")
                    for key, value in result['default_settings'].items():
                        print(f"   {key}: {value}")
                else:
                    error = await resp.json()
                    print(f"❌ 설정 리셋 실패: {error}")
        except Exception as e:
            print(f"❌ 설정 리셋 오류: {e}")

if __name__ == "__main__":
    print("💡 참고: Safety Server가 실행 중이어야 합니다.")
    print("   서버 시작: conda activate safety-server && python app.py")
    print()
    
    asyncio.run(test_room_settings())
