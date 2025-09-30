#!/usr/bin/env python3
"""
Safety Server WebSocket 연결 테스트
"""
import asyncio
import websockets
import numpy as np
import time

async def test_websocket_connection():
    """WebSocket 연결 테스트"""
    
    # 테스트 설정
    host = "localhost"
    port = 24015
    room_name = "테스트방"
    device_id = "test-client"
    sample_rate = 16000
    dtype = "int16"
    
    ws_url = f"ws://{host}:{port}/ws/room/{room_name}/{sample_rate}/{dtype}/{device_id}"
    
    print(f"🔗 WebSocket 연결 테스트: {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket 연결 성공!")
            
            # 테스트 오디오 데이터 생성 (440Hz 사인파)
            duration = 0.064  # 64ms
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            sine_wave = np.sin(2 * np.pi * 440 * t)
            audio_int16 = (sine_wave * 16383).astype(np.int16)
            
            print(f"🎵 테스트 오디오 생성: {len(audio_int16)} 샘플")
            
            # 10초간 오디오 송수신 테스트
            sent_frames = 0
            received_frames = 0
            start_time = time.time()
            
            async def sender():
                nonlocal sent_frames
                while True:
                    await websocket.send(audio_int16.tobytes())
                    sent_frames += 1
                    await asyncio.sleep(0.064)  # 64ms 간격
                    
            async def receiver():
                nonlocal received_frames
                async for message in websocket:
                    received_frames += 1
                    if received_frames % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"📊 {elapsed:.1f}초: 송신={sent_frames}, 수신={received_frames}")
            
            # 10초간 테스트
            try:
                await asyncio.wait_for(
                    asyncio.gather(sender(), receiver()),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                print(f"⏰ 테스트 완료 ({elapsed:.1f}초)")
                print(f"📈 최종 결과: 송신={sent_frames}, 수신={received_frames}")
                
                if received_frames > 0:
                    print("✅ WebSocket 스트리밍 정상 작동!")
                else:
                    print("⚠️ 수신 프레임 없음 - 서버 확인 필요")
            
    except Exception as e:
        print(f"❌ 연결 실패: {e}")

if __name__ == "__main__":
    print("🚀 Safety Server WebSocket 스트리밍 테스트")
    asyncio.run(test_websocket_connection())
