#!/usr/bin/env python3
"""
Safety Server WebSocket 클라이언트 (기본형)
실시간 오디오 스트리밍 송수신
"""
import asyncio
import websockets
import numpy as np
import pyaudio
import time
from typing import Optional

class SafetyServerClient:
    def __init__(self, 
                 server_host: str = "localhost",
                 server_port: int = 24015,
                 room_name: str = "보온팀",
                 device_id: str = "python-client-001",
                 sample_rate: int = 16000,
                 dtype: str = "int16"):
        
        self.server_host = server_host
        self.server_port = server_port
        self.room_name = room_name
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.frame_size = 1024  # Safety Server의 FRAME_SIZE와 동일
        
        # WebSocket URL 구성
        self.ws_url = f"ws://{server_host}:{server_port}/ws/room/{room_name}/{sample_rate}/{dtype}/{device_id}"
        
        # PyAudio 설정
        self.audio = pyaudio.PyAudio()
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
        print(f"🎵 Safety Server Client 초기화")
        print(f"   서버: {self.ws_url}")
        print(f"   샘플레이트: {sample_rate}Hz, 프레임크기: {self.frame_size}")

    async def create_room(self):
        """서버에 방 생성 요청"""
        import aiohttp
        create_url = f"http://{self.server_host}:{self.server_port}/v1/create_room"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(create_url, json={"room_name": self.room_name}) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"✅ 방 생성 성공: {result}")
                    else:
                        print(f"⚠️ 방 생성 응답: {response.status}")
            except Exception as e:
                print(f"❌ 방 생성 실패: {e}")

    def setup_audio(self):
        """오디오 입출력 스트림 설정"""
        try:
            # 마이크 입력 스트림
            self.input_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
            
            # 스피커 출력 스트림  
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.frame_size
            )
            
            print("🎤 오디오 스트림 설정 완료")
            
        except Exception as e:
            print(f"❌ 오디오 설정 실패: {e}")
            raise

    async def send_audio_loop(self):
        """마이크 → 서버 전송 루프"""
        try:
            while self.websocket and not self.websocket.closed:
                # 마이크에서 오디오 읽기
                audio_data = self.input_stream.read(self.frame_size, exception_on_overflow=False)
                
                # 서버로 전송
                await self.websocket.send(audio_data)
                
                # 프레임 간격 유지 (1024/16000 ≈ 64ms)
                await asyncio.sleep(0.064)
                
        except Exception as e:
            print(f"❌ 송신 오류: {e}")

    async def receive_audio_loop(self):
        """서버 → 스피커 수신 루프"""
        try:
            while self.websocket and not self.websocket.closed:
                # 서버에서 오디오 받기
                audio_data = await self.websocket.recv()
                
                # 스피커로 출력
                if isinstance(audio_data, bytes) and len(audio_data) > 0:
                    self.output_stream.write(audio_data)
                    
        except Exception as e:
            print(f"❌ 수신 오류: {e}")

    async def connect_and_stream(self):
        """WebSocket 연결 및 스트리밍 시작"""
        try:
            # 1. 방 생성 시도
            await self.create_room()
            
            # 2. 오디오 설정
            self.setup_audio()
            
            # 3. WebSocket 연결
            print(f"🔗 서버 연결 중...")
            async with websockets.connect(self.ws_url, max_size=None) as websocket:
                self.websocket = websocket
                print(f"✅ 연결 성공! 실시간 스트리밍 시작")
                
                # 4. 송신/수신 동시 실행
                await asyncio.gather(
                    self.send_audio_loop(),
                    self.receive_audio_loop()
                )
                
        except KeyboardInterrupt:
            print("\n🛑 사용자 중단")
        except Exception as e:
            print(f"❌ 연결 오류: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        if self.audio:
            self.audio.terminate()
        print("🧹 리소스 정리 완료")

async def main():
    # 클라이언트 생성 및 실행
    client = SafetyServerClient(
        server_host="localhost",  # Safety Server 주소
        room_name="보온팀",
        device_id="python-client-001"
    )
    
    await client.connect_and_stream()

if __name__ == "__main__":
    print("🚀 Safety Server Python Client")
    print("   Ctrl+C로 종료")
    asyncio.run(main())
