#!/usr/bin/env python3
"""
Safety Server WebSocket 클라이언트 (고급형)
파일 재생, 녹음, 실시간 분석 기능 포함
"""
import asyncio
import websockets
import numpy as np
import wave
import time
from datetime import datetime
from typing import Optional
import argparse

class AdvancedSafetyClient:
    def __init__(self, 
                 server_host: str = "localhost",
                 server_port: int = 24015,
                 room_name: str = "테스트방",
                 device_id: str = "advanced-client",
                 mode: str = "live"):  # live, file, test
        
        self.server_host = server_host
        self.server_port = server_port
        self.room_name = room_name
        self.device_id = device_id
        self.mode = mode
        self.sample_rate = 16000
        self.frame_size = 1024
        
        self.ws_url = f"ws://{server_host}:{server_port}/ws/room/{room_name}/{self.sample_rate}/int16/{device_id}"
        
        # 통계
        self.sent_frames = 0
        self.received_frames = 0
        self.start_time = time.time()
        self.received_audio = []  # 수신 오디오 저장
        
        print(f"�� 고급 Safety Client - {mode} 모드")

    async def create_room(self):
        """방 생성"""
        import aiohttp
        create_url = f"http://{self.server_host}:{self.server_port}/v1/create_room"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(create_url, json={"room_name": self.room_name}) as response:
                    result = await response.json()
                    print(f"✅ 방 생성: {result.get('message', 'Success')}")
            except Exception as e:
                print(f"⚠️ 방 생성 오류 (이미 존재할 수 있음): {e}")

    def generate_test_audio(self, duration_sec: float = 0.064) -> bytes:
        """테스트용 사인파 생성 (440Hz 톤)"""
        samples = int(self.sample_rate * duration_sec)
        t = np.linspace(0, duration_sec, samples, False)
        wave_data = np.sin(2 * np.pi * 440 * t)  # 440Hz A음
        audio_int16 = (wave_data * 16383).astype(np.int16)
        return audio_int16.tobytes()

    def load_wav_file(self, filename: str) -> np.ndarray:
        """WAV 파일 로드"""
        try:
            with wave.open(filename, 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                print(f"📁 파일 로드: {filename} ({len(audio)} 샘플)")
                return audio
        except Exception as e:
            print(f"❌ 파일 로드 실패: {e}")
            return np.array([], dtype=np.int16)

    async def file_mode_sender(self, websocket, filename: str):
        """파일 재생 모드"""
        audio_data = self.load_wav_file(filename)
        if len(audio_data) == 0:
            return
            
        print(f"🎵 파일 재생 시작: {filename}")
        
        for i in range(0, len(audio_data), self.frame_size):
            frame = audio_data[i:i + self.frame_size]
            if len(frame) < self.frame_size:
                # 마지막 프레임 패딩
                frame = np.pad(frame, (0, self.frame_size - len(frame)), 'constant')
            
            await websocket.send(frame.tobytes())
            self.sent_frames += 1
            await asyncio.sleep(0.064)  # 실제 재생 속도 유지

    async def live_mode_sender(self, websocket):
        """라이브 마이크 모드 (PyAudio 필요)"""
        try:
            import pyaudio
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
            
            print("🎤 라이브 마이크 모드")
            
            while True:
                audio_data = stream.read(self.frame_size, exception_on_overflow=False)
                await websocket.send(audio_data)
                self.sent_frames += 1
                await asyncio.sleep(0.01)
                
        except ImportError:
            print("❌ PyAudio가 설치되지 않음. 테스트 모드로 전환")
            await self.test_mode_sender(websocket)
        except Exception as e:
            print(f"❌ 라이브 모드 오류: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            if 'audio' in locals():
                audio.terminate()

    async def test_mode_sender(self, websocket):
        """테스트 톤 생성 모드"""
        print("🔊 테스트 톤 생성 (440Hz)")
        
        while True:
            test_audio = self.generate_test_audio()
            await websocket.send(test_audio)
            self.sent_frames += 1
            await asyncio.sleep(0.064)

    async def receiver_loop(self, websocket):
        """수신 루프 및 녹음"""
        print("👂 수신 시작")
        
        while True:
            try:
                audio_data = await websocket.recv()
                if isinstance(audio_data, bytes):
                    self.received_frames += 1
                    
                    # 수신 오디오 저장 (선택적)
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    self.received_audio.extend(audio_array)
                    
                    # 실시간 출력 (PyAudio 있는 경우)
                    try:
                        import pyaudio
                        if not hasattr(self, 'output_stream'):
                            audio = pyaudio.PyAudio()
                            self.output_stream = audio.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=self.sample_rate,
                                output=True,
                                frames_per_buffer=self.frame_size
                            )
                        self.output_stream.write(audio_data)
                    except:
                        pass  # PyAudio 없으면 조용히 넘어감
                        
            except Exception as e:
                print(f"❌ 수신 오류: {e}")
                break

    def save_received_audio(self, filename: str = None):
        """수신한 오디오를 파일로 저장"""
        if not self.received_audio:
            print("💾 저장할 오디오 없음")
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"received_audio_{timestamp}.wav"
            
        audio_array = np.array(self.received_audio, dtype=np.int16)
        
        with wave.open(filename, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_array.tobytes())
            
        duration = len(audio_array) / self.sample_rate
        print(f"💾 수신 오디오 저장: {filename} ({duration:.1f}초)")

    def print_statistics(self):
        """통계 출력"""
        elapsed = time.time() - self.start_time
        print(f"\n📊 통계 ({elapsed:.1f}초)")
        print(f"   송신: {self.sent_frames} 프레임")
        print(f"   수신: {self.received_frames} 프레임")
        print(f"   송신율: {self.sent_frames/elapsed:.1f} fps")
        print(f"   수신율: {self.received_frames/elapsed:.1f} fps")

    async def run(self, audio_file: str = None):
        """메인 실행"""
        try:
            await self.create_room()
            
            print(f"🔗 연결 중: {self.ws_url}")
            async with websockets.connect(self.ws_url, max_size=None) as websocket:
                print("✅ 연결 성공!")
                
                # 송신 태스크 선택
                if self.mode == "file" and audio_file:
                    send_task = asyncio.create_task(self.file_mode_sender(websocket, audio_file))
                elif self.mode == "live":
                    send_task = asyncio.create_task(self.live_mode_sender(websocket))
                else:
                    send_task = asyncio.create_task(self.test_mode_sender(websocket))
                
                # 수신 태스크
                recv_task = asyncio.create_task(self.receiver_loop(websocket))
                
                # 둘 다 실행
                await asyncio.gather(send_task, recv_task)
                
        except KeyboardInterrupt:
            print("\n🛑 사용자 중단")
        except Exception as e:
            print(f"❌ 오류: {e}")
        finally:
            self.print_statistics()
            self.save_received_audio()

def main():
    parser = argparse.ArgumentParser(description="Safety Server Advanced Client")
    parser.add_argument("--host", default="localhost", help="서버 호스트")
    parser.add_argument("--port", type=int, default=24015, help="서버 포트")
    parser.add_argument("--room", default="테스트방", help="방 이름")
    parser.add_argument("--device", default="advanced-client", help="디바이스 ID")
    parser.add_argument("--mode", choices=["live", "file", "test"], default="test", 
                       help="모드: live(마이크), file(파일재생), test(테스트톤)")
    parser.add_argument("--file", help="재생할 오디오 파일 (file 모드)")
    
    args = parser.parse_args()
    
    client = AdvancedSafetyClient(
        server_host=args.host,
        server_port=args.port,
        room_name=args.room,
        device_id=args.device,
        mode=args.mode
    )
    
    print("🚀 Advanced Safety Client")
    print("   Ctrl+C로 종료")
    
    asyncio.run(client.run(args.file))

if __name__ == "__main__":
    main()
