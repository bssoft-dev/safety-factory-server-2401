#!/usr/bin/env python3
"""
라즈베리파이용 LiveKit 음성통신 클라이언트
기존 WebSocket 기반 시스템을 LiveKit으로 전환한 버전
"""

import asyncio
import aiohttp
import sounddevice as sd
import numpy as np
import soundfile as sf
from collections import deque
import os
import jwt
from livekit import rtc

# 오디오 설정
CHANNELS = 1
DTYPE = np.int16
RECONNECT_DELAY = 3

class RaspberryPiLiveKitClient:
    def __init__(self):
        # 설정값 (실제 환경에서는 config.txt에서 읽어옴)
        self.livekit_url = "wss://livekit.bs-soft.co.kr"  # 실제 서버 도메인 (포트 443 사용)
        self.api_base_url = "https://safety-server.bs-soft.co.kr"  # 실제 서버 도메인
        self.room_name = "safety-room"
        self.participant_name = "RaspberryPi-001"
        self.participant_identity = "raspberry-001"
        
        # 오디오 설정
        self.RATE = 16000
        self.CHUNK = 1024
        self.IN_DEVICE_NUM = 0  # 마이크 장치 번호
        self.OUT_DEVICE_NUM = 0  # 스피커 장치 번호
        self.BUFFER_SIZE = int(self.RATE / self.CHUNK) * 2
        
        # LiveKit 객체
        self.room = rtc.Room()
        self.local_audio_track = None
        
        # 오디오 스트림
        self.input_stream = None
        self.output_stream = None
        self.input_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.output_buffer = deque(maxlen=self.BUFFER_SIZE)
        
        # 상태 관리
        self.is_running = True
        self.loop = asyncio.get_event_loop()
        
        # 알림음 로드 (파일이 있는 경우)
        try:
            self.dangerwav = sf.read(f'./wav/danger{self.RATE}.wav', dtype="int16")[0].reshape(-1, CHANNELS)
            self.disconnectwav = sf.read(f'./wav/disconnect{self.RATE}.wav', dtype="int16")[0].reshape(-1, CHANNELS)
        except:
            # 알림음 파일이 없으면 무음 생성
            self.dangerwav = np.zeros((self.RATE, CHANNELS), dtype=np.int16)
            self.disconnectwav = np.zeros((self.RATE, CHANNELS), dtype=np.int16)

    async def get_token(self):
        """API 서버에서 JWT 토큰 가져오기"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/api/livekit/join"
                data = {
                    "room_name": self.room_name,
                    "participant_name": self.participant_name,
                    "participant_identity": self.participant_identity
                }
                
                print(f"토큰 요청 중: {url}")
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"토큰 획득 성공: {len(result.get('token', ''))}자")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"토큰 요청 실패: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"토큰 요청 오류: {e}")
            return None

    async def connect(self):
        """LiveKit 서버에 연결"""
        while self.is_running:
            try:
                print("LiveKit 서버에 연결 시도 중...")
                
                # 토큰 가져오기
                token_info = await self.get_token()
                if not token_info:
                    print("토큰 가져오기 실패, 재시도 중...")
                    await asyncio.sleep(RECONNECT_DELAY)
                    continue
                
                # LiveKit 방에 연결
                await self.room.connect(token_info["url"], token_info["token"])
                print(f"LiveKit 서버에 연결되었습니다: {self.participant_name}")
                
                # 이벤트 핸들러 설정
                self.room.on("participant_connected", self.on_participant_connected)
                self.room.on("participant_disconnected", self.on_participant_disconnected)
                self.room.on("track_subscribed", self.on_track_subscribed)
                self.room.on("track_unsubscribed", self.on_track_unsubscribed)
                
                return True
                
            except Exception as e:
                print(f"LiveKit 연결 실패: {e}")
                await asyncio.sleep(RECONNECT_DELAY)
        
        return False

    async def publish_microphone(self):
        """마이크 오디오 발행"""
        try:
            # LiveKit 오디오 소스 생성
            audio_source = rtc.AudioSource(self.RATE, CHANNELS)
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "microphone", audio_source
            )
            
            # 트랙 발행
            await self.room.local_participant.publish_track(
                self.local_audio_track, rtc.TrackPublishOptions()
            )
            print("마이크 오디오가 발행되었습니다.")
            
        except Exception as e:
            print(f"마이크 발행 오류: {e}")

    def input_callback(self, indata, frames, time, status):
        """마이크 입력 콜백"""
        try:
            if status:
                print(f"Input status: {status}")
            
            # LiveKit으로 오디오 데이터 전송
            if self.local_audio_track:
                audio_data = indata.flatten().astype(np.float32) / 32768.0
                self.local_audio_track.push_frame(audio_data)
            
            # 로컬 버퍼에도 저장 (필요시)
            self.input_buffer.append(indata.copy())
                
        except Exception as e:
            print(f"Error in input callback: {e}")

    def audio_callback(self, outdata, frames, time, status):
        """스피커 출력 콜백"""
        if status:
            print(f"Output status: {status}")
        
        try:
            if len(self.output_buffer) > 0:
                np_audio_data = self.output_buffer.popleft()
                if len(np_audio_data) < len(outdata):
                    outdata[:len(np_audio_data)] = np_audio_data
                    outdata[len(np_audio_data):] = np.zeros((len(outdata) - len(np_audio_data), CHANNELS), dtype=DTYPE)
                else:
                    outdata[:] = np_audio_data[:len(outdata)]
            else:
                outdata[:] = np.zeros((len(outdata), CHANNELS), dtype=DTYPE)
                
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata[:] = np.zeros((len(outdata), CHANNELS), dtype=DTYPE)

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 이벤트"""
        print(f"참가자가 연결되었습니다: {participant.identity}")

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 해제 이벤트"""
        print(f"참가자가 연결 해제되었습니다: {participant.identity}")

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 이벤트"""
        print(f"트랙이 구독되었습니다: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self.handle_audio_track(track, participant)

    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 해제 이벤트"""
        print(f"트랙 구독이 해제되었습니다: {track.kind} from {participant.identity}")

    def handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """오디오 트랙 처리"""
        print(f"오디오 트랙을 처리합니다: {participant.identity}")
        
        @track.on("data_received")
        def on_audio_data(frame: rtc.AudioFrame):
            try:
                # 오디오 프레임을 numpy 배열로 변환
                audio_data = np.frombuffer(frame.data, dtype=np.int16)
                audio_array = audio_data.reshape(-1, CHANNELS)
                
                # 출력 버퍼에 추가
                self.write_audio(audio_array)
                
            except Exception as e:
                print(f"오디오 데이터 처리 오류: {e}")

    def write_audio(self, audio_array):
        """오디오 데이터를 출력 버퍼에 쓰기"""
        for i in range(0, len(audio_array), self.CHUNK):
            self.output_buffer.append(audio_array[i:i+self.CHUNK])

    def write_np_audio(self, audio_array, start_idx, write_len):
        """numpy 오디오 배열을 출력 버퍼에 쓰기"""
        if start_idx + write_len < audio_array.shape[0]:
            for i in range(0, write_len, self.CHUNK):
                self.output_buffer.append(audio_array[start_idx+i:start_idx+i+self.CHUNK])
            return False
        else:
            self.output_buffer.append(audio_array[start_idx:])
            return True
        
    async def play_np_audio(self, np_audio):
        """numpy 오디오 배열 재생"""
        play_idx = 0
        audio_end = False
        while not audio_end:
            play_len = self.CHUNK * self.BUFFER_SIZE // 2
            audio_end = self.write_np_audio(np_audio, play_idx, play_len)
            play_idx = play_idx + play_len
            while len(self.output_buffer) > 3:  # 버퍼 크기 충분할 때까지 대기
                await asyncio.sleep(0.001)

    async def danger_monitor(self):
        """위험 신호 모니터링"""
        while self.is_running:
            if os.path.exists("./danger.sig"):
                os.remove("./danger.sig")
                print("위험 신호 감지! 경보음 재생")
                # 버퍼 클리어
                for _ in range(len(self.output_buffer)):
                    if self.output_buffer:
                        self.output_buffer.popleft()
                # 경보음 재생
                await self.play_np_audio(self.dangerwav)
            await asyncio.sleep(0.1)

    async def run(self):
        """메인 실행 함수"""
        print("라즈베리파이 LiveKit 클라이언트 시작")
        
        if not await self.connect():
            print("연결 실패")
            return
        
        # 오디오 스트림 시작
        print("오디오 스트림 시작")
        self.input_stream = sd.InputStream(
            device=self.IN_DEVICE_NUM,
            samplerate=self.RATE, blocksize=self.CHUNK, channels=CHANNELS,
            dtype=DTYPE, callback=self.input_callback
        )
        self.output_stream = sd.OutputStream(
            device=self.OUT_DEVICE_NUM,
            samplerate=self.RATE, blocksize=self.CHUNK, channels=CHANNELS,
            dtype=DTYPE, callback=self.audio_callback
        )
        
        self.input_stream.start()
        self.output_stream.start()
        
        # 마이크 발행
        await self.publish_microphone()
        
        print("음성통신 시작! Ctrl+C로 종료")
        
        # 태스크 실행
        try:
            await asyncio.gather(
                self.danger_monitor(),
            )
        except KeyboardInterrupt:
            print("사용자에 의해 중단됨")
        
    def close(self):
        """리소스 정리"""
        print("LiveKit Audio Client 종료")
        self.is_running = False
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        if self.room:
            asyncio.create_task(self.room.disconnect())

async def main():
    """메인 함수"""
    client = RaspberryPiLiveKitClient()
    try:
        await client.run()
    except KeyboardInterrupt:
        print("클라이언트를 종료합니다.")
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())
