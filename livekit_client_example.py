#!/usr/bin/env python3
"""
LiveKit 클라이언트 예제
이 예제는 LiveKit 서버에 연결하여 음성통신을 테스트하는 방법을 보여줍니다.
"""

import asyncio
import logging
import aiohttp
import json
from livekit import rtc
from utils.sys import aprint

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitTestClient:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.room = rtc.Room()
        self.local_audio_track = None
        
    async def get_token(self, room_name: str, participant_name: str, participant_identity: str = None):
        """API 서버에서 토큰 가져오기"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.api_base_url}/api/livekit/join"
                data = {
                    "room_name": room_name,
                    "participant_name": participant_name,
                    "participant_identity": participant_identity
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        aprint(f"토큰 요청 실패: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            aprint(f"토큰 요청 오류: {e}")
            return None
    
    async def connect(self, room_name: str, participant_name: str, participant_identity: str = None):
        """LiveKit 서버에 연결"""
        try:
            # 토큰 가져오기
            token_info = await self.get_token(room_name, participant_name, participant_identity)
            if not token_info:
                aprint("토큰 가져오기 실패")
                return False
            
            # 방에 연결
            await self.room.connect(token_info["url"], token_info["token"])
            aprint(f"LiveKit 서버에 연결되었습니다: {participant_name}")
            
            # 이벤트 리스너 설정
            self.room.on("participant_connected", self.on_participant_connected)
            self.room.on("participant_disconnected", self.on_participant_disconnected)
            self.room.on("track_subscribed", self.on_track_subscribed)
            self.room.on("track_unsubscribed", self.on_track_unsubscribed)
            
            return True
            
        except Exception as e:
            aprint(f"연결 실패: {e}")
            return False
    
    async def disconnect(self):
        """연결 해제"""
        try:
            await self.room.disconnect()
            aprint("연결이 해제되었습니다.")
        except Exception as e:
            aprint(f"연결 해제 오류: {e}")
    
    async def publish_microphone(self):
        """마이크 오디오 발행"""
        try:
            # 마이크 트랙 생성
            audio_source = rtc.AudioSource(16000, 1)  # 16kHz, 모노
            self.local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "microphone", audio_source
            )
            
            # 트랙 발행
            await self.room.local_participant.publish_track(
                self.local_audio_track, rtc.TrackPublishOptions()
            )
            aprint("마이크 오디오가 발행되었습니다.")
            
        except Exception as e:
            aprint(f"마이크 발행 오류: {e}")
    
    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 이벤트"""
        aprint(f"참가자가 연결되었습니다: {participant.identity}")
    
    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """참가자 연결 해제 이벤트"""
        aprint(f"참가자가 연결 해제되었습니다: {participant.identity}")
    
    def on_track_subscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 이벤트"""
        aprint(f"트랙이 구독되었습니다: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            self.handle_audio_track(track, participant)
    
    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """트랙 구독 해제 이벤트"""
        aprint(f"트랙 구독이 해제되었습니다: {track.kind} from {participant.identity}")
    
    def handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant):
        """오디오 트랙 처리"""
        aprint(f"오디오 트랙을 처리합니다: {participant.identity}")
        
        @track.on("data_received")
        def on_audio_data(frame: rtc.AudioFrame):
            # 오디오 프레임 처리
            aprint(f"오디오 데이터 수신: {len(frame.data)} bytes from {participant.identity}")

async def main():
    """메인 함수"""
    # 클라이언트 생성
    client = LiveKitTestClient()
    
    try:
        # 연결
        room_name = "test-room"
        participant_name = "Python Test Client"
        
        if await client.connect(room_name, participant_name):
            aprint("연결 성공! 30초 동안 대기합니다...")
            
            # 마이크 발행 (선택사항)
            # await client.publish_microphone()
            
            # 30초 대기
            await asyncio.sleep(30)
            
        else:
            aprint("연결에 실패했습니다.")
            
    except KeyboardInterrupt:
        aprint("사용자에 의해 중단되었습니다.")
    finally:
        # 연결 해제
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
