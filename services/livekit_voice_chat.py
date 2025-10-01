import asyncio
import time
import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import jwt
from livekit import rtc
from sqlmodel import Session, select

from database import engine
from models import Rooms, Devices
from services.audio_utils import AudioUtils
from services.stt import SttProcessor
from services.voice_enhance import VoiceEnhancer, LightVoiceEnhancer
from utils.sys import aprint


class LiveKitVoiceChat(AudioUtils):
    """
    LiveKit 기반 음성통신 서비스
    - LiveKit 서버를 통한 실시간 음성통신
    - WebRTC 기반의 안정적인 연결
    - 기존 음성 향상 및 STT 기능 통합
    """

    def __init__(self, livekit_url: str = "ws://localhost:7880", 
                 api_key: str = "devkey", api_secret: str = "secret"):
        super().__init__()
        
        # LiveKit 설정
        self.livekit_url = livekit_url
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 방 관리
        self.rooms: Dict[str, rtc.Room] = {}  # {room_name: Room}
        self.room_participants: Dict[str, Dict[str, rtc.RemoteParticipant]] = {}  # {room_name: {identity: participant}}
        self.room_audio_buffers: Dict[str, Dict[str, List[np.ndarray]]] = {}  # {room_name: {identity: [audio_frames]}}
        
        # 클라이언트 정보
        self.client_info: Dict[str, dict] = {}  # {identity: {sr, dtype, person_name}}
        
        # STT 프로세서
        self.stt_processor = SttProcessor()
        
        # 소음제거 모델 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.voice_enhancer = VoiceEnhancer(self.device)
        self.light_enhancer = LightVoiceEnhancer(self.device)
        
        # 참가자별 소음제거 설정
        self.participant_enhance_settings: Dict[str, dict] = {}  # {identity: {enabled: bool, type: str}}
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 방별 설정 캐시
        self.room_settings_cache: Dict[str, dict] = {}
        
        # 전역 기본값
        self.default_settings = {
            'use_voice_enhance': True,
            'hear_me': False,
            'record_audio': True,
            'classify_event': True,
            'do_stt': True,
            'enhance_volume': 0,
            'keep_test_room': True
        }

    def generate_token(self, room_name: str, participant_name: str, 
                      participant_identity: str = None, 
                      can_publish: bool = True, can_subscribe: bool = True) -> str:
        """LiveKit JWT 토큰 생성"""
        try:
            now = int(time.time())
            identity = participant_identity or participant_name
            
            payload = {
                "iss": self.api_key,
                "sub": identity,
                "iat": now,
                "exp": now + 3600,  # 1시간 후 만료
                "nbf": now,
                "name": participant_name,
                "video": {
                    "room": room_name,
                    "roomJoin": True,
                    "canPublish": can_publish,
                    "canSubscribe": can_subscribe,
                    "canPublishData": True,
                    "hidden": False,
                    "recorder": False
                }
            }
            
            token = jwt.encode(payload, self.api_secret, algorithm="HS256")
            return token
            
        except Exception as e:
            aprint(f"토큰 생성 오류: {e}")
            return None

    async def create_room(self, room_name: str) -> dict:
        """방 생성"""
        try:
            if room_name not in self.rooms:
                # 새 방 생성
                room = rtc.Room()
                self.rooms[room_name] = room
                self.room_participants[room_name] = {}
                self.room_audio_buffers[room_name] = {}
                
                # DB에 방 정보 저장
                with Session(engine) as session:
                    existing_room = session.exec(
                        select(Rooms).where(Rooms.room_name == room_name)
                    ).first()
                    
                    if not existing_room:
                        new_room = Rooms(
                            room_name=room_name,
                            num_person=0,
                            persons=""
                        )
                        session.add(new_room)
                        session.commit()
                
                aprint(f"LiveKit 방 '{room_name}' 생성 완료")
                return {"message": f"Room '{room_name}' created successfully"}
            else:
                return {"message": f"Room '{room_name}' already exists"}
                
        except Exception as e:
            aprint(f"방 생성 오류: {e}")
            return {"error": f"Failed to create room: {e}"}

    async def delete_room(self, room_id: int) -> dict:
        """방 삭제"""
        try:
            with Session(engine) as session:
                room = session.exec(select(Rooms).filter(Rooms.id == room_id)).first()
                if not room:
                    return {"error": "Room not found"}
                
                room_name = room.room_name
                
                # LiveKit 방 정리
                if room_name in self.rooms:
                    livekit_room = self.rooms[room_name]
                    try:
                        await livekit_room.disconnect()
                    except:
                        pass
                    
                    self.rooms.pop(room_name, None)
                    self.room_participants.pop(room_name, None)
                    self.room_audio_buffers.pop(room_name, None)
                
                # DB에서 방 삭제
                session.delete(room)
                session.commit()
                
                aprint(f"LiveKit 방 '{room_name}' 삭제 완료")
                return {"message": f"Room '{room_name}' deleted successfully"}
                
        except Exception as e:
            aprint(f"방 삭제 오류: {e}")
            return {"error": f"Failed to delete room: {e}"}

    async def join_room(self, room_name: str, participant_name: str, 
                       participant_identity: str = None, device_id: str = None) -> dict:
        """방 참가"""
        try:
            if room_name not in self.rooms:
                await self.create_room(room_name)
            
            identity = participant_identity or participant_name
            token = self.generate_token(room_name, participant_name, identity)
            
            if not token:
                return {"error": "Failed to generate token"}
            
            # 클라이언트 정보 저장
            self.client_info[identity] = {
                "sr": 16000,
                "dtype": "int16",
                "person_name": participant_name,
                "device_id": device_id
            }
            
            # DB 업데이트
            with Session(engine) as session:
                room = session.exec(select(Rooms).where(Rooms.room_name == room_name)).first()
                if room:
                    room.num_person += 1
                    if room.persons:
                        room.persons += f", {participant_name}"
                    else:
                        room.persons = participant_name
                    session.add(room)
                    session.commit()
            
            return {
                "token": token,
                "url": self.livekit_url,
                "room_name": room_name,
                "participant_name": participant_name,
                "participant_identity": identity
            }
            
        except Exception as e:
            aprint(f"방 참가 오류: {e}")
            return {"error": f"Failed to join room: {e}"}

    def set_participant_enhancement(self, participant_identity: str, enabled: bool = True, 
                                  enhancement_type: str = "light") -> dict:
        """참가자별 소음제거 설정"""
        try:
            if enhancement_type not in ["light", "full"]:
                return {"error": "Invalid enhancement type. Use 'light' or 'full'"}
            
            self.participant_enhance_settings[participant_identity] = {
                "enabled": enabled,
                "type": enhancement_type
            }
            
            # 스트리머 설정
            if enabled:
                if enhancement_type == "full":
                    self.voice_enhancer.add_streamer(hash(participant_identity))
                else:
                    self.light_enhancer.add_streamer(hash(participant_identity))
            else:
                # 스트리머 제거
                if enhancement_type == "full":
                    self.voice_enhancer.remove_streamer(hash(participant_identity))
                else:
                    self.light_enhancer.remove_streamer(hash(participant_identity))
            
            aprint(f"참가자 {participant_identity} 소음제거 설정: {enhancement_type} {'활성화' if enabled else '비활성화'}")
            return {
                "message": f"Enhancement settings updated for {participant_identity}",
                "enabled": enabled,
                "type": enhancement_type
            }
            
        except Exception as e:
            aprint(f"소음제거 설정 오류: {e}")
            return {"error": f"Failed to set enhancement: {e}"}

    def get_participant_enhancement(self, participant_identity: str) -> dict:
        """참가자별 소음제거 설정 조회"""
        settings = self.participant_enhance_settings.get(participant_identity, {
            "enabled": True,
            "type": "light"
        })
        return {
            "participant_identity": participant_identity,
            "enabled": settings["enabled"],
            "type": settings["type"]
        }

    def apply_enhancement(self, audio_data: np.ndarray, participant_identity: str) -> np.ndarray:
        """오디오 데이터에 소음제거 적용"""
        try:
            settings = self.participant_enhance_settings.get(participant_identity, {
                "enabled": True,
                "type": "light"
            })
            
            if not settings["enabled"]:
                return audio_data
            
            # 오디오 데이터를 torch 텐서로 변환
            if audio_data.dtype == np.int16:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
            else:
                audio_tensor = torch.from_numpy(audio_data.astype(np.float32))
            
            # 소음제거 적용
            if settings["type"] == "full":
                enhanced_tensor = self.voice_enhancer.denoise(audio_tensor, hash(participant_identity))
            else:
                enhanced_tensor = self.light_enhancer.denoise(audio_data, hash(participant_identity))
            
            # 결과를 numpy 배열로 변환
            if isinstance(enhanced_tensor, torch.Tensor):
                enhanced_audio = enhanced_tensor.cpu().numpy()
            else:
                enhanced_audio = enhanced_tensor
            
            # int16으로 변환
            if enhanced_audio.dtype != np.int16:
                enhanced_audio = (enhanced_audio * 32768.0).astype(np.int16)
            
            return enhanced_audio
            
        except Exception as e:
            aprint(f"소음제거 적용 오류: {e}")
            return audio_data  # 오류 시 원본 반환

    async def setup_room_handlers(self, room_name: str):
        """방 이벤트 핸들러 설정"""
        if room_name not in self.rooms:
            return
        
        room = self.rooms[room_name]
        
        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            aprint(f"[{room_name}] 참가자 연결: {participant.identity}")
            self.room_participants[room_name][participant.identity] = participant
            self.room_audio_buffers[room_name][participant.identity] = []
        
        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            aprint(f"[{room_name}] 참가자 연결 해제: {participant.identity}")
            self.room_participants[room_name].pop(participant.identity, None)
            self.room_audio_buffers[room_name].pop(participant.identity, None)
            
            # DB 업데이트
            try:
                with Session(engine) as session:
                    room_db = session.exec(select(Rooms).where(Rooms.room_name == room_name)).first()
                    if room_db:
                        room_db.num_person = max(0, room_db.num_person - 1)
                        person_name = self.client_info.get(participant.identity, {}).get("person_name", "")
                        if person_name and person_name in room_db.persons:
                            persons = room_db.persons.replace(f", {person_name}", "").replace(f"{person_name}, ", "")
                            if persons.endswith(person_name):
                                persons = persons[:-len(person_name)]
                            room_db.persons = persons.strip().strip(",").strip()
                        session.add(room_db)
                        session.commit()
            except Exception as e:
                aprint(f"DB 업데이트 오류: {e}")
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, 
                               participant: rtc.RemoteParticipant):
            aprint(f"[{room_name}] 트랙 구독: {track.kind} from {participant.identity}")
            
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                self.handle_audio_track(track, participant, room_name)
        
        @room.on("track_unsubscribed")
        def on_track_unsubscribed(track: rtc.Track, publication: rtc.TrackPublication, 
                                 participant: rtc.RemoteParticipant):
            aprint(f"[{room_name}] 트랙 구독 해제: {track.kind} from {participant.identity}")

    def handle_audio_track(self, track: rtc.Track, participant: rtc.RemoteParticipant, room_name: str):
        """오디오 트랙 처리"""
        try:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            
            # 오디오 프레임 처리
            @track.on("data_received")
            def on_audio_data(frame: rtc.AudioFrame):
                try:
                    # 오디오 프레임을 numpy 배열로 변환
                    audio_data = np.frombuffer(frame.data, dtype=np.int16)
                    
                    # 소음제거 적용
                    enhanced_audio = self.apply_enhancement(audio_data, participant.identity)
                    
                    # 버퍼에 저장 (향상된 오디오)
                    if participant.identity in self.room_audio_buffers[room_name]:
                        self.room_audio_buffers[room_name][participant.identity].append(enhanced_audio)
                        
                        # 버퍼 크기 제한 (최대 100 프레임)
                        if len(self.room_audio_buffers[room_name][participant.identity]) > 100:
                            self.room_audio_buffers[room_name][participant.identity].pop(0)
                    
                    # STT 처리 (선택사항)
                    if self.do_stt and len(audio_data) > 0:
                        asyncio.create_task(self.process_stt(audio_data, room_name, participant.identity))
                        
                except Exception as e:
                    aprint(f"오디오 데이터 처리 오류: {e}")
        
        except Exception as e:
            aprint(f"오디오 트랙 핸들러 설정 오류: {e}")

    async def process_stt(self, audio_data: np.ndarray, room_name: str, participant_identity: str):
        """STT 처리"""
        try:
            person_name = self.client_info.get(participant_identity, {}).get("person_name", "unknown")
            audio_bytes = audio_data.astype(np.int16).tobytes()
            await self.stt_processor.send_audio(audio_bytes, room_name, person_name, participant_identity)
        except Exception as e:
            aprint(f"STT 처리 오류: {e}")

    async def get_room_participants(self, room_name: str) -> List[dict]:
        """방 참가자 목록 조회"""
        try:
            if room_name not in self.room_participants:
                return []
            
            participants = []
            for identity, participant in self.room_participants[room_name].items():
                client_info = self.client_info.get(identity, {})
                participants.append({
                    "identity": identity,
                    "name": participant.name or identity,
                    "person_name": client_info.get("person_name", identity),
                    "is_speaking": participant.is_speaking,
                    "connection_quality": participant.connection_quality.name if participant.connection_quality else "unknown"
                })
            
            return participants
            
        except Exception as e:
            aprint(f"참가자 목록 조회 오류: {e}")
            return []

    async def get_rooms(self) -> List[dict]:
        """방 목록 조회"""
        try:
            rooms = []
            for room_name in self.rooms.keys():
                participants = await self.get_room_participants(room_name)
                rooms.append({
                    "room_name": room_name,
                    "num_person": len(participants),
                    "participants": participants
                })
            return rooms
            
        except Exception as e:
            aprint(f"방 목록 조회 오류: {e}")
            return []

    async def broadcast_audio_to_room(self, room_name: str, audio_data: np.ndarray, 
                                    exclude_identity: str = None):
        """방에 오디오 브로드캐스트"""
        try:
            if room_name not in self.rooms:
                return
            
            room = self.rooms[room_name]
            
            # 모든 참가자에게 오디오 전송 (자신 제외)
            for identity, participant in self.room_participants[room_name].items():
                if exclude_identity and identity == exclude_identity:
                    continue
                
                try:
                    # 오디오 프레임 생성
                    audio_frame = rtc.AudioFrame(
                        data=audio_data.tobytes(),
                        sample_rate=16000,
                        num_channels=1,
                        samples_per_channel=len(audio_data)
                    )
                    
                    # 데이터 채널을 통해 전송 (실제 구현에서는 적절한 방법 사용)
                    # participant.send_data(audio_frame.data)
                    
                except Exception as e:
                    aprint(f"오디오 브로드캐스트 오류 ({identity}): {e}")
        
        except Exception as e:
            aprint(f"방 브로드캐스트 오류: {e}")

    def get_room_settings(self, room_name: str) -> dict:
        """방별 설정 가져오기"""
        if room_name in self.room_settings_cache:
            return self.room_settings_cache[room_name]
        return self.default_settings.copy()

    def update_room_settings(self, room_name: str, **kwargs) -> bool:
        """방 설정 업데이트"""
        try:
            if room_name not in self.room_settings_cache:
                self.room_settings_cache[room_name] = self.default_settings.copy()
            
            self.room_settings_cache[room_name].update(kwargs)
            aprint(f"방 '{room_name}' 설정 업데이트: {kwargs}")
            return True
            
        except Exception as e:
            aprint(f"방 설정 업데이트 오류: {e}")
            return False

    async def cleanup(self):
        """리소스 정리"""
        try:
            for room_name, room in self.rooms.items():
                try:
                    await room.disconnect()
                except:
                    pass
            
            self.rooms.clear()
            self.room_participants.clear()
            self.room_audio_buffers.clear()
            self.client_info.clear()
            
            aprint("LiveKit 리소스 정리 완료")
            
        except Exception as e:
            aprint(f"리소스 정리 오류: {e}")
