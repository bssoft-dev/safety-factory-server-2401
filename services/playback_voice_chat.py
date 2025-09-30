import asyncio
import time
import os
import glob
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
import wave
from fastapi import WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from database import engine
from models import Rooms, Devices
from utils.sys import aprint


class PlaybackVoiceChat:
    """
    recordings 폴더의 WAV 파일을 재생하는 간단한 음성통신 시스템
    - 클라이언트가 방에 입장하면 자동으로 WAV 파일 재생
    - 실시간 마이크 입력 대신 녹음 파일 사용
    """

    # ====== 오디오 관련 파라미터 ======
    FRAME_SIZE = 960  # 20ms @ 48kHz
    SYNC_INTERVAL = 0.02  # 20ms
    QUEUE_MAXLEN_FRAMES = 50  # 클라이언트별 입력 큐 최대 프레임 수

    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}                    # {room_name: [WebSocket, ...]}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {}           # {room_name: {client_id: deque[(ts, frame), ...]}}
        self.client_info: Dict[int, dict] = {}                         # {client_id: {sr, dtype, is_webbrowser, person_name}}
        self.room_tasks: Dict[str, asyncio.Task] = {}                  # 방 처리 태스크 핸들
        self.playback_tasks: Dict[int, asyncio.Task] = {}              # 재생 태스크 핸들
        self.playback_data: Dict[int, List[np.ndarray]] = {}           # 재생할 오디오 데이터

    # ========== WAV 파일 로드 ==========
    def load_wav_file(self, file_path: str) -> List[np.ndarray]:
        """WAV 파일을 로드하여 프레임 단위로 분할"""
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # WAV 파일 정보
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # 오디오 데이터 읽기
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # 바이트를 numpy 배열로 변환
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.int32)
                else:
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8)
                
                # 스테레오를 모노로 변환
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # 48kHz로 리샘플링 (필요한 경우)
                if sample_rate != 48000:
                    # 간단한 리샘플링 (실제로는 더 정교한 방법 사용)
                    ratio = 48000 / sample_rate
                    new_length = int(len(audio_array) * ratio)
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array), new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    ).astype(np.int16)
                
                # 프레임 단위로 분할
                frames = []
                for i in range(0, len(audio_array), self.FRAME_SIZE):
                    frame = audio_array[i:i + self.FRAME_SIZE]
                    if len(frame) == self.FRAME_SIZE:
                        frames.append(frame)
                    elif len(frame) > 0:
                        # 마지막 프레임을 패딩
                        padded_frame = np.zeros(self.FRAME_SIZE, dtype=np.int16)
                        padded_frame[:len(frame)] = frame
                        frames.append(padded_frame)
                
                aprint(f"WAV 파일 로드 완료: {file_path}, {len(frames)} 프레임")
                return frames
                
        except Exception as e:
            aprint(f"WAV 파일 로드 오류: {file_path}, {e}")
            return []

    def find_wav_files(self, room_name: str = "test") -> List[str]:
        """recordings 폴더에서 WAV 파일 찾기"""
        try:
            # recordings 폴더 내의 모든 WAV 파일 검색
            pattern = f"recordings/**/{room_name}/**/*.wav"
            wav_files = glob.glob(pattern, recursive=True)
            
            # input 파일들만 선택 (더 깔끔한 음성)
            input_files = [f for f in wav_files if "input_" in f]
            
            if not input_files:
                # input 파일이 없으면 모든 WAV 파일 사용
                input_files = wav_files
            
            aprint(f"발견된 WAV 파일: {len(input_files)}개")
            return input_files
            
        except Exception as e:
            aprint(f"WAV 파일 검색 오류: {e}")
            return []

    # ========== 유틸 ==========
    def is_webbrowser(self, websocket: WebSocket) -> bool:
        ua = websocket.headers.get("user-agent", "").lower()
        return not ("python" in ua)

    @staticmethod
    def _now(loop: asyncio.AbstractEventLoop) -> float:
        return loop.time()

    # ========== 방/참가자 관리 ==========
    def get_rooms(self):
        return [{"room_name": rn, "num_person": len(ws_list)} for rn, ws_list in self.rooms.items()]

    async def create_room(self, room_name: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = []
            self.audio_buffers[room_name] = {}
            task = asyncio.create_task(self.process_a_room_audio_runner(room_name))
            self.room_tasks[room_name] = task
        return {"message": f"Room '{room_name}' created successfully"}

    async def delete_room(self, id: int):
        # DB에서 룸 찾고 → 연결 끊고 → 태스크 취소 → DB 삭제
        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.id == id)).first()
            if not room:
                return None
            room_name = room.room_name

            if room_name in self.rooms:
                for ws in list(self.rooms[room_name]):
                    try:
                        await ws.close(code=4004)
                    except Exception:
                        pass
                self.rooms.pop(room_name, None)
                self.audio_buffers.pop(room_name, None)

            task = self.room_tasks.pop(room_name, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            room2 = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            if room2:
                session.delete(room2)
                session.commit()

        return {"message": f"Room '{room_name}' deleted successfully"}

    def _add_person_to_room(self, room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket) -> int:
        client_id = id(websocket)
        self.rooms[room_name].append(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=self.QUEUE_MAXLEN_FRAMES)

        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            device = session.exec(select(Devices).filter(Devices.device_id == device_id)).first()
            person_name = '웹접속' if device is None else device.owner
            if room.num_person == 0:
                room.persons = f"{person_name}"
            else:
                room.persons = f"{room.persons}, {person_name}"
            room.num_person += 1
            session.add(room)
            session.commit()

        self.client_info[client_id] = {
            "sr": sr,
            "dtype": dtype,
            "is_webbrowser": self.is_webbrowser(websocket),
            "person_name": person_name
        }
        return client_id

    def _remove_person_from_room(self, room_name: str, websocket: WebSocket) -> int:
        client_id = id(websocket)
        person_name = self.client_info.get(client_id, {}).get("person_name", "알수없음")

        # 재생 태스크 정리
        if client_id in self.playback_tasks:
            self.playback_tasks[client_id].cancel()
            self.playback_tasks.pop(client_id, None)
        self.playback_data.pop(client_id, None)

        if self.rooms.get(room_name):
            try:
                self.rooms[room_name].remove(websocket)
            except ValueError:
                pass

            self.audio_buffers[room_name].pop(client_id, None)
            self.client_info.pop(client_id, None)

            with Session(engine) as session:
                room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
                if room:
                    room.num_person = max(0, room.num_person - 1)
                    if room.num_person == 0:
                        room.persons = ''
                    else:
                        persons = (room.persons or '')
                        persons = persons.replace(f", {person_name}", '')
                        persons = persons.replace(f"{person_name}, ", '')
                        if persons.endswith(person_name):
                            persons = persons[: -len(person_name)]
                        room.persons = persons.strip().strip(",").strip()
                    session.add(room)
                    session.commit()
        return client_id

    async def join_room(self, room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
        if room_name not in self.rooms:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        client_id = self._add_person_to_room(room_name, sr, dtype, device_id, websocket)
        aprint(f"Client {client_id} joined room '{room_name}'")
        
        # WAV 파일 재생 시작
        await self.start_playback(client_id, room_name)
        
        # 연결 유지 (실제로는 마이크 입력을 받지 않음)
        await self._keep_connection(room_name, client_id, websocket)

    async def exit_room(self, room_name: str, websocket: WebSocket):
        client_id = self._remove_person_from_room(room_name, websocket)
        aprint(f"Client {client_id} exited room '{room_name}'")

    # ========== 재생 로직 ==========
    async def start_playback(self, client_id: int, room_name: str):
        """WAV 파일 재생 시작"""
        try:
            # WAV 파일 찾기
            wav_files = self.find_wav_files(room_name)
            if not wav_files:
                aprint(f"방 '{room_name}'에 대한 WAV 파일을 찾을 수 없습니다.")
                return
            
            # 첫 번째 파일 로드
            wav_file = wav_files[0]
            frames = self.load_wav_file(wav_file)
            if not frames:
                aprint(f"WAV 파일 로드 실패: {wav_file}")
                return
            
            self.playback_data[client_id] = frames
            
            # 재생 태스크 시작
            task = asyncio.create_task(self._playback_runner(client_id, room_name))
            self.playback_tasks[client_id] = task
            
            aprint(f"Client {client_id} 재생 시작: {wav_file}")
            
        except Exception as e:
            aprint(f"재생 시작 오류: {e}")

    async def _playback_runner(self, client_id: int, room_name: str):
        """WAV 파일 재생 루프"""
        loop = asyncio.get_event_loop()
        frame_index = 0
        
        while client_id in self.playback_data and room_name in self.rooms:
            try:
                frames = self.playback_data[client_id]
                
                if frame_index >= len(frames):
                    # 재생 완료, 다시 처음부터
                    frame_index = 0
                    aprint(f"Client {client_id} 재생 루프")
                
                # 현재 프레임을 버퍼에 추가
                if room_name in self.audio_buffers and client_id in self.audio_buffers[room_name]:
                    q = self.audio_buffers[room_name][client_id]
                    ts = self._now(loop)
                    q.append((ts, frames[frame_index]))
                
                frame_index += 1
                
                # 20ms 대기 (실제 오디오 프레임 간격)
                await asyncio.sleep(0.02)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                aprint(f"재생 오류: {e}")
                break
        
        aprint(f"Client {client_id} 재생 종료")

    async def _keep_connection(self, room_name: str, client_id: int, websocket: WebSocket):
        """연결 유지 (실제 마이크 입력은 받지 않음)"""
        while True:
            try:
                # 연결 상태 확인만
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # 타임아웃은 정상 (연결 유지)
                continue
            except Exception as e:
                aprint(f"[{room_name}] connection error: client={client_id}, {e}")
                await self.exit_room(room_name, websocket)
                return

    # ========== 전송 ==========
    async def _safe_send_audio(self, ws: WebSocket, pcm_int16: np.ndarray, dtype: str, sr: int):
        try:
            if dtype == "float32":
                # int16을 float32로 변환
                pcm_float32 = (pcm_int16.astype(np.float32) / 32767.0)
                await ws.send_bytes(pcm_float32.tobytes())
            else:
                await ws.send_bytes(pcm_int16.tobytes())
        except (WebSocketDisconnect, RuntimeError) as e:
            # 연결 종료로 판단하고 정리
            room_name = None
            for rn, lst in self.rooms.items():
                if ws in lst:
                    room_name = rn
                    break
            aprint(f"send_audio disconnect: {e}")
            if room_name:
                await self.exit_room(room_name, ws)
        except Exception as e:
            aprint(f"send_audio error: {e}")

    # ========== 방 오디오 처리 루프 ==========
    async def process_a_room_audio_runner(self, room_name: str):
        loop = asyncio.get_event_loop()
        last_process_time = loop.time()

        while room_name in self.rooms:
            current_time = loop.time()
            if (current_time - last_process_time) >= self.SYNC_INTERVAL:
                if room_name not in self.rooms:
                    break
                room_buffer = self.audio_buffers.get(room_name)
                if not room_buffer:
                    await asyncio.sleep(0.01)
                    continue

                # 활성 클라이언트 파악
                active_clients: List[Tuple[int, int]] = []  # [(ws_idx, client_id), ...]

                for ws_idx, ws in enumerate(list(self.rooms.get(room_name, []))):
                    cid = id(ws)
                    q: deque = room_buffer.get(cid)
                    if not q or len(q) == 0:
                        continue
                    active_clients.append((ws_idx, cid))

                if len(active_clients) == 0:
                    await asyncio.sleep(0.01)
                    continue

                # 각 클라이언트에서 한 프레임씩 가져와서 믹스
                combined_audio_int16: List[np.ndarray] = []
                for ws_idx, cid in active_clients:
                    q: deque = room_buffer[cid]
                    if q:
                        frame = q.popleft()[1]  # 타임스탬프 제거하고 프레임만
                        combined_audio_int16.append(frame)
                    else:
                        # 빈 프레임 추가
                        combined_audio_int16.append(np.zeros(self.FRAME_SIZE, dtype=np.int16))

                # 믹스된 오디오를 각 클라이언트로 전송
                for audio_idx, (ws_idx, cid) in enumerate(active_clients):
                    ws_list = self.rooms.get(room_name, [])
                    if ws_idx >= len(ws_list):
                        continue
                    ws = ws_list[ws_idx]
                    if ws is None:
                        continue

                    # 자신의 음성 제외 (hear_me = False)
                    exclude_idx = audio_idx
                    mixed_audio = self.mix_audio_simple(combined_audio_int16, exclude_idx)
                    
                    info = self.client_info.get(cid, {})
                    dtype = info.get("dtype", "int16")
                    sr = info.get("sr", 16000)
                    
                    asyncio.create_task(self._safe_send_audio(ws, mixed_audio, dtype, sr))

                last_process_time = current_time

            await asyncio.sleep(0.01)

        aprint(f"[{room_name}] audio runner stopped")

    def mix_audio_simple(self, audio_frames: List[np.ndarray], exclude_idx: int) -> np.ndarray:
        """간단한 오디오 믹싱 - 자신의 음성 제외"""
        if not audio_frames:
            return np.zeros(self.FRAME_SIZE, dtype=np.int16)
        
        # exclude_idx 제외하고 믹스
        frames_to_mix = []
        for i, frame in enumerate(audio_frames):
            if i != exclude_idx:
                frames_to_mix.append(frame)
        
        if not frames_to_mix:
            return np.zeros(self.FRAME_SIZE, dtype=np.int16)
        
        # 간단한 평균 믹싱
        mixed = np.mean(frames_to_mix, axis=0).astype(np.int16)
        return mixed
