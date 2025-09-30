import asyncio
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from database import engine
from models import Rooms, Devices
from utils.sys import aprint


class SimpleVoiceChat:
    """
    기본적인 음성통신만 지원하는 간소화된 버전
    - 복잡한 음성 향상 제거
    - STT 처리 제거
    - 이벤트 분류 제거
    - 녹음 기능 제거
    - 지연 보정 로직 단순화
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

    # ========== 유틸 ==========
    def is_webbrowser(self, websocket: WebSocket) -> bool:
        ua = websocket.headers.get("user-agent", "").lower()
        return not ("python" in ua)

    @staticmethod
    def _downsample_48k_to_16k(x: np.ndarray) -> np.ndarray:
        """48k → 16k 간단 저역통과(3 샘플 평균) 후 축소."""
        n = (len(x) // 3) * 3
        if n == 0:
            return np.array([], dtype=np.int16)
        m = x[:n].astype(np.int32).reshape(-1, 3).mean(axis=1)
        return m.astype(np.int16)

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
        await self._receive_voice_runner(room_name, client_id, websocket)

    async def exit_room(self, room_name: str, websocket: WebSocket):
        client_id = self._remove_person_from_room(room_name, websocket)
        aprint(f"Client {client_id} exited room '{room_name}'")

    # ========== 수신/전송 ==========
    async def _receive_voice_runner(self, room_name: str, client_id: int, client_ws: WebSocket):
        loop = asyncio.get_event_loop()
        while True:
            try:
                byte_data = await asyncio.wait_for(client_ws.receive_bytes(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                aprint(f"[{room_name}] receive error: client={client_id}, {e}")
                await self.exit_room(room_name, client_ws)
                return

            if not byte_data:
                await asyncio.sleep(0.01)
                continue

            # 디코딩
            try:
                info = self.client_info.get(client_id, {})
                dtype = info.get("dtype")
                sr = info.get("sr", 16000)

                if dtype == "float32":
                    data_f32 = np.frombuffer(byte_data, dtype=np.float32).reshape(-1)
                    # float32를 int16으로 변환
                    data_int16 = (data_f32 * 32767).astype(np.int16)
                else:
                    if sr == 16000:
                        data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                    elif sr == 48000:
                        raw_i16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                        data_int16 = self._downsample_48k_to_16k(raw_i16)
                    else:
                        data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)

                # 프레임 단위로 쪼개고 큐에 저장
                if len(data_int16) >= self.FRAME_SIZE:
                    q = self.audio_buffers[room_name][client_id]
                    for i in range(0, len(data_int16), self.FRAME_SIZE):
                        frame = data_int16[i:i + self.FRAME_SIZE]
                        if len(frame) == self.FRAME_SIZE:
                            ts = self._now(loop)
                            q.append((ts, frame))
            except Exception as e:
                aprint(f"[{room_name}] parse error: client={client_id}, {e}")

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
