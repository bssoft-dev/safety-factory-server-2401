import asyncio
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlmodel import Session, select

from database import engine
from models import Rooms, Devices
from services.audio_utils import AudioUtils
from services.stt import SttProcessor
from utils.sys import aprint


class VoiceChat(AudioUtils):
    """
    - 누적 지연 방지:
      1) 프레임에 타임스탬프 부여 + LATENCY_BUDGET 초과 프레임 드랍
      2) backlog 크면 틱당 더 많이 드레인(적응형)
      3) enqueue 시 큐가 찼다면 오래된 것 드랍(정책 A)
      4) 송신 경로는 클라이언트별 in-flight 1개만(세마포어)
      5) 간단 클록 드리프트 보정(주기적 드랍/복제)
    """

    # ====== 오디오/지연 관련 파라미터 ======
    LATENCY_BUDGET = 0.20          # sec, 200ms 이상 늦은 프레임은 버림
    MAX_FRAMES_PER_TICK = 5        # 한 틱에서 처리할 최대 프레임(기본)
    BACKLOG_BOOST_FACTOR = 2       # backlog 크면 MAX*2까지 처리
    TARGET_BACKLOG_FRAMES = 3      # 각 큐의 목표 백로그(프레임 수)
    DRIFT_TOLERANCE_FRAMES = 3     # 목표 대비 ±3 프레임 이상이면 드랍/복제
    QUEUE_MAXLEN_FRAMES = 200      # 클라이언트별 입력 큐 최대 프레임 수

    # 타임아웃 로그 억제
    TIMEOUT_LOG_INTERVAL = 5.0     # sec

    def __init__(self):
        super().__init__()
        self.rooms: Dict[str, List[WebSocket]] = {}                    # {room_name: [WebSocket, ...]}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {}           # {room_name: {client_id: deque[(ts, frame), ...]}}
        self.client_info: Dict[int, dict] = {}                         # {client_id: {sr, dtype, is_webbrowser, person_name}}
        self.room_tasks: Dict[str, asyncio.Task] = {}                  # 방 처리 태스크 핸들
        self.send_sem: Dict[int, asyncio.Semaphore] = {}               # 클라이언트별 송신 세마포어(1)
        self._last_timeout_log = 0.0

        # 방별 설정 캐시
        self.room_settings_cache: Dict[str, dict] = {}                 # {room_name: settings_dict}
        
        # 전역 기본값 (백업용)
        self.default_settings = {
            'use_voice_enhance': True,
            'hear_me': False,
            'record_audio': True,
            'classify_event': True,
            'do_stt': True,
            'enhance_volume': 0,
            'keep_test_room': True  # 이건 전역 설정으로 유지
        }

        # 기존 전역 플래그들 (호환성 유지)
        self.use_voice_enhance = True
        self.hear_me = False
        self.record_audio = True
        self.keep_test_room = True
        self.classify_event = True
        self.do_stt = True
        self.enhance_volume = 0

        self.stt_processor = SttProcessor()

        # 테스트용 플래그들
        self.test_mode = False
        self.test_send_raw = False  # 원본 데이터 그대로 전송
        self.test_send_silence = False  # 무음 전송
        self.test_send_sine = False  # 사인파 전송
        self.test_skip_mixing = False  # 믹싱 건너뛰기
        self.test_skip_enhance = False  # 음성 향상 건너뛰기
        self.test_disable_frame_drop = False  # 프레임 드랍 비활성화
        self.test_sync_interval = 0.02  # 테스트용 SYNC_INTERVAL (20ms)
        self.test_continuous_sine = False  # 연속 사인파 전송
        self.test_long_sine = False  # 긴 사인파 전송 (1초)
        self.test_raw_mixed = False  # raw + 믹싱
        self.test_raw_mixed_enhanced = False  # raw + 믹싱 + 음성향상
        self.test_direct_send = False  # 직접 전송 (combined_audio 사용 안함)
        self.test_bypass_all = False  # 모든 처리 우회
        self.test_ultra_simple = False  # 초간단 테스트 (무음만)
        self.test_no_sync = False  # SYNC_INTERVAL 무시
        self.test_frame_512 = False  # 512 샘플 프레임 테스트
        self.test_frame_1024 = False  # 1024 샘플 프레임 테스트
        self.test_frame_2048 = False  # 2048 샘플 프레임 테스트
        
        # 테스트용 카운터
        self.test_frame_count = 0
        self.test_drop_count = 0
        self.test_send_count = 0
        
        # 메모리 모니터링
        self.memory_monitor_enabled = False
        self.memory_log_interval = 30  # 30초마다 로그
        self.last_memory_log = 0

    # ========== 유틸 ==========
    def is_webbrowser(self, websocket: WebSocket) -> bool:
        ua = websocket.headers.get("user-agent", "").lower()
        return not ("python" in ua)

    @staticmethod
    def _safe_clip_int16(arr: np.ndarray, gain: int = 1) -> np.ndarray:
        """볼륨 증폭 시 int16 오버플로 보호."""
        if gain and gain != 1:
            x = arr.astype(np.int32) * gain
            return np.clip(x, -32768, 32767).astype(np.int16)
        return arr

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
        # event loop 시간(모노토닉)을 사용 (time.time보다 지연 계산에 안전)
        return loop.time()

    # ========== 방별 설정 관리 ==========
    def get_room_settings(self, room_name: str) -> dict:
        """방별 설정 가져오기 (캐시 우선, DB 조회 후 캐시)"""
        if room_name in self.room_settings_cache:
            return self.room_settings_cache[room_name]
        
        # DB에서 조회
        try:
            from database import engine
            from sqlmodel import Session, select
            from models import RoomSettings
            
            with Session(engine) as session:
                settings = session.exec(
                    select(RoomSettings).where(RoomSettings.room_name == room_name)
                ).first()
                
                if settings:
                    settings_dict = {
                        'use_voice_enhance': settings.use_voice_enhance,
                        'hear_me': settings.hear_me,
                        'record_audio': settings.record_audio,
                        'classify_event': settings.classify_event,
                        'do_stt': settings.do_stt,
                        'enhance_volume': settings.enhance_volume
                    }
                    self.room_settings_cache[room_name] = settings_dict
                    return settings_dict
                else:
                    # 설정이 없으면 기본값으로 생성
                    return self.create_default_room_settings(room_name)
                    
        except Exception as e:
            aprint(f"방 설정 조회 오류 ({room_name}): {e}")
            return self.default_settings.copy()

    def create_default_room_settings(self, room_name: str) -> dict:
        """방의 기본 설정 생성"""
        try:
            from database import engine
            from sqlmodel import Session
            from models import RoomSettings
            
            settings_dict = self.default_settings.copy()
            settings_dict.pop('keep_test_room', None)  # 전역 설정 제거
            
            with Session(engine) as session:
                new_settings = RoomSettings(
                    room_name=room_name,
                    **settings_dict
                )
                session.add(new_settings)
                session.commit()
                
                self.room_settings_cache[room_name] = settings_dict
                aprint(f"방 '{room_name}' 기본 설정 생성 완료")
                return settings_dict
                
        except Exception as e:
            aprint(f"방 설정 생성 오류 ({room_name}): {e}")
            return self.default_settings.copy()

    def update_room_settings(self, room_name: str, **kwargs) -> bool:
        """방 설정 업데이트"""
        try:
            from database import engine
            from sqlmodel import Session, select
            from models import RoomSettings
            from datetime import datetime
            
            with Session(engine) as session:
                settings = session.exec(
                    select(RoomSettings).where(RoomSettings.room_name == room_name)
                ).first()
                
                if not settings:
                    # 설정이 없으면 생성
                    self.create_default_room_settings(room_name)
                    settings = session.exec(
                        select(RoomSettings).where(RoomSettings.room_name == room_name)
                    ).first()
                
                # 설정 업데이트
                for key, value in kwargs.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)
                
                settings.updated_at = datetime.now()
                session.commit()
                
                # 캐시 업데이트
                if room_name in self.room_settings_cache:
                    self.room_settings_cache[room_name].update(kwargs)
                else:
                    self.room_settings_cache[room_name] = self.get_room_settings(room_name)
                
                aprint(f"방 '{room_name}' 설정 업데이트: {kwargs}")
                return True
                
        except Exception as e:
            aprint(f"방 설정 업데이트 오류 ({room_name}): {e}")
            return False

    def get_room_setting(self, room_name: str, setting_name: str, default_value=None):
        """특정 방의 특정 설정값 가져오기"""
        settings = self.get_room_settings(room_name)
        return settings.get(setting_name, default_value)

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
        self.audio_buffers[room_name][client_id] = deque(maxlen=self.QUEUE_MAXLEN_FRAMES)  # FIX: maxlen
        self.input_rec_buffer[client_id] = []
        self.output_rec_buffer[client_id] = []
        self.send_sem[client_id] = asyncio.Semaphore(1)  # FIX: 송신 세마포어
        self.voice_enhancer.add_streamer(client_id)

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
        self.voice_enhancer.remove_streamer(client_id)

        if self.rooms.get(room_name):
            try:
                self.rooms[room_name].remove(websocket)
            except ValueError:
                pass

            self.audio_buffers[room_name].pop(client_id, None)
            self.client_info.pop(client_id, None)
            self.send_sem.pop(client_id, None)
            self.input_rec_buffer.pop(client_id, None)
            self.output_rec_buffer.pop(client_id, None)

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
        client_id = id(websocket)

        # 종료 전 녹음 저장(예외 안전)
        try:
            if self.record_audio and client_id in self.input_rec_buffer and len(self.input_rec_buffer[client_id]) > 0:
                self.save_audio(
                    self.input_rec_buffer[client_id],
                    self.client_info.get(client_id, {}).get("person_name", "unknown"),
                    room_name,
                    "input_exit"
                )
            if self.record_audio and client_id in self.output_rec_buffer and len(self.output_rec_buffer[client_id]) > 0:
                self.save_audio(
                    self.output_rec_buffer[client_id],
                    self.client_info.get(client_id, {}).get("person_name", "unknown"),
                    room_name,
                    "output_exit"
                )
        except Exception as e:
            aprint(f"Error in save_audio on exit: {e}")

        # STT 정리
        try:
            self.stt_processor.cleanup_client(client_id)
        except Exception as e:
            aprint(f"STT cleanup error: {e}")

        client_id = self._remove_person_from_room(room_name, websocket)
        aprint(f"Client {client_id} exited room '{room_name}'")

    # ========== 수신/전송 ==========
    def _push_frame(self, q: deque, item: Tuple[float, np.ndarray]):
        """FIX: enqueue 시 드랍 정책 - 꽉 차면 가장 오래된 것 버리고 최신 유지."""
        if len(q) >= q.maxlen:
            q.popleft()
        q.append(item)

    async def _receive_voice_runner(self, room_name: str, client_id: int, client_ws: WebSocket):
        loop = asyncio.get_event_loop()
        while True:
            try:
                byte_data = await asyncio.wait_for(client_ws.receive_bytes(), timeout=0.5)
            except asyncio.TimeoutError:
                now = time.time()
                if (now - self._last_timeout_log) > self.TIMEOUT_LOG_INTERVAL:
                    aprint(f"[{room_name}] audio timeout: client={client_id}")
                    self._last_timeout_log = now
                continue
            except Exception as e:
                aprint(f"[{room_name}] receive error: client={client_id}, {e}")
                await self.exit_room(room_name, client_ws)

                if (room_name in self.rooms) and (not self.rooms[room_name]) and (room_name == '보온팀') and (not self.keep_test_room):
                    self.rooms.pop(room_name, None)
                    self.audio_buffers.pop(room_name, None)
                    task = self.room_tasks.pop(room_name, None)
                    if task:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
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
                    data_int16 = self.float32_to_int16(data_f32)
                else:
                    if sr == 16000:
                        data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                        if self.enhance_volume > 0:
                            data_int16 = self._safe_clip_int16(data_int16, self.enhance_volume)
                    elif sr == 48000:
                        raw_i16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                        data_int16 = self._downsample_48k_to_16k(raw_i16)
                    else:
                        data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)

                # 프레임 단위로 쪼개고 (ts, frame)으로 큐에 저장
                if len(data_int16) >= self.FRAME_SIZE:
                    q = self.audio_buffers[room_name][client_id]
                    for i in range(0, len(data_int16), self.FRAME_SIZE):
                        frame = data_int16[i:i + self.FRAME_SIZE]
                        if len(frame) == self.FRAME_SIZE:
                            ts = self._now(loop)
                            self._push_frame(q, (ts, frame))
            except Exception as e:
                aprint(f"[{room_name}] parse error: client={client_id}, {e}")

    async def _safe_send_audio(self, ws: WebSocket, pcm_int16: np.ndarray, dtype: str, sr: int):
        try:
            await self.send_audio(ws, pcm_int16, dtype, sr)
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

    async def _safe_send_audio_bounded(self, ws: WebSocket, pcm_int16: np.ndarray, dtype: str, sr: int):
        """FIX: 클라이언트별 in-flight 1개 보장. 이미 전송 중이면 최신 프레임만 유지하고 스킵."""
        cid = id(ws)
        sem = self.send_sem.get(cid)
        if sem is None:
            return
        if not sem.locked():
            async with sem:
                await self._safe_send_audio(ws, pcm_int16, dtype, sr)
        else:
            # 이전 프레임 전송 중 → 이번 것은 드랍(최신성이 중요)
            pass

    # ========== 방 오디오 처리 루프 ==========
    async def process_a_room_audio_runner(self, room_name: str):
        loop = asyncio.get_event_loop()
        last_process_time = loop.time()

        while room_name in self.rooms:
            current_time = loop.time()
            
            # 메모리 모니터링 (30초마다)
            if self.memory_monitor_enabled and (current_time - self.last_memory_log) >= self.memory_log_interval:
                self.log_memory_usage()
                self.last_memory_log = current_time
            
            # 테스트 모드에서는 다른 SYNC_INTERVAL 사용 (테스트 방에서만)
            sync_interval = self.test_sync_interval if (room_name == '보온팀' and (self.test_send_silence or self.test_send_sine or self.test_send_raw or self.test_skip_enhance or self.test_skip_mixing or self.test_raw_mixed or self.test_raw_mixed_enhanced or self.test_direct_send or self.test_bypass_all or self.test_ultra_simple or self.test_no_sync or self.test_frame_512 or self.test_frame_1024 or self.test_frame_2048)) else self.SYNC_INTERVAL
            if (current_time - last_process_time) >= sync_interval:
                if room_name not in self.rooms:
                    break
                room_buffer = self.audio_buffers.get(room_name)
                if not room_buffer:
                    await asyncio.sleep(0.01)
                    continue

                # 1) 활성 클라이언트/최소 큐 길이 파악 + 오래된 프레임 drop
                active_clients: List[Tuple[int, int]] = []  # [(ws_idx, client_id), ...]
                min_len = float("inf")

                for ws_idx, ws in enumerate(list(self.rooms.get(room_name, []))):
                    cid = id(ws)
                    q: deque = room_buffer.get(cid)
                    if not q:
                        continue

                    # FIX: 너무 오래된 프레임 버리기 (fast-forward)
                    if not (room_name == '테스트' and self.test_disable_frame_drop):
                        now = loop.time()
                        while q and (now - q[0][0]) > self.LATENCY_BUDGET:
                            q.popleft()

                    if len(q) > 0:
                        active_clients.append((ws_idx, cid))
                        min_len = min(min_len, len(q))

                if min_len == float("inf") or len(active_clients) == 0:
                    await asyncio.sleep(0.01)
                    continue

                # 2) 적응형 드레인: backlog 크면 더 많이 처리
                backlog = min_len
                if backlog >= (self.MAX_FRAMES_PER_TICK * 2):
                    take_len = min(backlog, self.MAX_FRAMES_PER_TICK * self.BACKLOG_BOOST_FACTOR)
                else:
                    take_len = min(backlog, self.MAX_FRAMES_PER_TICK)

                # 3) per-client 프레임 모으기
                combined_audio_int16: List[np.ndarray] = []
                per_client_frames: List[np.ndarray] = []
                for ws_idx, cid in active_clients:
                    q: deque = room_buffer[cid]
                    frames = [q.popleft()[1] for _ in range(min(take_len, len(q)))]
                    if not frames:
                        # 드레인할 게 없으면 건너뜀
                        per_client_frames.append(np.array([], dtype=self.FORMAT))
                        combined_audio_int16.append(np.array([], dtype=self.FORMAT))
                        continue
                    one = np.array(frames, dtype=self.FORMAT).reshape(-1)
                    per_client_frames.append(one)
                    combined_audio_int16.append(one)

                # (옵션) 이벤트 분류
                if self.classify_event:
                    asyncio.create_task(self.classify_audio(combined_audio_int16, room_name))

                # (옵션) STT: 각 클라이언트별로
                if self.do_stt:
                    for audio_idx, (ws_idx, cid) in enumerate(active_clients):
                        arr = per_client_frames[audio_idx]
                        if arr.size > 0:
                            asyncio.create_task(self.stt_audio([arr], room_name, cid))

                # 4) 간단 드리프트 보정: 목표 백로그와 비교하여 드랍/복제
                for _, cid in active_clients:
                    q: deque = room_buffer.get(cid)
                    if not q:
                        continue
                    qlen = len(q)
                    if (qlen - self.TARGET_BACKLOG_FRAMES) >= self.DRIFT_TOLERANCE_FRAMES:
                        # 과다 → 한 프레임 더 버려서 당기기
                        if q:
                            q.popleft()
                    elif (self.TARGET_BACKLOG_FRAMES - qlen) >= self.DRIFT_TOLERANCE_FRAMES:
                        # 부족 → 마지막 프레임 복제하여 매우 소량 보강(가청 영향 거의 없음)
                        if q:
                            last_ts, last_fr = q[-1]
                            q.append((last_ts, last_fr.copy()))

                # 5) 믹스 후 각 클라이언트로 송신 (세마포어로 폭주 방지)
                for audio_idx, (ws_idx, cid) in enumerate(active_clients):
                    ws_list = self.rooms.get(room_name, [])
                    if ws_idx >= len(ws_list):
                        continue
                    ws = ws_list[ws_idx]
                    if ws is None:
                        continue

                    exclude_idx = None if self.hear_me else audio_idx
                    asyncio.create_task(
                        self._talk_to_each_client(combined_audio_int16, exclude_idx, ws, room_name)
                    )

                last_process_time = current_time

            await asyncio.sleep(0.01)

        aprint(f"[{room_name}] audio runner stopped")

    # ========== 송신 로직 ==========
    async def _talk_to_each_client(self, combined_audio: List[np.ndarray], exclude_client_idx: int, ws: WebSocket, room_name: str):
        cid = id(ws)
        info = self.client_info.get(cid)
        if not info:
            return
        dtype = info["dtype"]
        sr = info["sr"]
        person_name = info.get("person_name", "unknown")

        try:
            # 테스트 모드인 경우 테스트 로직 실행 (테스트 방에서만)
            if (room_name == '보온팀' and 
                (self.test_send_silence or self.test_send_sine or 
                 self.test_send_raw or self.test_skip_enhance or self.test_skip_mixing or 
                 self.test_continuous_sine or self.test_long_sine or 
                 self.test_raw_mixed or self.test_raw_mixed_enhanced or 
                 self.test_direct_send or self.test_bypass_all or 
                 self.test_ultra_simple or self.test_no_sync or 
                 self.test_frame_512 or self.test_frame_1024 or self.test_frame_2048)):
                await self._talk_to_each_client_test(combined_audio, exclude_client_idx, ws, room_name)
                return
            if self.use_voice_enhance:
                try:
                    f32_tensor = self.mix_audio_to_torch(combined_audio, exclude_client_idx)
                    enhanced_i16 = self.voice_enhance(f32_tensor, cid)
                except Exception as e:
                    aprint(f"voice enhance error: {e}")
                    # 폴백: f32가 있으면 변환, 없으면 일반 믹스
                    if 'f32_tensor' in locals():
                        pcm_i16 = self.torch_float32_to_int16(f32_tensor)
                    else:
                        pcm_i16 = self.mix_audio(combined_audio, exclude_client_idx)
                    await self._safe_send_audio_bounded(ws, pcm_i16, dtype, sr)
                    if self.record_audio:
                        try:
                            self.output_rec_buffer[cid] = await self.recording_audio(
                                self.output_rec_buffer.get(cid, []), pcm_i16, room_name, person_name, "output"
                            )
                        except Exception as re:
                            aprint(f"record fallback error: {re}")
                    return

                await self._safe_send_audio_bounded(ws, enhanced_i16, dtype, sr)

                if self.record_audio:
                    try:
                        pcm_i16_in = self.torch_float32_to_int16(f32_tensor)
                        self.input_rec_buffer[cid] = await self.recording_audio(
                            self.input_rec_buffer.get(cid, []), pcm_i16_in, room_name, person_name, "input"
                        )
                        self.output_rec_buffer[cid] = await self.recording_audio(
                            self.output_rec_buffer.get(cid, []), enhanced_i16, room_name, person_name, "output"
                        )
                    except Exception as e:
                        aprint(f"record error: {e}")

            else:
                pcm_i16 = self.mix_audio(combined_audio, exclude_client_idx)
                await self._safe_send_audio_bounded(ws, pcm_i16, dtype, sr)
                if self.record_audio:
                    self.output_rec_buffer[cid] = await self.recording_audio(
                        self.output_rec_buffer.get(cid, []), pcm_i16, room_name, person_name, "output"
                    )

        except Exception as e:
            aprint(f"[{room_name}] talk error: {e}")

    # ========== STT ==========
    async def stt_audio(self, combined_audio_int16, room_name: str, client_id: int):
        try:
            if not combined_audio_int16:
                return
            arr = (np.concatenate(combined_audio_int16)
                   if len(combined_audio_int16) > 1 else combined_audio_int16[0])
            if arr.size == 0:
                return
            worker = self.client_info.get(client_id, {}).get("person_name", "unknown")
            audio_bytes = arr.astype(np.int16).tobytes()
            await self.stt_processor.send_audio(audio_bytes, room_name, worker, client_id)
        except Exception as e:
            aprint(f"STT 처리 오류: {e}")

    # ========== 테스트용 송신 로직 ==========
    async def _talk_to_each_client_test(self, combined_audio: List[np.ndarray], exclude_client_idx: int, ws: WebSocket, room_name: str):
        """테스트용 송신 로직 - 문제 원인 파악"""
        cid = id(ws)
        info = self.client_info.get(cid)
        if not info:
            return
            
        dtype = info["dtype"]
        sr = info["sr"]
        person_name = info.get("person_name", "unknown")
        
        self.test_frame_count += 1
        
        try:
            # 테스트 1: 무음 전송
            if self.test_send_silence:
                silence = np.zeros(self.FRAME_SIZE, dtype=np.int16)
                await self._safe_send_audio(ws, silence, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 무음 전송: frame={self.test_frame_count}")
                return
            
            # 테스트 2: 사인파 전송 (더 긴 프레임 사용)
            if self.test_send_sine:
                # 더 긴 프레임 사용 (100ms = 1600 샘플)
                long_frame_size = 1600  # 100ms @ 16kHz
                t = np.linspace(0, 0.1, long_frame_size)  # 100ms
                frequency = 440  # A4 음
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 사인파 전송 (100ms): frame={self.test_frame_count}")
                return
            
            # 테스트 2-1: 연속 사인파 전송 (단일 프레임, 지속적)
            if self.test_continuous_sine:
                # 현재 시간 기반으로 연속적인 사인파 생성
                import time
                current_time = time.time()
                t = np.linspace(0, 0.02, self.FRAME_SIZE)  # 20ms 프레임
                frequency = 440  # A4 음
                # 시간 오프셋을 추가하여 연속성 보장
                time_offset = (current_time * 1000) % 1000  # 밀리초 단위
                sine_wave = (np.sin(2 * np.pi * frequency * (t + time_offset/1000)) * 16000).astype(np.int16)
                
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 연속 사인파 전송: frame={self.test_frame_count}")
                return
            
            # 테스트 2-2: 긴 사인파 전송 (1초 분량)
            if self.test_long_sine:
                # 1초 분량의 사인파 생성
                long_frame_size = self.RATE  # 1초 = 16000 샘플
                t = np.linspace(0, 1.0, long_frame_size)  # 1초
                frequency = 440  # A4 음
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 긴 사인파 전송 (1초): frame={self.test_frame_count}")
                return
            
            # 테스트 3: 원본 데이터 그대로 전송 (믹싱 없이)
            if self.test_send_raw:
                if combined_audio and len(combined_audio) > 0:
                    # 첫 번째 클라이언트의 원본 데이터만 전송
                    raw_audio = combined_audio[exclude_client_idx] if len(combined_audio) > 0 else np.zeros(self.FRAME_SIZE, dtype=np.int16)
                    await self._safe_send_audio(ws, raw_audio, dtype, sr)
                    self.test_send_count += 1
                    aprint(f"[TEST] 원본 전송: frame={self.test_frame_count}, size={len(raw_audio)}")
                return
            
            # 테스트 4: 믹싱만 하고 음성 향상 건너뛰기
            if self.test_skip_enhance:
                # 디버깅 정보 추가
                if combined_audio:
                    aprint(f"[TEST] 믹싱 전 - combined_audio 길이: {len(combined_audio)}, exclude_idx: {exclude_client_idx}")
                    for i, audio in enumerate(combined_audio):
                        if audio is not None:
                            aprint(f"[TEST]   audio[{i}]: 길이={len(audio)}, dtype={audio.dtype}")
                        else:
                            aprint(f"[TEST]   audio[{i}]: None")
                
                pcm_i16 = self.mix_audio(combined_audio, exclude_client_idx)
                
                # 결과 검증
                if pcm_i16 is not None and len(pcm_i16) > 0:
                    await self._safe_send_audio(ws, pcm_i16, dtype, sr)
                    self.test_send_count += 1
                    aprint(f"[TEST] 믹싱만 전송: frame={self.test_frame_count}, 결과 길이={len(pcm_i16)}")
                else:
                    aprint(f"[TEST] 믹싱 결과가 비어있음: frame={self.test_frame_count}")
                return
            
            # 테스트 5: 모든 처리 건너뛰고 단순 전송
            if self.test_skip_mixing:
                # 단순히 첫 번째 클라이언트 데이터만 전송
                if combined_audio and len(combined_audio) > 0:
                    simple_audio = combined_audio[0]
                    await self._safe_send_audio(ws, simple_audio, dtype, sr)
                    self.test_send_count += 1
                    aprint(f"[TEST] 단순 전송: frame={self.test_frame_count}")
                return
            
            # 테스트 6: raw + 믹싱 (자신 제외)
            if self.test_raw_mixed:
                if combined_audio and len(combined_audio) > 0:
                    # 믹싱하되 자신의 음성은 제외
                    mixed_audio = self.mix_audio_to_torch(combined_audio, exclude_client_idx)
                    await self._safe_send_audio(ws, mixed_audio, dtype, sr)
                    self.test_send_count += 1
                    aprint(f"[TEST] raw+믹싱 전송: frame={self.test_frame_count}, exclude_idx={exclude_client_idx}")
                return
            
            # 테스트 7: raw + 믹싱 + 음성향상
            if self.test_raw_mixed_enhanced:
                if combined_audio and len(combined_audio) > 0:
                    try:
                        # 믹싱 + 음성향상
                        f32_tensor = self.mix_audio_to_torch(combined_audio, exclude_client_idx)
                        enhanced_i16 = self.voice_enhance(f32_tensor, cid)
                        await self._safe_send_audio(ws, enhanced_i16, dtype, sr)
                        self.test_send_count += 1
                        aprint(f"[TEST] raw+믹싱+향상 전송: frame={self.test_frame_count}")
                    except Exception as e:
                        aprint(f"[TEST] raw+믹싱+향상 오류: {e}")
                        # 폴백: 믹싱만
                        mixed_audio = self.mix_audio(combined_audio, exclude_client_idx)
                        await self._safe_send_audio(ws, mixed_audio, dtype, sr)
                        self.test_send_count += 1
                        aprint(f"[TEST] raw+믹싱 폴백: frame={self.test_frame_count}")
                return
            
            # 테스트 8: 직접 전송 (combined_audio 사용 안함)
            if self.test_direct_send:
                # combined_audio를 전혀 사용하지 않고 직접 사인파 생성
                t = np.linspace(0, 0.02, self.FRAME_SIZE)  # 20ms
                frequency = 440  # A4 음
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 직접 전송: frame={self.test_frame_count}")
                return
            
            # 테스트 9: 모든 처리 우회 (가장 기본적인 전송)
            if self.test_bypass_all:
                # 아무 처리 없이 무음 전송
                silence = np.zeros(self.FRAME_SIZE, dtype=np.int16)
                await self._safe_send_audio(ws, silence, dtype, sr)
                self.test_send_count += 1
                aprint(f"[TEST] 모든 처리 우회: frame={self.test_frame_count}")
                return
            
            # 테스트 10: 초간단 테스트 (무음만, 로그 최소화)
            if self.test_ultra_simple:
                silence = np.zeros(self.FRAME_SIZE, dtype=np.int16)
                await self._safe_send_audio(ws, silence, dtype, sr)
                self.test_send_count += 1
                # 로그 최소화 (100프레임마다만)
                if self.test_frame_count % 100 == 0:
                    aprint(f"[TEST] 초간단: frame={self.test_frame_count}")
                return
            
            # 테스트 11: SYNC_INTERVAL 무시 (연속 전송)
            if self.test_no_sync:
                # SYNC_INTERVAL을 무시하고 연속으로 전송
                t = np.linspace(0, 0.02, self.FRAME_SIZE)
                frequency = 440
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                if self.test_frame_count % 100 == 0:
                    aprint(f"[TEST] SYNC 무시: frame={self.test_frame_count}")
                return
            
            # 테스트 12: 512 샘플 프레임 테스트
            if self.test_frame_512:
                frame_size = 512
                t = np.linspace(0, frame_size / self.RATE, frame_size)
                frequency = 440
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                if self.test_frame_count % 50 == 0:
                    aprint(f"[TEST] 512 프레임: frame={self.test_frame_count}")
                return
            
            # 테스트 13: 1024 샘플 프레임 테스트
            if self.test_frame_1024:
                frame_size = 1024
                t = np.linspace(0, frame_size / self.RATE, frame_size)
                frequency = 440
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                if self.test_frame_count % 50 == 0:
                    aprint(f"[TEST] 1024 프레임: frame={self.test_frame_count}")
                return
            
            # 테스트 14: 2048 샘플 프레임 테스트
            if self.test_frame_2048:
                frame_size = 2048
                t = np.linspace(0, frame_size / self.RATE, frame_size)
                frequency = 440
                sine_wave = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)
                await self._safe_send_audio(ws, sine_wave, dtype, sr)
                self.test_send_count += 1
                if self.test_frame_count % 50 == 0:
                    aprint(f"[TEST] 2048 프레임: frame={self.test_frame_count}")
                return
                
        except Exception as e:
            aprint(f"[TEST] 송신 오류: {e}")
            self.test_drop_count += 1

    async def _safe_send_audio_bounded_test(self, ws: WebSocket, pcm_int16: np.ndarray, dtype: str, sr: int):
        """테스트용 송신 - 세마포어 없이 직접 전송"""
        try:
            # 세마포어 없이 직접 전송
            await self._safe_send_audio(ws, pcm_int16, dtype, sr)
            self.test_send_count += 1
        except Exception as e:
            aprint(f"[TEST] 직접 송신 오류: {e}")
            self.test_drop_count += 1

    def get_test_stats(self):
        """테스트 통계 반환"""
        return {
            "frame_count": self.test_frame_count,
            "send_count": self.test_send_count,
            "drop_count": self.test_drop_count,
            "drop_rate": self.test_drop_count / max(1, self.test_frame_count) * 100
        }

    def reset_test_stats(self):
        """테스트 통계 초기화"""
        self.test_frame_count = 0
        self.test_send_count = 0
        self.test_drop_count = 0
    
    def log_memory_usage(self):
        """메모리 사용량 로그"""
        try:
            import psutil
            import gc
            
            # 프로세스 메모리 사용량
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 오디오 버퍼 상태
            total_buffers = 0
            total_frames = 0
            for room_name, room_buffer in self.audio_buffers.items():
                for client_id, buffer in room_buffer.items():
                    total_buffers += 1
                    total_frames += len(buffer)
            
            # 가비지 컬렉션
            gc.collect()
            
            aprint(f"[MEMORY] 사용량: {memory_mb:.1f}MB, 버퍼: {total_buffers}개, 프레임: {total_frames}개")
            
        except ImportError:
            aprint("[MEMORY] psutil이 설치되지 않아 메모리 모니터링을 사용할 수 없습니다")
        except Exception as e:
            aprint(f"[MEMORY] 메모리 모니터링 오류: {e}")
    
    def start_memory_monitoring(self):
        """메모리 모니터링 시작"""
        self.memory_monitor_enabled = True
        self.last_memory_log = time.time()
        aprint("[MEMORY] 메모리 모니터링 시작")
    
    def stop_memory_monitoring(self):
        """메모리 모니터링 중지"""
        self.memory_monitor_enabled = False
        aprint("[MEMORY] 메모리 모니터링 중지")
    
    def cleanup_audio_buffers(self):
        """오디오 버퍼 정리"""
        try:
            total_cleaned = 0
            for room_name, room_buffer in self.audio_buffers.items():
                for client_id, buffer in room_buffer.items():
                    if len(buffer) > self.QUEUE_MAXLEN_FRAMES * 2:  # 큐 크기의 2배 이상이면 정리
                        # 오래된 프레임들 제거 (최신 50%만 유지)
                        keep_count = len(buffer) // 2
                        for _ in range(len(buffer) - keep_count):
                            if buffer:
                                buffer.popleft()
                                total_cleaned += 1
            
            if total_cleaned > 0:
                aprint(f"[CLEANUP] {total_cleaned}개 프레임 정리 완료")
                
        except Exception as e:
            aprint(f"[CLEANUP] 버퍼 정리 오류: {e}")
    
    def force_garbage_collection(self):
        """강제 가비지 컬렉션"""
        try:
            import gc
            collected = gc.collect()
            aprint(f"[GC] {collected}개 객체 정리 완료")
        except Exception as e:
            aprint(f"[GC] 가비지 컬렉션 오류: {e}")
