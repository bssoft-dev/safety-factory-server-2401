import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict
import asyncio
from collections import deque
from database import engine
from sqlmodel import Session, select
from models import Rooms, Devices
from utils.sys import aprint
from bs_sound_utils.audio_utils import AudioUtils


class VoiceChat(AudioUtils):
    def __init__(self):
        super().__init__()
        self.rooms: Dict[str, list[WebSocket]] = {} # {room_name : [웹소켓 객체 set]}
        self.client_info: Dict[int, dict] = {} # {client_id: {dtype: , sr: , is_webbrowser: , person_name: }}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {} # {room_name: { client_id: 오디오버퍼 deque}}
        self.use_voice_enhance = True
        self.hear_me = False
        self.save_audio = False
        self.keep_test_room = True
        self.classify_event = False
        self.do_stt = False
        self.enhance_volume = 0
        self.input_rec_buffer: Dict[int, list] = {} # client_id, 오디오 버퍼
        self.output_rec_buffer: Dict[int, list] = {} # client_id, 오디오 버퍼

    def is_webbrowser(self, websocket: WebSocket):
        headers = websocket.headers
        return not("python" in headers.get("user-agent", "").lower())
            
    def add_person_to_room(self, room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
        client_id = id(websocket)
        self.rooms[room_name].append(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=self.BUFFER_SIZE)
        if self.save_audio:
            self.input_rec_buffer[client_id] = []
            self.output_rec_buffer[client_id] = []
        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            devices = session.exec(select(Devices).filter(Devices.device_id == device_id)).first()
            person_name = '웹접속' if devices is None else devices.owner
            if room.num_person == 0:
                room.persons = f"{person_name}"
            else:
                room.persons = f"{room.persons}, {person_name}"
            room.num_person = room.num_person + 1
            session.add(room)
            session.commit()
        self.client_info[client_id] = {"sr": sr, "dtype": dtype, "is_webbrowser": self.is_webbrowser(websocket), "person_name": person_name}
        return client_id
    
    def remove_person_from_room(self, room_name: str, websocket: WebSocket):
        client_id = id(websocket)
        person_name = self.client_info[client_id]["person_name"]
        if self.rooms.get(room_name):
            self.rooms[room_name].remove(websocket)
            del self.audio_buffers[room_name][client_id]
            del self.client_info[client_id]
            with Session(engine) as session:
                room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
                room.num_person = room.num_person - 1
                if room.num_person == 0:
                    room.persons = ''
                else:
                    room.persons = room.persons.replace(f", {person_name}", '', 1)
                session.add(room)
                session.commit()
        return client_id

    def get_rooms(self):
        room_info = []
        for room_name in self.rooms.keys():
            room_info.append({"room_name": room_name, "num_person": len(self.rooms[room_name])})
        return room_info
    
    async def create_room(self, room_name: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = []
            self.audio_buffers[room_name] = {}
            asyncio.create_task(self.process_a_room_audio_runner(room_name))
        return {"message": f"Room '{room_name}' created successfully"}
    
    async def delete_room(self, id: int):
        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.id == id)).first()
            if room is None:
                return None
            room_name = room.room_name
            if room_name in self.rooms:
                for ws in self.rooms[room_name]:
                    try:
                        await ws.close(code=4004)
                    except:
                        pass
                del self.rooms[room_name]
                del self.audio_buffers[room_name]
            room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            session.delete(room)
            session.commit()
        return {"message": f"Room '{room_name}' deleted successfully"}
    
    async def talk_to_room_runner(self, room_name: str, client_id: int, client_ws: WebSocket):
        while True:
            try:
                byte_data = await client_ws.receive_bytes()
                # print(f"Received data from client {client_id}: {len(byte_data)}")
            # except (WebSocketDisconnect, KeyError, RuntimeError):
            except:
                await self.exit_room(room_name, client_ws)
                if self.save_audio:
                    self.save_recordings(self.input_rec_buffer[client_id], self.output_rec_buffer[client_id])
                # 보온팀 방이 비어있으면 방 삭제
                if (not self.rooms[room_name]) and (room_name == '보온팀') and (not self.keep_test_room):
                   del self.rooms[room_name]
                   del self.audio_buffers[room_name]  
                return
            if self.client_info[client_id]["dtype"] == "float32": # webbrowser
                data_f32 = np.frombuffer(byte_data, dtype=np.float32).reshape(-1)
                data_int16 = self.float32_to_int16(data_f32)
                # print(f"Received data from webbrowser: {len(data)}")
            else:
                if self.client_info[client_id]["sr"] == 16000: # edge device
                    data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                    # enhance volume
                    if self.enhance_volume > 0:
                        data_int16 = data_int16 * self.enhance_volume
                elif self.client_info[client_id]["sr"] == 48000: # sync node device
                    # 48000 -> 16000
                    data_int16 = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)[::3]
                # print(f"Received data from device: {len(data)}")
            if len(data_int16) >= self.FRAME_SIZE:
                for i in range(0, len(data_int16), self.FRAME_SIZE):
                    self.audio_buffers[room_name][client_id].append(data_int16[i:i+self.FRAME_SIZE])

    async def join_room(self, room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
        if room_name not in self.rooms:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        client_id = self.add_person_to_room(room_name, sr, dtype, device_id, websocket)
        print(f"Client {client_id} joined room '{room_name}'")
        await self.talk_to_room_runner(room_name, client_id, websocket)
    
    async def exit_room(self, room_name: str, websocket: WebSocket):
        client_id = self.remove_person_from_room(room_name, websocket)
        print(f"Client {client_id} exited room '{room_name}'")

    async def process_a_room_audio_runner(self, room_name: str):
        last_process_time = asyncio.get_event_loop().time()
        while room_name in self.rooms:
            current_time = asyncio.get_event_loop().time()
            time_interval = current_time - last_process_time
            if time_interval >= self.SYNC_INTERVAL:
                room_buffer = self.audio_buffers[room_name]
                combined_audio_int16 = []
                active_clients = []
                # read_buffer_len = []
                min_length = float("inf")

                # First: find minimum length of buffers
                for ws_idx, ws in enumerate(self.rooms[room_name]):
                    client_id = id(ws)
                    if len(room_buffer[client_id]) > 0:
                        active_clients.append((ws_idx, client_id))
                        min_length = min(min_length, len(room_buffer[client_id]))
                # Second: read audio data from buffer and all combine
                if len(active_clients) > 0:
                    print(f"Active clients: {len(active_clients)}")
                    for ws_idx, client_id in active_clients:
                        print(f"Client {client_id} buffer length: {len(room_buffer[client_id])}")
                        buffers = []
                        for _ in range(min_length):
                            buffers.append(room_buffer[client_id].popleft())
                        combined_audio_int16.append(np.array(buffers, dtype=self.FORMAT).reshape(-1))
                        # if self.use_voice_enhance:
                        #     combined_audio_int16[i] = apply_lowpass_filter(combined_audio_int16[i])
                    if self.classify_event:
                        asyncio.create_task(self.classify_audio(combined_audio_int16, room_name))
                    if self.do_stt:
                        asyncio.create_task(self.stt_audio(combined_audio_int16))
                # Third: Mix audio and send to all clients non-blocking
                    for audio_idx, (ws_idx, client_id) in enumerate(active_clients):
                        if self.hear_me == True:
                            exclude_idx = None
                        else:
                            exclude_idx = audio_idx
                        asyncio.create_task(
                            self.talk_to_each_client(
                                combined_audio_int16, 
                                exclude_idx, 
                                self.rooms[room_name][ws_idx]
                                )
                            )
                last_process_time = current_time
            await asyncio.sleep(0.01)

    async def talk_to_each_client(self, combined_audio, exclude_client_idx, ws):
        processed_data_int16 = self.mix_audio(combined_audio, exclude_client_idx)
        if self.use_voice_enhance:
            processed_data_int16 = self.voice_enhance(processed_data_int16)
        # # Apply soft clipping
        # processed_data_int16 = self.soft_clip(processed_data_int16 / 32767) * 32767
        client_id = id(ws)
        await self.send_audio(ws, client_id, processed_data_int16)
        if self.save_audio:
            self.input_rec_buffer[client_id].append(combined_audio[exclude_client_idx])
            self.output_rec_buffer[client_id].append(processed_data_int16)