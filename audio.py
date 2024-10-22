import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
from collections import deque
from voice_enhance import enhancer
import torch, torchaudio
from scipy.signal import butter, lfilter
from database import engine
from sqlmodel import Session, select
from models import Rooms
from utils.sys import aprint

# Audio parameters
FORMAT = np.int16
CHANNELS = 1
RATE = 16000
FRAME_SIZE = 128
BUFFER_SIZE = int(RATE / FRAME_SIZE) * 2 # buffer size 2sec
SYNC_INTERVAL = (FRAME_SIZE / RATE) * 64 # Sync interval for processing audio is about 0.512 sec


def butter_lowpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=10):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

class AudioProcessor:
    def __init__(self):
        self.rooms: Dict[str, list[WebSocket]] = {} # room_name, 웹소켓 객체 set
        self.client_info: Dict[int, dict] = {} # client_id, dtype, sr, is_webbrowser
        self.audio_buffers: Dict[str, Dict[int, deque]] = {} # room_name, 오디오 버퍼 client_id, deque
        self.use_voice_enhance = False
        self.mute_me = False

    def is_webbrowser(self, websocket: WebSocket):
        headers = websocket.headers
        return not("python" in headers.get("user-agent", "").lower())
            
    def add_person_to_room(self, room_name: str, sr: int, dtype: str, websocket: WebSocket):
        client_id = id(websocket)
        self.rooms[room_name].append(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=BUFFER_SIZE)
        self.client_info[client_id] = {"sr": sr, "dtype": dtype, "is_webbrowser": self.is_webbrowser(websocket)}
        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            room.num_person = room.num_person + 1
            session.add(room)
            session.commit()
        return client_id
    
    def remove_person_from_room(self, room_name: str, websocket: WebSocket):
        client_id = id(websocket)
        if self.rooms.get(room_name):
            self.rooms[room_name].remove(websocket)
            del self.audio_buffers[room_name][client_id]
            del self.client_info[client_id]
            with Session(engine) as session:
                room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
                room.num_person = room.num_person - 1
                session.add(room)
                session.commit()
        return client_id

    # def get_num_rooms(self):
    #     return len(self.rooms)

    def get_rooms(self):
        room_info = []
        for room_name in self.rooms.keys():
            room_info.append({"room_name": room_name, "num_person": len(self.rooms[room_name])})
        return room_info
    
    async def create_room(self, room_name: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = []
            self.audio_buffers[room_name] = {}
            asyncio.create_task(self.process_room_audio(room_name))
        return {"message": f"Room '{room_name}' created successfully"}
    
    async def talk_to_room_buffer(self, room_name: str, client_id: int, client_ws: WebSocket):
        if self.client_info[client_id]["dtype"] == "float32":
            while True:
                # json_data = await client_ws.receive_json()
                # data = json_data["data"]
                byte_data = await client_ws.receive_bytes()
                data = np.frombuffer(byte_data, dtype=np.float32).reshape(-1)
                # print(f"Received data from webbrowser: {len(data)}")
                if len(data) > FRAME_SIZE:
                    for i in range(0, len(data), FRAME_SIZE):
                        self.audio_buffers[room_name][client_id].append(data[i:i+FRAME_SIZE])
        else:
            while True:
                data = await client_ws.receive_bytes()
                # data = np.frombuffer(data, dtype=FORMAT)
                data = np.frombuffer(byte_data, dtype=np.float32).reshape(-1)
                print(f"Received data from device: {len(data)}")
                self.audio_buffers[room_name][client_id].append(data)

    async def join_room(self, room_name: str, sr: int, dtype: str, websocket: WebSocket):
        if room_name not in self.rooms:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        client_id = self.add_person_to_room(room_name, sr, dtype, websocket)
        print(f"Client {client_id} joined room '{room_name}'")
        try:
            await self.talk_to_room_buffer(room_name, client_id, websocket)
        except WebSocketDisconnect:
            self.remove_person_from_room(room_name, websocket)
            print(f"Client {client_id} exited room '{room_name}'")
            #if (not self.rooms[room_name]) and (room_name == '보온팀') :
            #    del self.rooms[room_name]
            #    del self.audio_buffers[room_name]                

    async def process_room_audio(self, room_name: str):
        last_process_time = asyncio.get_event_loop().time()
        while room_name in self.rooms:
            current_time = asyncio.get_event_loop().time()
            time_interval = current_time - last_process_time
            if time_interval >= SYNC_INTERVAL:
                room_buffer = self.audio_buffers[room_name]
                combined_audio_int16 = []
                active_clients = []
                read_buffer_len = []
                min_length = float("inf")

                # First pass: read audio data from buffer and find minimum length and find minimum dtype
                for client_id, client_buffer in room_buffer.items():
                    print(f"Client {client_id} buffer length: {len(client_buffer)}")
                    if client_buffer:
                        active_clients.append(client_id) # combined_audio의 순서대로 client_id 저장
                        print(f"Client {client_id} buffer length: {len(client_buffer)}")
                        read_buffer_len.append(len(client_buffer))
                        # 클라이언트들의 데이터를 int16(클라이언트들의 최소 데이터 타입)으로 일괄 변환
                        if self.client_info[client_id]["dtype"] == "float32":
                            print("what is wrong with you")
                            try:
                                float_data = np.array(client_buffer, dtype='float32').reshape(-1)
                            except Exception as e:
                                print(e)
                            print("Oh my god")
                            audio_data = (float_data*32767).astype(FORMAT)
                            print("Im not sure")
                        elif self.client_info[client_id]["dtype"] == "int16":
                            #byte_data = b''.join(client_buffer)
                            #audio_data = np.frombuffer(byte_data, dtype=FORMAT)
                            audio_data = np.array(client_buffer, dtype=FORMAT).reshape(-1)
                        min_length = min(min_length, len(audio_data))
                        combined_audio_int16.append(audio_data)
                        print(f"Combined audio: {audio_data.dtype}, {audio_data.shape}")
                print(f"Active clients: {active_clients}")
                print(f"Read buffer length: {read_buffer_len}")

                # Second pass: if received data is valid, process audio and update buffers
                if combined_audio_int16 and (min_length >= FRAME_SIZE):
                    min_length = min_length - (min_length % FRAME_SIZE)
                    for i, client_id in enumerate(active_clients):
                        client_buffer = room_buffer[client_id]
                        audio_data = combined_audio_int16[i]
                        processing_data = audio_data[:min_length]
                        remaining_data = audio_data[min_length:]
                        combined_audio_int16[i] = processing_data
                        # Remove buffer for processed data
                        for _ in range(read_buffer_len[i]):
                            client_buffer.popleft()
                        # Update buffer with remaining data
                        if self.client_info[client_id]["dtype"] == "float32":
                            print(f"Client {client_id} remaining data length: {len(remaining_data)}")
                            if len(remaining_data) > 0:
                                # append remaining data to buffer in order by reversing
                                for j in range(len(remaining_data), 0, -FRAME_SIZE):
                                    client_buffer.appendleft((remaining_data[j-FRAME_SIZE:j]/32767).astype(np.float32))
                        else:
                            if len(remaining_data) > 0:
                                for j in range(len(remaining_data), 0, -FRAME_SIZE):
                                    client_buffer.appendleft(remaining_data)
                                # for j in range(len(remaining_data), 0, -FRAME_SIZE):
                                #     client_buffer.appendleft(remaining_data[j-FRAME_SIZE:j])
                        if self.use_voice_enhance:
                            combined_audio_int16[i] = apply_lowpass_filter(combined_audio_int16[i], 2000, RATE)
                    # Send processed data to all clients non-blocking
                    for i, client_id in enumerate(active_clients):
                        print(f"Sending processed data to client {client_id}")
                        # print(f"Sending processed data to client {len(processed_data)}")
                        if self.mute_me == False:
                            exclude_idx = None
                        asyncio.create_task(
                            self.individual_audio_task(
                                combined_audio_int16, 
                                i, 
                                self.rooms[room_name][i]
                                )
                            )
                last_process_time = current_time
            await asyncio.sleep(0.01)

    def _process_audio(self, combined_audio, exclude_client_idx):
        selected_audio = combined_audio.copy()
        if exclude_client_idx != None and len(selected_audio) > 1:
            # Exclude client itself's audio 
            # selected_audio = np.delete(selected_audio, exclude_client_idx, 0)
            del selected_audio[exclude_client_idx]
        # Audio mixing: use average
        mixed_audio_int16 = np.mean(selected_audio, axis=0, dtype=FORMAT)
        if self.use_voice_enhance:
            torch_audio = torchaudio.transforms.Resample(orig_freq=RATE, new_freq=16000)(torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0))            
            torch_audio = torch_audio.to("cuda:1")
            mixed_audio = enhancer(torch_audio)
            mixed_audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=RATE)(mixed_audio.cpu()).squeeze(0)
            mixed_audio = mixed_audio.numpy()
            
        # # Apply soft clipping
        # mixed_audio = self.soft_clip(mixed_audio / 32767) * 32767
        
        return mixed_audio_int16
        
            

    def soft_clip(self, x, threshold=0.9):
        return np.tanh(x / threshold) * threshold
    
    async def send_audio(self, ws, client_id, processed_data_int16):
        try:
            if self.client_info[client_id]["dtype"] == "float32":
                await ws.send_bytes((processed_data_int16/32767).astype(np.float32).tobytes())
            else:
                await ws.send_bytes(processed_data_int16.tobytes())
        except:
            print("Couldn't send data to client")
            pass  # Ignore disconnected clients

    async def individual_audio_task(self, combined_audio, exclude_client_idx, ws):
        processed_data_int16 = self._process_audio(combined_audio, exclude_client_idx)
        client_id = id(ws)
        await self.send_audio(ws, client_id, processed_data_int16)
