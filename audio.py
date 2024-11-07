import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
from collections import deque
from voice_enhance import enhancer
import torch, torchaudio
from scipy.signal import butter, lfilter, resample
from database import engine
from sqlmodel import Session, select
from models import Rooms, Events
from utils.sys import aprint
import wave, struct, aiohttp

# Audio parameters
FORMAT = np.int16
CHANNELS = 1
RATE = 16000
FRAME_SIZE = 2048
BUFFER_SIZE = int(RATE / FRAME_SIZE) * 2 # buffer size 2sec
SYNC_INTERVAL = (FRAME_SIZE / RATE) * 4 # Sync interval for processing audio is about 0.512 sec

byte_headers = {"Content-Type": "application/octet-stream"}

def butter_lowpass(cutoff, fs, order=10):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_lowpass(2000, RATE, order=10)

def apply_lowpass_filter(data):
    y = lfilter(b, a, data)
    #float64 -> int16
    y = (y*32767).astype(FORMAT)
    return y

def save_wav(filename, rate, data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(rate)
        wf.writeframes(struct.pack('%dh' % len(data), *data))

def save_recordings(input_rec_buffer, output_rec_buffer):
    input_filename = "input.wav"
    output_filename = "output.wav"
    fs_target = 16000
    # 입력 오디오 저장
    save_wav(input_filename, fs_target, np.concatenate(input_rec_buffer))
    print(f"Input audio saved as {input_filename}")
    # 출력 오디오 저장
    save_wav(output_filename, fs_target, np.concatenate(output_rec_buffer))
    print(f"Output audio saved as {output_filename}")

class AudioProcessor:
    def __init__(self):
        self.rooms: Dict[str, list[WebSocket]] = {} # room_name, 웹소켓 객체 set
        self.client_info: Dict[int, dict] = {} # client_id, dtype, sr, is_webbrowser
        self.audio_buffers: Dict[str, Dict[int, deque]] = {} # room_name, 오디오 버퍼 client_id, deque
        self.use_voice_enhance = False
        self.mute_me = True
        self.save_audio = False
        self.keep_test_room = True
        self.classify_event = False
        self.do_stt = False
        self.input_rec_buffer: Dict[int, list] = {} # client_id, 오디오 버퍼
        self.output_rec_buffer: Dict[int, list] = {} # client_id, 오디오 버퍼

    def is_webbrowser(self, websocket: WebSocket):
        headers = websocket.headers
        return not("python" in headers.get("user-agent", "").lower())
            
    def add_person_to_room(self, room_name: str, sr: int, dtype: str, websocket: WebSocket):
        client_id = id(websocket)
        self.rooms[room_name].append(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=BUFFER_SIZE)
        if self.save_audio:
            self.input_rec_buffer[client_id] = []
            self.output_rec_buffer[client_id] = []
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

    async def classify_audio(self, combined_audio, room_name: str):
        processed_data_int16 = self._process_audio(combined_audio, exclude_client_idx = None, remove_noise = False)
        send_data = processed_data_int16.tobytes()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api-2106.bs-soft.co.kr/v2401/classify-audio/byte", 
                    data = send_data,
                    headers=byte_headers
                ) as res:
                    if res.status == 200:
                        data = await res.json()
                        print(data)
                        if data["is_danger"] == 1:
                            with Session(engine) as session:
                                session.add(Events(room_name=room_name, event=data["event"], time=data["time"]))
                                session.commit()
                    else:
                        print(f"Classify audio failed: {res.status}")
        except Exception as e:
            print(f"Classify audio error: {e}")
    
    async def stt_audio(self, audio_data):
        pass

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
        while True:
            try:
                byte_data = await client_ws.receive_bytes()
            except (WebSocketDisconnect, KeyError):
                self.remove_person_from_room(room_name, client_ws)
                print(f"Client {client_id} exited room '{room_name}'")
                if self.save_audio:
                    save_recordings(self.input_rec_buffer[client_id], self.output_rec_buffer[client_id])
                # 보온팀 방이 비어있으면 방 삭제
                if (not self.rooms[room_name]) and (room_name == '보온팀') and (not self.keep_test_room):
                   del self.rooms[room_name]
                   del self.audio_buffers[room_name]  
                return
            if self.client_info[client_id]["dtype"] == "float32": # webbrowser
                data = np.frombuffer(byte_data, dtype=np.float32).reshape(-1)
                # print(f"Received data from webbrowser: {len(data)}")
                if len(data) >= FRAME_SIZE:
                    for i in range(0, len(data), FRAME_SIZE):
                        self.audio_buffers[room_name][client_id].append(data[i:i+FRAME_SIZE])
            else:
                if self.client_info[client_id]["sr"] == 16000: # edge device
                    data = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)
                    # enhance volume
                    data = data * 30
                elif self.client_info[client_id]["sr"] == 48000: # sync node device
                    # 48000 -> 16000
                    data = np.frombuffer(byte_data, dtype=np.int16).reshape(-1)[::3]
                # print(f"Received data from device: {len(data)}")
                self.audio_buffers[room_name][client_id].append(data)

    async def join_room(self, room_name: str, sr: int, dtype: str, websocket: WebSocket):
        if room_name not in self.rooms:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        client_id = self.add_person_to_room(room_name, sr, dtype, websocket)
        print(f"Client {client_id} joined room '{room_name}'")
        await self.talk_to_room_buffer(room_name, client_id, websocket)             

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
                        #print(f"Client {client_id} buffer length: {len(client_buffer)}")
                        read_buffer_len.append(len(client_buffer))
                        # 클라이언트들의 데이터를 int16(클라이언트들의 최소 데이터 타입)으로 일괄 변환
                        if self.client_info[client_id]["dtype"] == "float32":
                            try:
                                float_data = np.array(client_buffer, dtype='float32').reshape(-1)
                            except Exception as e:
                                print(e)
                            audio_data = (float_data*32767).astype(FORMAT)
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
                        # remaining_data = audio_data[min_length:]
                        combined_audio_int16[i] = processing_data
                        # Remove buffer for processed data
                        # for _ in range(read_buffer_len[i]):
                        for _ in range(min_length//FRAME_SIZE):
                            client_buffer.popleft()
                        # if self.use_voice_enhance:
                        #     combined_audio_int16[i] = apply_lowpass_filter(combined_audio_int16[i])
                    # 위험 사운드 감지
                    if self.classify_event:
                        asyncio.create_task(self.classify_audio(combined_audio_int16, room_name))
                    if self.do_stt:
                        asyncio.create_task(self.stt_audio(combined_audio_int16))
                    # Send processed data to all clients non-blocking
                    for i, client_id in enumerate(active_clients):
                        print(f"Sending processed data to client {client_id}")
                        # print(f"Sending processed data to client {len(processed_data)}")
                        if self.mute_me == False:
                            exclude_idx = None
                        else:
                            exclude_idx = i
                        asyncio.create_task(
                            self.individual_audio_task(
                                combined_audio_int16, 
                                exclude_idx, 
                                self.rooms[room_name][i]
                                )
                            )
                last_process_time = current_time
            await asyncio.sleep(0.01)

    def _process_audio(self, combined_audio, exclude_client_idx, remove_noise = True) -> np.ndarray:
        selected_audio = combined_audio.copy()
        if (exclude_client_idx != None) and (len(selected_audio) > 1):
            # Exclude client itself's audio from mixing
            del selected_audio[exclude_client_idx]
        # Audio mixing: use average
        mixed_audio_int16 = np.mean(np.array(selected_audio), axis=0, dtype=FORMAT)
        if self.use_voice_enhance and remove_noise:
            torch_audio = torch.tensor(mixed_audio_int16/32767, dtype=torch.float32).unsqueeze(0)
            torch_audio = torch_audio.to("cuda:1")
            mixed_audio = enhancer(torch_audio)
            mixed_audio = mixed_audio.cpu().squeeze(0).numpy()
            print(f"Mixed audio: {mixed_audio.dtype}, {mixed_audio.shape}")
            mixed_audio_int16 = (mixed_audio*32767).astype(FORMAT)
            
        # # Apply soft clipping
        # mixed_audio = self.soft_clip(mixed_audio / 32767) * 32767
        
        return mixed_audio_int16

    def soft_clip(self, x, threshold=0.9):
        return np.tanh(x / threshold) * threshold
    
    async def send_audio(self, ws, client_id, processed_data_int16):
        try:
            if self.client_info[client_id]["dtype"] == "float32":
                # float32 -> int16
                await ws.send_bytes((processed_data_int16/32767).astype(np.float32).tobytes())
            else:
                if self.client_info[client_id]["sr"] == 48000:
                    # upsample 16000 -> 48000
                    processed_data_int16 = np.repeat(processed_data_int16, 3)
                await ws.send_bytes(processed_data_int16.tobytes())
        except:
            print("Couldn't send data to client")
            pass  # Ignore disconnected clients

    async def individual_audio_task(self, combined_audio, exclude_client_idx, ws):
        processed_data_int16 = self._process_audio(combined_audio, exclude_client_idx)
        client_id = id(ws)
        await self.send_audio(ws, client_id, processed_data_int16)
        if self.save_audio:
            self.input_rec_buffer[client_id].append(combined_audio[exclude_client_idx])
            self.output_rec_buffer[client_id].append(processed_data_int16)
