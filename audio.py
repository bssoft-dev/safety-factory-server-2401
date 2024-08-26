import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
from collections import deque
from voice_enhance import enhancer
import torch, torchaudio
from scipy.signal import butter, lfilter
# import noisereduce as nr
# from type_converter import int16_to_float32, float32_to_int16

# Audio parameters
FORMAT = np.int16
CHANNELS = 1
RATE = 8000
SYNC_INTERVAL = 0.2  # Sync at every 100ms
BUFFER_SIZE = RATE * 10  # Audio Data buffer for 10sec
FRAME_SIZE = 2048

def is_browser(headers):
    return not("python" in headers.get("user-agent", "").lower())

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
        self.rooms: Dict[str, Set[WebSocket]] = {}
        self.rooms_num_person: Dict[str, int] = {}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {}
        self.use_voice_enhance = False

    def get_num_rooms(self):
        return len(self.rooms)

    def get_rooms(self):
        room_info = []
        for room_name in self.rooms.keys():
            room_info.append({"room_name": room_name, "num_person": self.rooms_num_person[room_name]})
        return room_info
    
    async def create_room(self, room_name: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = set()
            self.rooms_num_person[room_name] = 0
            self.audio_buffers[room_name] = {}
            asyncio.create_task(self.process_room_audio(room_name))
        return {"message": f"Room '{room_name}' created successfully"}

    async def join_room(self, websocket: WebSocket, room_name: str):
        if room_name not in self.rooms:
            await websocket.close(code=4004)
            return
        await websocket.accept()
        self.rooms[room_name].add(websocket)
        headers = websocket.headers
        if is_browser(headers):
            client_id = -1
            self.rooms_num_person[room_name] += 1
        else:
            client_id = id(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=BUFFER_SIZE)
        try:
            if client_id == -1:
                while True:
                    json_data = await websocket.receive_json()
                    data = json_data["data"]
                    for i in range(0, len(data), FRAME_SIZE):
                        self.audio_buffers[room_name][client_id].append(data[i:i+FRAME_SIZE])
            else:
                while True:
                    data = await websocket.receive_bytes()
                    self.audio_buffers[room_name][client_id].append(data)
        except WebSocketDisconnect:
            if client_id == -1:
                self.rooms_num_person[room_name] -= 1
            self.rooms[room_name].remove(websocket)
            del self.audio_buffers[room_name][client_id]
            # if not self.rooms[room_name]:
            #     del self.rooms[room_name]
            #     del self.audio_buffers[room_name]                

    async def process_room_audio(self, room_name: str):
        last_process_time = asyncio.get_event_loop().time()
        while room_name in self.rooms:
            current_time = asyncio.get_event_loop().time()
            time_interval = current_time - last_process_time
            if time_interval >= SYNC_INTERVAL:
                room_buffer = self.audio_buffers[room_name]
                combined_audio = []
                active_clients = []
                read_buffer_len = []
                min_length = float("inf")

                # First pass: read audio data from buffer and find minimum length
                for client_id, client_buffer in room_buffer.items():
                    if client_buffer:
                        active_clients.append(client_id)
                        read_buffer_len.append(len(client_buffer))
                        if client_id == -1: # If client is webbrowser
                            float_data = np.array(client_buffer, dtype=np.float32).reshape(-1)
                            audio_data = (float_data*32767).astype(FORMAT)
                        else:
                            byte_data = b''.join(client_buffer)
                            audio_data = np.frombuffer(byte_data, dtype=FORMAT)
                        min_length = min(min_length, len(audio_data))
                        combined_audio.append(audio_data)
                        

                if combined_audio and (min_length >= FRAME_SIZE):
                    min_length = min_length - (min_length % FRAME_SIZE)
                    # Second pass: process audio and update buffers
                    for i, client_id in enumerate(active_clients):
                        client_buffer = room_buffer[client_id]
                        audio_data = combined_audio[i]
                        processing_data = audio_data[:min_length]
                        remaining_data = audio_data[min_length:]
                        combined_audio[i] = processing_data
                        # Remove buffer for processed data
                        for _ in range(read_buffer_len[i]):
                            client_buffer.popleft()
                        # Update buffer with remaining data
                        if client_id == -1:  # If client is webbrowser
                                # client_buffer.appendleft((remaining_data / 32767).tolist())
                            if len(remaining_data) > 0:
                                # append remaining data to buffer in order by reversing
                                for j in range(len(remaining_data), 0, -FRAME_SIZE):
                                    client_buffer.appendleft((remaining_data[j-FRAME_SIZE:j]/32767).tolist())
                        else:
                            if len(remaining_data) > 0:
                                client_buffer.appendleft(remaining_data.tobytes())
                        if self.use_voice_enhance:
                            combined_audio[i] = apply_lowpass_filter(combined_audio[i], 2000, RATE)
                    processed_data = self._process_audio(combined_audio)
                    
                    # Send processed data to all clients non-blocking
                    for client in self.rooms[room_name]:
                        asyncio.create_task(self.send_audio(client, processed_data))
                last_process_time = current_time
            await asyncio.sleep(0.01)

    def _process_audio(self, combined_audio):
        # Remove empty audio
        
        # # Trim to minimum length for every audio
        # min_length = min(len(audio) for audio in combined_audio)
        # print(min_length)
        # trimmed_audio = [audio[:min_length] for audio in combined_audio]

        # Audio mixing: use average
        # mixed_audio = np.mean(trimmed_audio, axis=0)
        mixed_audio = np.mean(combined_audio, axis=0)
        if self.use_voice_enhance:
            torch_audio = torchaudio.transforms.Resample(orig_freq=RATE, new_freq=16000)(torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0))            
            torch_audio = torch_audio.to("cuda:0")
            # torch_audio = torch.tensor(mixed_audio, dtype=torch.float32).unsqueeze(0)
            mixed_audio = enhancer(torch_audio)
            mixed_audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=RATE)(mixed_audio.cpu()).squeeze(0)
            mixed_audio = mixed_audio.numpy()
            
            # # mixed_audio = mixed_audio.squeeze(0).numpy()
            # mixed_audio = torchaudio.transforms.Resample(orig_freq=16000, new_freq=RATE)(mixed_audio).squeeze(0)
            # mixed_audio = mixed_audio.detach().numpy()            

            # data = int16_to_float32(mixed_audio)
            # nData = int16_to_float32(np.frombuffer(b''.join(noise), np.int16))
            # data = nr.reduce_noise(audio_clip=data, noise_clip=nData,
            #                        verbose=False, n_std_thresh=1.5, prop_decrease=1,
            #                        win_length=WIN_LENGTH, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
            #                        n_grad_freq=4)
            
        # Apply soft clipping
        mixed_audio = self.soft_clip(mixed_audio / 32767) * 32767

        # Convert to int16 (apply round)
        return np.round(mixed_audio).astype(FORMAT)
        # return np.round(mixed_audio)

    def soft_clip(self, x, threshold=0.9):
        return np.tanh(x / threshold) * threshold
    
    async def send_audio(self, client, processed_data):
        try:
            await client.send_bytes(processed_data.tobytes())
        except:
            print("Couldn't send data to client")
            pass  # Ignore disconnected clients

    async def send_processed_audio(self, room_name, processed_data):
        for client in self.rooms[room_name]:
            try:
                await client.send_bytes(processed_data.tobytes())
            except:
                print("Couldn't send data to client")
                pass  # Ignore disconnected clients
