import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
from collections import deque

# Audio parameters
FORMAT = np.int16
CHANNELS = 1
RATE = 8000
SYNC_INTERVAL = 0.2  # Sync at every 100ms
BUFFER_SIZE = RATE * 10  # Audio Data buffer for 10sec

def is_browser(headers):
    return not("python" in headers.get("user-agent", "").lower())

class AudioProcessor:
    def __init__(self):
        self.rooms: Dict[str, Set[WebSocket]] = {}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {}

    def get_num_rooms(self):
        return len(self.rooms)

    def get_rooms(self):
        room_info = []
        for room_name in self.rooms.keys():
            room_info.append({"room_name": room_name, "num_person": len(self.rooms[room_name])})
        return room_info

    async def create_room(self, room_name: str):
        if room_name not in self.rooms:
            self.rooms[room_name] = set()
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
        else:
            client_id = id(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=BUFFER_SIZE)
        try:
            if client_id == -1:
                while True:
                    json_data = await websocket.receive_json()
                    data = json_data["data"]
                    self.audio_buffers[room_name][client_id].append(data)
            else:
                while True:
                    data = await websocket.receive_bytes()
                    self.audio_buffers[room_name][client_id].append(data)
        except WebSocketDisconnect:
            self.rooms[room_name].remove(websocket)
            del self.audio_buffers[room_name][client_id]
            if not self.rooms[room_name]:
                del self.rooms[room_name]
                del self.audio_buffers[room_name]                

    async def process_room_audio(self, room_name: str):
        last_process_time = asyncio.get_event_loop().time()
        while room_name in self.rooms:
            current_time = asyncio.get_event_loop().time()
            time_interval = current_time - last_process_time
            if time_interval >= SYNC_INTERVAL:
                room_buffer = self.audio_buffers[room_name]
                combined_audio = []
                min_length = float("inf")
                active_clients = []

                # First pass: read audio data from buffer and find minimum length
                for client_id, client_buffer in room_buffer.items():
                    #client_data = client_buffer.get_data()
                    if client_buffer:
                        #client_data = client_buffer.popleft()
                        if client_id == -1: # If client is webbrowser
                            float_data = np.array(client_buffer, dtype=np.float32).reshape(-1)
                            client_buffer.clear()
                            audio_data = (float_data*32767).astype(FORMAT)
                        else:
                            byte_data = b''.join(client_buffer)
                            client_buffer.clear()
                            audio_data = np.frombuffer(byte_data, dtype=FORMAT)
                            
                        min_length = min(min_length, len(audio_data))
                        combined_audio.append(audio_data)
                        active_clients.append(client_id)

                if combined_audio:
                    # Second pass: process audio and update buffers
                    for i, client_id in enumerate(active_clients):
                        client_buffer = room_buffer[client_id]
                        audio_data = combined_audio[i]
                        processing_data = audio_data[:min_length]
                        remaining_data = audio_data[min_length:]
                        combined_audio[i] = processing_data

                        # Update buffer with remaining data
                        if client_id == -1:  # If client is webbrowser
                            if len(remaining_data) > 0:
                                client_buffer.appendleft((remaining_data / 32767).tolist())
                        else:
                            if len(remaining_data) > 0:
                                client_buffer.appendleft(remaining_data.tobytes())
                    processed_data = self._process_audio(combined_audio)
                    # Send processed data to all clients non-blocking
                    for client in self.rooms[room_name]:
                        asyncio.create_task(self.send_audio(client, processed_data))
                last_process_time = current_time
            await asyncio.sleep(0.001)

    def _process_audio(self, combined_audio):
        # Remove empty audio
        
        # # Trim to minimum length for every audio
        # min_length = min(len(audio) for audio in combined_audio)
        # print(min_length)
        # trimmed_audio = [audio[:min_length] for audio in combined_audio]

        # Audio mixing: use average
        # mixed_audio = np.mean(trimmed_audio, axis=0)
        mixed_audio = np.mean(combined_audio, axis=0)

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
