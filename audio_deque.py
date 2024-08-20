import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Audio parameters
CHUNK = 1024
FORMAT = np.int16
CHANNELS = 1
RATE = 16000
SYNC_INTERVAL = 0.1  # Sync at every 100ms
BUFFER_SIZE = RATE * 10  # Audio Data buffer for 10sec

class AudioProcessor:
    def __init__(self):
        self.rooms: Dict[str, Set[WebSocket]] = {}
        self.audio_buffers: Dict[str, Dict[int, deque]] = {}
        self.executor = ThreadPoolExecutor(max_workers=6)

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
        client_id = id(websocket)
        self.audio_buffers[room_name][client_id] = deque(maxlen=BUFFER_SIZE)
        try:
            while True:
                data = await websocket.receive_bytes()
                audio_data = np.frombuffer(data, dtype=FORMAT)
                self.audio_buffers[room_name][client_id].append(audio_data)
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
            if current_time - last_process_time >= SYNC_INTERVAL:
                room_buffer = self.audio_buffers[room_name]
                combined_audio = []

                # Read data from every client's buffer
                for client_buffer in room_buffer.values():
                    #client_data = client_buffer.get_data()
                    if client_buffer:
                        client_data = client_buffer.popleft()
                    #if len(client_data) > 0:
                        combined_audio.append(client_data)

                if combined_audio:
                    processed_data = self._process_audio(combined_audio)

                    # Send processed data
                    await self.send_processed_audio(room_name, processed_data)

                    # Move buffer pointing position after send data
                    processed_amount = len(processed_data)
                    #for client_buffer in room_buffer.values():
                    #    client_buffer.clear_processed(processed_amount)

                last_process_time = current_time
            await asyncio.sleep(0.001)

    def _process_audio(self, combined_audio):
        # Trim to minimum length for every audio
        min_length = min(len(audio) for audio in combined_audio)
        trimmed_audio = [audio[:min_length] for audio in combined_audio]

        # Audio mixing: use average
        mixed_audio = np.mean(trimmed_audio, axis=0)

        # Apply soft clipping
        mixed_audio = self.soft_clip(mixed_audio / 32767) * 32767

        # Convert to int16 (apply round)
        return np.round(mixed_audio).astype(FORMAT)

    def soft_clip(self, x, threshold=0.9):
        return np.tanh(x / threshold) * threshold

    async def send_processed_audio(self, room_name, processed_data):
        for client in self.rooms[room_name]:
            try:
                await client.send_bytes(processed_data.tobytes())
            except:
                print("Couldn't send data to client")
                pass  # Ignore disconnected clients
