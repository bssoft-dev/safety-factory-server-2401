import numpy as np
import os, time, asyncio
import aiofiles, struct
from scipy.signal import butter, lfilter
import torch
from utils.sys import aprint

from services.voice_enhance import VoiceEnhancer, LightVoiceEnhancer, VoiceEnhancer2
from services.event_classification import EventClassifier
from services.stt import SttProcessor

class AudioUtils:
    def __init__(self):
        # Audio parameters
        self.FORMAT = np.int16
        self.CHANNELS = 1
        self.RATE = 16000
        self.FRAME_SIZE = 1024
        self.BUFFER_SIZE = int(self.RATE / self.FRAME_SIZE) * 1 # buffer size about 2 sec
        self.save_audio_dir = "./recordings"
        self.save_audio_sec = 60
        self.save_audio_len = self.RATE * self.save_audio_sec
        self.SYNC_INTERVAL = (self.FRAME_SIZE / self.RATE) * 4 # Sync interval for processing audio is about 0.512 sec
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.b, self.a = self.butter_lowpass(2000, self.RATE, order=10)
        self.voice_enhancer = VoiceEnhancer(self.device)
        # self.voice_enhancer = VoiceEnhancer2(self.device)
        # self.voice_enhancer = LightVoiceEnhancer(self.device)
        self.event_classifier = EventClassifier(self.device)
        self.stt = SttProcessor()
        self.input_rec_buffer: dict[int, list] = {} # client_id, 오디오 버퍼
        self.output_rec_buffer: dict[int, list] = {} # client_id, 오디오 버퍼
        self.async_save_audio = {"input": False, "output": False}
        
    def butter_lowpass(self, cutoff, fs, order=10):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_lowpass_filter(self, data):
        y = lfilter(self.b, self.a, data)
        #float64 -> int16
        y = (y*32767).astype(self.FORMAT)
        return y

    def soft_clip(self, x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        부드러운 클리핑으로 오디오 왜곡 방지
        threshold: 클리핑이 시작되는 임계값 (0~1)
        """
        mask = np.abs(x) > threshold
        x[mask] = threshold * np.tanh(x[mask] / threshold)
        return x
    
    def int16_to_torch_float32(self, in_data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(in_data).to(device=self.device, dtype=torch.float32) / 32768.0
    
    def torch_float32_to_int16(self, in_data: torch.Tensor) -> np.ndarray:
        return (in_data*32767).cpu().numpy().astype(np.int16)

    def int16_to_float32(self, data: np.ndarray) -> np.ndarray:
        if np.max(np.abs(data)) > 32768:
            raise ValueError("Data has values above 32768")
        return (data / 32768.0).astype("float32")
    
    def float32_to_int16(self, data: np.ndarray) -> np.ndarray:
        if np.max(data) > 1:
            data = data / np.max(np.abs(data))
        return np.array(data * 32767).astype("int16")
    
    def exclude_client_audio(self, data: list[np.ndarray], exclude_idx: int | None = None) -> list[np.ndarray]:
        if exclude_idx != None:
            if len(data) > 1:
                return np.delete(np.array(data), exclude_idx, axis=0)
            else:
                return np.zeros_like(data[0]).reshape(1, -1)
        return np.array(data)
    
    def mix_audio_to_torch(self, data: list[np.ndarray], exclude_idx: int | None = None) -> torch.Tensor:
        data_array = self.exclude_client_audio(data, exclude_idx)
        return torch.mean(torch.from_numpy(data_array).to(device=self.device, dtype=torch.float32) / 32768.0, dim=0)
    
    def mix_audio(self, data: list[np.ndarray], exclude_idx: int | None = None) -> np.ndarray:
        dtype = data[0].dtype
        data_array = self.exclude_client_audio(data, exclude_idx)               
        return np.mean(data_array, axis=0).astype(dtype)
    
    async def classify_audio(self, audio: list[np.ndarray], room_name: str):
        processed_data_int16 = self.mix_audio(audio)
        audio_data = self.int16_to_float32(processed_data_int16)
        await self.event_classifier.infer(audio_data, room_name)
    
    async def stt_audio(self, data: list[np.ndarray], room_name: str):
        processed_data_int16 = self.mix_audio(data)
        send_data = processed_data_int16.tobytes()
        await self.stt.send_audio(send_data, room_name)

    def voice_enhance(self, in_data: torch.Tensor, client_id: int) -> np.ndarray:
        try:
            enhanced_audio = self.voice_enhancer.denoise(in_data, client_id)
            return self.torch_float32_to_int16(enhanced_audio)
        except Exception as e:
            print(f"Error in voice enhance: {e}")
            return self.torch_float32_to_int16(in_data)

    async def recording_audio(self, buffer: list[np.ndarray], in_data, room_name, person_name, tag: str):
        buffer.append(in_data)
        if (np.concatenate(buffer, axis=0).shape[0] >= self.save_audio_len) and not self.async_save_audio[tag]:
            saving_audio = buffer.copy()
            buffer = []
            asyncio.create_task(self.save_audio(saving_audio, person_name, room_name, tag))
        return buffer

    async def save_audio(self, rec_buffer: list[np.ndarray], person_name: str, room_name: str, tag: str):
        self.async_save_audio[tag] = True
        now = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        [date, now_time] = now.split('_')
        os.makedirs(f"{self.save_audio_dir}/{date}/{room_name}", exist_ok=True)
        input_filename = f"{self.save_audio_dir}/{date}/{room_name}/{tag}_{now_time}_{person_name}.wav"
        await self.save_wav(input_filename, self.RATE, np.concatenate(rec_buffer, axis=0))
        self.async_save_audio[tag] = False
        aprint(f"Audio saved as {input_filename}")

    async def send_audio(self, ws, processed_data_int16, dtype: str, sr: int):
        try:
            if dtype == "float32":
                # int16 ->float32
                await ws.send_bytes(self.int16_to_float32(processed_data_int16).tobytes())
            else:
                if sr == 48000:
                    # upsample 16000 -> 48000
                    processed_data_int16 = np.repeat(processed_data_int16, 3)
                await ws.send_bytes(processed_data_int16.tobytes())
        except Exception as e:
            aprint(f"Couldn't send data to client: {e}")
        
    async def save_wav(self, filename: str, rate: int, data: np.ndarray):
        """비동기적으로 WAV 파일 저장. 블로킹되지 않고 빠름.
        WAV 파일 형식:
        - RIFF 헤더 (12 bytes)
        - fmt 청크 (24 bytes)
        - data 청크 헤더 (8 bytes)
        - 실제 오디오 데이터
        """
        # WAV 파일 헤더 생성
        header = bytearray()
        data_bytes = struct.pack('%dh' % len(data), *data)
        data_size = len(data_bytes)
        
        # RIFF 헤더
        header.extend(b'RIFF')
        header.extend((data_size + 36).to_bytes(4, 'little'))  # 파일 크기
        header.extend(b'WAVE')
        
        # fmt 청크
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))  # fmt 청크 크기
        header.extend((1).to_bytes(2, 'little'))   # PCM 포맷
        header.extend((1).to_bytes(2, 'little'))   # 채널 수
        header.extend(rate.to_bytes(4, 'little'))  # 샘플레이트
        header.extend((rate * 2).to_bytes(4, 'little'))  # 바이트레이트
        header.extend((2).to_bytes(2, 'little'))   # 블록 얼라인
        header.extend((16).to_bytes(2, 'little'))  # 비트 뎁스
        
        # 데이터 청크
        header.extend(b'data')
        header.extend(data_size.to_bytes(4, 'little'))
        
        # 파일 비동기 쓰기
        async with aiofiles.open(filename, 'wb') as f:
            await f.write(header)
            await f.write(data_bytes)
