import numpy as np
import os, time
import wave, struct
from scipy.signal import butter, lfilter
import torch
from utils.sys import aprint

from bs_sound_utils.voice_enhance import VoiceEnhancer, LightVoiceEnhancer, VoiceEnhancer2
from bs_sound_utils.event_classification import EventClassifier
from bs_sound_utils.stt import SttProcessor

class AudioUtils:
    def __init__(self):
        # Audio parameters
        self.FORMAT = np.int16
        self.CHANNELS = 1
        self.RATE = 16000
        self.FRAME_SIZE = 2048
        self.BUFFER_SIZE = int(self.RATE / self.FRAME_SIZE) * 1 # buffer size about 2 sec
        self.save_audio_dir = "./recordings"
        self.save_audio_sec = 60
        self.save_audio_len = self.RATE * self.save_audio_sec
        self.SYNC_INTERVAL = (self.FRAME_SIZE / self.RATE) * 4 # Sync interval for processing audio is about 0.512 sec
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.b, self.a = self.butter_lowpass(2000, self.RATE, order=10)
        # self.voice_enhancer = VoiceEnhancer(self.device)
        # self.voice_enhancer = VoiceEnhancer2(self.device)
        self.voice_enhancer = LightVoiceEnhancer()
        self.event_classifier = EventClassifier(self.device)
        self.stt = SttProcessor()
        self.input_rec_buffer: dict[int, list] = {} # client_id, 오디오 버퍼
        self.output_rec_buffer: dict[int, list] = {} # client_id, 오디오 버퍼
        
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
                return [np.zeros_like(data[0])]
        return np.array(data)
    
    def mix_audio_to_torch(self, data: list[np.ndarray], exclude_idx: int | None = None) -> torch.Tensor:
        data_array = self.exclude_client_audio(data, exclude_idx)
        return torch.mean(torch.from_numpy(data_array).to(device=self.device, dtype=torch.float32) / 32768.0, dim=0)
    
    def mix_audio(self, data: list[np.ndarray], exclude_idx: int | None = None) -> np.ndarray:
        dtype = data[0].dtype
        data_array = self.exclude_client_audio(data, exclude_idx)

        # 오디오 믹싱 개선
        mixed = np.mean(data_array, axis=0)
        
        # 피크 정규화 (클리핑 방지)
        peak = np.max(np.abs(mixed))
        if peak > 0:
            # 32767의 90%를 목표로 정규화 (헤드룸 확보)
            target_peak = 29490  # 32767 * 0.9
            scale_factor = target_peak / peak
            mixed = mixed * min(scale_factor, 1.0)  # 볼륨이 작을 때는 증폭하지 않음
        
        # RMS 레벨 조정 (전체적인 볼륨 균일화)
        rms = np.sqrt(np.mean(mixed**2))
        target_rms = 3277  # 32767 * 0.1 (적절한 RMS 레벨)
        if rms > 0:
            rms_scale = target_rms / rms
            mixed = mixed * min(rms_scale, 2.0)  # 최대 2배까지만 증폭
        
        # 소프트 클리핑으로 급격한 피크 방지
        mixed = self.soft_clip(mixed / 32767.0) * 32767.0
        
        return mixed.astype(dtype)
    
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
            #print("Voice enhance - denoising")
            enhanced_audio = self.voice_enhancer.denoise(in_data, client_id)
            # enhanced_audio = self.voice_enhancer.denoise(in_data)
            # return self.torch_float32_to_int16(enhanced_audio)
            return enhanced_audio
        except Exception as e:
            print(f"Error in voice enhance: {e}")
            return in_data

    def recording_audio(self, buffer: list[np.ndarray], in_data, room_name, person_name, tag: str):
        buffer.append(in_data)
        if np.concatenate(buffer, axis=0).shape[0] >= self.save_audio_len:
            self.save_audio(buffer, person_name, room_name, tag)
            buffer = []
        return buffer

    def save_audio(self, rec_buffer: list[np.ndarray], person_name: str, room_name: str, tag: str):
        now = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        [date, now_time] = now.split('_')
        os.makedirs(f"{self.save_audio_dir}/{date}/{room_name}", exist_ok=True)
        input_filename = f"{self.save_audio_dir}/{date}/{room_name}/{tag}_{now_time}_{person_name}.wav"
        self.save_wav(input_filename, self.RATE, np.concatenate(rec_buffer, axis=0))
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
        
    def save_wav(self, filename, rate, data):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(rate)
            wf.writeframes(struct.pack('%dh' % len(data), *data))
