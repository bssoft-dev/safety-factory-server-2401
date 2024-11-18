import numpy as np
import wave
import struct
from scipy.signal import butter, lfilter
import torch

from bs_sound_utils.voice_enhance import VoiceEnhancer
from bs_sound_utils.event_classification import EventClassifier
from bs_sound_utils.stt import SttProcessor

class AudioUtils:
    def __init__(self):
        # Audio parameters
        self.FORMAT = np.int16
        self.CHANNELS = 1
        self.RATE = 16000
        self.FRAME_SIZE = 2048
        self.BUFFER_SIZE = int(self.RATE / self.FRAME_SIZE) * 1 # buffer size about 1 sec
        self.SYNC_INTERVAL = (self.FRAME_SIZE / self.RATE) * 4 # Sync interval for processing audio is about 0.512 sec
        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.b, self.a = self.butter_lowpass(2000, self.RATE, order=10)
        self.voice_enhancer = VoiceEnhancer(device)
        self.event_classifier = EventClassifier()
        self.stt = SttProcessor()
        
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

    def soft_clip(self, x, threshold=0.9):
        return np.tanh(x / threshold) * threshold

    def int16_to_float32(self, data: np.ndarray) -> np.ndarray:
        if np.max(np.abs(data)) > 32768:
            raise ValueError("Data has values above 32768")
        return (data / 32768.0).astype("float32")
    
    def float32_to_int16(self, data: np.ndarray) -> np.ndarray:
        if np.max(data) > 1:
            data = data / np.max(np.abs(data))
        return np.array(data * 32767).astype("int16")
    
    def mix_audio(self, data: list[np.ndarray], exclude_idx: int = None) -> np.ndarray:
        dtype = data[0].dtype
        selected_audio = data.copy()
        if exclude_idx != None:
            if len(selected_audio) > 1:
                # Exclude client itself's audio from mixing
                del selected_audio[exclude_idx]
            else:
                # If there is only one client, mute its audio
                selected_audio = [np.zeros_like(selected_audio[0])]
        # Audio mixing: use average
        return np.mean(np.array(selected_audio), axis=0, dtype=dtype)
    
    async def classify_audio(self, send_data: bytes, room_name: str):
        await self.event_classifier.classify_audio(send_data, room_name)
    
    async def stt_audio(self, send_data: bytes, room_name: str):
        await self.stt.send_audio(send_data, room_name)

    def voice_enhance(self, audio_data):
        return self.voice_enhancer.enhance(audio_data)
    
    async def send_audio(self, ws, client_id, processed_data_int16):
        try:
            if self.client_info[client_id]["dtype"] == "float32":
                # int16 ->float32
                await ws.send_bytes(self.int16_to_float32(processed_data_int16).tobytes())
            else:
                if self.client_info[client_id]["sr"] == 48000:
                    # upsample 16000 -> 48000
                    processed_data_int16 = np.repeat(processed_data_int16, 3)
                await ws.send_bytes(processed_data_int16.tobytes())
        except:
            print("Couldn't send data to client")
            pass  # Ignore disconnected clients
    
    def save_wav(self, filename, rate, data):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(rate)
            wf.writeframes(struct.pack('%dh' % len(data), *data))

    def save_recordings(self, input_rec_buffer, output_rec_buffer):
        input_filename = "input.wav"
        output_filename = "output.wav"
        fs_target = 16000
        # 입력 오디오 저장
        self.save_wav(input_filename, fs_target, np.concatenate(input_rec_buffer))
        print(f"Input audio saved as {input_filename}")
        # 출력 오디오 저장
        self.save_wav(output_filename, fs_target, np.concatenate(output_rec_buffer))
        print(f"Output audio saved as {output_filename}")