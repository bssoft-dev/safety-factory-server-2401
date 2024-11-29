import torch
from denoiser.demucs import Demucs
from denoiser.demucs import DemucsStreamer
import numpy as np
import librosa

class VoiceEnhancer:
    def __init__(self, device):
        self.device = device
        self.dry = 0.04
        self.num_frames = 10
        self.model = Demucs(hidden = 48, sample_rate=16_000)
        state_dict = torch.load("ai_models/speech_enhancement/dns48-11decc9d8e3f0998.th", 'cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        self.streamers : dict[int, DemucsStreamer] = {}
    
    def add_streamer(self, client_id: int):
        self.streamers[client_id] = DemucsStreamer(self.model, dry=self.dry, num_frames=self.num_frames)
        
    def remove_streamer(self, client_id: int):
        del self.streamers[client_id]
        
    def denoise(self, audio_tensor: torch.Tensor, client_id: int) -> torch.Tensor:
        with torch.no_grad():
            denoised = self.streamers[client_id].feed(audio_tensor[None])[0]
        if not denoised.numel():
            print("denoised is empty")
            return torch.zeros(audio_tensor.shape)
        return denoised

import torch
import numpy as np
from speechbrain.inference.separation import SepformerSeparation

class VoiceEnhancer2:
    def __init__(self, device):
        self.device = device
        # 미리 학습된 Speechbrain SE 모델 로드
        self.enhancer = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-dns4-16k-enhancement",
            savedir="pretrained_models/sepformer-dns4-16k-enhancement",
            run_opts={"device": self.device}
        )
        self.sample_rate = 16000
        self.frame_size = 16000  # 1초 단위로 처리
        self.hop_length = 4000   # 0.5초 오버랩
        self.streamers = {}
        
    # def add_streamer(self, client_id: int):
    #     self.streamers[client_id] = {
    #         'buffer': np.array([], dtype=np.int16)
    #     }
        
    # def remove_streamer(self, client_id: int):
    #     if client_id in self.streamers:
    #         del self.streamers[client_id]
            
    def denoise(self, mixed_audio: np.ndarray) -> np.ndarray:
        # np.int16 -> torch.float32
        torch_audio = torch.tensor(mixed_audio/32767, dtype=torch.float32).unsqueeze(0)
        torch_audio = torch_audio.to(self.device)
        enhanced_audio = self.enhancer(torch_audio)
        enhanced_audio = enhanced_audio.cpu().squeeze(0).numpy()
        print(f"Audio enhanced")
        # np.float32 -> np.int16
        return (enhanced_audio*32767).astype(np.int16)
    
class LightVoiceEnhancer:
    def __init__(self):
        self.sample_rate = 16000  # 16kHz 고정
        self.frame_size = 480     # 30ms 프레임
        self.hop_length = 160     # 10ms 홉
        self.streamers = {}
        
    def add_streamer(self, client_id: int):
        """새로운 스트리머 추가"""
        self.streamers[client_id] = {
            'buffer': np.array([], dtype=np.int16)
        }
        
    def remove_streamer(self, client_id: int):
        """스트리머 제거"""
        if client_id in self.streamers:
            del self.streamers[client_id]
            
    def _spectral_gating(self, frames):
        """스펙트럼 기반 노이즈 게이팅"""
        # STFT 변환
        D = librosa.stft(frames.astype(np.float32) / 32768.0, 
                        n_fft=512, 
                        hop_length=self.hop_length,
                        win_length=self.frame_size)
        
        # 파워 스펙트럼 계산
        S = np.abs(D) ** 2
        
        # 노이즈 추정 (간단한 통계 기반)
        noise_thresh = np.mean(S, axis=1) * 1.5
        
        # 게이팅 마스크 생성
        mask = (S.T > noise_thresh).T
        
        # 마스크 적용
        D_denoised = D * mask
        
        # 역변환
        y_denoised = librosa.istft(D_denoised, 
                                 hop_length=self.hop_length,
                                 win_length=self.frame_size)
        
        return (y_denoised * 32768.0).astype(np.int16)
        
    def denoise(self, audio: np.ndarray, client_id: int) -> np.ndarray:
        """오디오 프레임 노이즈 제거"""
        # 버퍼에 새 데이터 추가
        self.streamers[client_id]['buffer'] = np.concatenate([
            self.streamers[client_id]['buffer'], 
            audio
        ])
        
        buffer = self.streamers[client_id]['buffer']
        
        # 프레임 크기만큼 데이터가 쌓였을 때 처리
        if len(buffer) >= self.frame_size:
            # 노이즈 제거 처리
            denoised = self._spectral_gating(buffer[:self.frame_size])
            
            # 버퍼 업데이트 (홉 길이만큼 이동)
            self.streamers[client_id]['buffer'] = buffer[self.hop_length:]
            
            return denoised[:self.hop_length]
            
        # 버퍼가 부족하면 빈 배열 반환
        return np.array([], dtype=np.int16)