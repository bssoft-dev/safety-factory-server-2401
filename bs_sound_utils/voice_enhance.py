import torch
from denoiser.demucs import Demucs
from denoiser.demucs import DemucsStreamer


class VoiceEnhancer:
    def __init__(self, device):
        self.device = device
        self.dry = 0.04
        self.num_frames = 1
        self.model = Demucs(hidden = 64, sample_rate=16_000)
        state_dict = torch.load("ai_models/speech_enhancement/dns64-a7761ff99a7d5bb6.th", 'cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        self.streamer = DemucsStreamer(self.model, dry=self.dry, num_frames=self.num_frames)

    def denoise(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            denoised = self.streamer.feed(audio_tensor[None])[0]
        if not denoised.numel():
            print("denoised is empty")
            return torch.zeros(audio_tensor.shape)
        return denoised