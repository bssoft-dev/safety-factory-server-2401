from speechbrain.inference.enhancement import WaveformEnhancement
from speechbrain.inference.separation import SepformerSeparation
import torch
import numpy as np
# enhancer = SepformerSeparation.from_hparams(
#     source="speechbrain/sepformer-wham-enhancement",
#     savedir='./models/sepformer_enhancement'
# )



# from speechbrain.inference.enhancement import SpectralMaskEnhancement

# enhancer = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir='./models/spectral_mask_enhancement'
# )

class VoiceEnhancer:
    def __init__(self, device):
        self.device = device
        self.enhancer = WaveformEnhancement.from_hparams(
            source="speechbrain/mtl-mimic-voicebank",
            savedir='./models/waveform_enhancement',
            run_opts={"device": device}
        )

    def enhance(self, mixed_audio: np.ndarray) -> np.ndarray:
        # np.int16 -> torch.float32
        torch_audio = torch.tensor(mixed_audio/32767, dtype=torch.float32).unsqueeze(0)
        torch_audio = torch_audio.to(self.device)
        enhanced_audio = self.enhancer(torch_audio)
        enhanced_audio = enhanced_audio.cpu().squeeze(0).numpy()
        print(f"Audio enhanced")
        # np.float32 -> np.int16
        return (enhanced_audio*32767).astype(np.int16)
