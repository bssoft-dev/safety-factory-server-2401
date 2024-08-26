from speechbrain.inference.enhancement import WaveformEnhancement
from speechbrain.inference.separation import SepformerSeparation
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'
# enhancer = SepformerSeparation.from_hparams(
#     source="speechbrain/sepformer-wham-enhancement",
#     savedir='./models/sepformer_enhancement'
# )

enhancer = WaveformEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    savedir='./models/waveform_enhancement',
    run_opts={"device": device}
)

# from speechbrain.inference.enhancement import SpectralMaskEnhancement

# enhancer = SpectralMaskEnhancement.from_hparams(
#     source="speechbrain/metricgan-plus-voicebank",
#     savedir='./models/spectral_mask_enhancement'
# )