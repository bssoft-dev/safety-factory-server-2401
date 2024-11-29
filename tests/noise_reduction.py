import torch, torchaudio
import os, numpy as np
import soundfile as sf
from denoiser.demucs import Demucs
import torchmetrics

device = 'cuda:1'
dry = 0.04

def denoise_wav_file(input_path, output_path, model):
    """
    Denoise a WAV audio file using the Denoiser package.
    
    Args:
        input_path (str): Path to the input noisy WAV file
        output_path (str): Path to save the denoised WAV file
        model (torch.nn.Module): The denoising model
    """
    # load numpy int16
    signal, sample_rate = sf.read(input_path)
    # convert to float32
    signal = torch.from_numpy(signal.astype(np.float32)).to(device=device)
    with torch.no_grad():
        enhanced_audio = model(signal[None])
        enhanced_audio = enhanced_audio * (1 - dry) + signal[None] * dry
    # Convert back to numpy int16 array
    denoised_audio = (enhanced_audio * 32768).squeeze().cpu().numpy().astype(np.int16)
    # Save the denoised audio
    sf.write(output_path, denoised_audio, sample_rate, subtype='PCM_16')
    
    print(f"Denoised audio saved to {output_path}")

if __name__ == "__main__":
    SIG_DIR = "tests/sounds/before_noise"
    NOISY_DIR = "tests/sounds/after_noise"
    NOISE_DIR = "tests/sounds/noise/trim"
    TARGET_DIR = "tests/sounds/after_noise_denoised"
    
    print("Loading model...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    model = Demucs(hidden = 64, sample_rate=16_000)
    state_dict = torch.load("ai_models/speech_enhancement/dns64-a7761ff99a7d5bb6.th", 'cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Denoising audio files...")
    for file in os.listdir(NOISY_DIR):
        if not file.endswith(".wav"):
            continue
        denoise_wav_file(os.path.join(NOISY_DIR, file), os.path.join(TARGET_DIR, file), model)
    
    print("Calculating SDR...")
    sdr_denoised = torchmetrics.audio.SignalDistortionRatio()
    sdr_noisy = torchmetrics.audio.SignalDistortionRatio()
    noisy_sdrs = []
    denoised_sdrs = []
    video_wavs = os.listdir(NOISE_DIR)
    counter = 1
    for speech_num in range(1, 6):
        signal, _ = torchaudio.load(os.path.join(SIG_DIR, f"speech{speech_num}.wav"))
        if signal.shape[0] == 2:
            signal = signal.mean(dim=0).unsqueeze(0)
        for video_wav in video_wavs:
            if video_wav.endswith(".wav"):
                video_wav = video_wav[:-4]
                noisy, _ = torchaudio.load(os.path.join(NOISY_DIR, f"speech{speech_num}_{video_wav}.wav"))
                denoised, _ = torchaudio.load(os.path.join(TARGET_DIR, f"speech{speech_num}_{video_wav}.wav"))
                noisy_sdr = sdr_noisy(signal, noisy)
                denoised_sdr = sdr_denoised(signal, denoised)
                noisy_sdrs.append(noisy_sdr)
                denoised_sdrs.append(denoised_sdr)
                print(f"{counter}. file: speech{speech_num}_{video_wav}, SDR noisy: {noisy_sdr}, SDR denoised: {denoised_sdr}")
            counter += 1
    
    print(f"SDR for noisy: {np.mean(noisy_sdrs)}, SDR for denoised: {np.mean(denoised_sdrs)}")