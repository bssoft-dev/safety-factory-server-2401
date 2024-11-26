import torch, torchaudio
import os, numpy as np
import soundfile as sf
from denoiser.demucs import Demucs
import torchmetrics

device = 'cuda:1'

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
    print(signal.shape)
    # convert to float32
    signal = torch.from_numpy(signal).to(device=device, dtype=torch.float32) / 32768.0
    print(signal.shape)
    
    with torch.no_grad():
        enhanced_audio = model(signal[None])[0]
    
    # Convert back to numpy int16 array
    denoised_audio = (enhanced_audio * 32768).cpu().numpy().astype(np.int16).T
    # Save the denoised audio
    sf.write(output_path, denoised_audio, sample_rate)
    
    print(f"Denoised audio saved to {output_path}")

if __name__ == "__main__":
    SIG_DIR = "sounds/before_noise"
    NOISY_DIR = "sounds/after_noise"
    TARGET_DIR = "sounds/after_noise_denoised"
    
    print("Denoising audio files...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    model = Demucs(hidden = 64, sample_rate=16_000)
    state_dict = torch.load("../ai_models/speech_enhancement/dns64-a7761ff99a7d5bb6.th", 'cpu')
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    for file in os.listdir(NOISY_DIR):
        if not file.endswith(".wav"):
            continue
        denoise_wav_file(os.path.join(NOISY_DIR, file), os.path.join(TARGET_DIR, file), model)
    
    print("Calculating SDR...")
    sdr_denoised = torchmetrics.SignalDistortionRatio()
    sdr_noisy = torchmetrics.SignalDistortionRatio()
    snr = [-5, -10, -20]
    
    for speech_num in range(1, 5):
        signal, _ = torchaudio.load(os.path.join(SIG_DIR, f"speech{speech_num}.wav"))
        for video_num in range(1, 6):
            for snr_val in snr:
                noisy, _ = torchaudio.load(os.path.join(NOISY_DIR, f"speech{speech_num}_video{video_num}_{snr_val}.wav"))
                denoised, _ = torchaudio.load(os.path.join(TARGET_DIR, f"speech{speech_num}_video{video_num}_{snr_val}_denoised.wav"))
                sdr_denoised.update(signal, denoised)
                sdr_noisy.update(signal, noisy)
                print(f"file: speech{speech_num}_video{video_num}_{snr_val}, SDR noisy: {sdr_noisy.compute()}, SDR denoised: {sdr_denoised.compute()}")
    print(f"SDR denoised: {sdr_denoised.compute()}")
    print(f"SDR noisy: {sdr_noisy.compute()}")