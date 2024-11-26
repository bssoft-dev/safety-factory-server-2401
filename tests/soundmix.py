import numpy as np
import soundfile as sf
import os

skip_sec = 4
sr = 16000

def mix_audio_with_snr(signal, noise, snr):
    # if the signal is shorter than the noise, cut the noise
    if len(signal) < len(noise):
        noise = noise[sr*skip_sec:sr*skip_sec+len(signal)]
    
    # get the initial energy for reference
    signal_energy = np.mean(signal**2)
    noise_energy = np.mean(noise**2)
    # calculates the gain to be applied to the noise 
    # to achieve the given SNR
    g = np.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
    
    # Assumes signal and noise to be decorrelated
    # and calculate (a, b) such that energy of 
    # a*signal + b*noise matches the energy of the input signal
    a = np.sqrt(1 / (1 + g**2))
    b = np.sqrt(g**2 / (1 + g**2))
    print(g, a, b)
    # mix the signals
    return a * signal + b * noise


if __name__ == "__main__":
    SIG_DIR = "sounds/before_noise"
    NOISE_DIR = "sounds/noise"
    TARGET_DIR = "sounds/after_noise"
    os.makedirs(TARGET_DIR, exist_ok=True)
    SNR = [-5, -10, -20]
    for snr in SNR:
        for sig_file in os.listdir(SIG_DIR):
            if not sig_file.endswith(".wav"):
                continue
            for noise_file in os.listdir(NOISE_DIR):
                if not noise_file.endswith(".wav"):
                    continue
                signal, sr = sf.read(os.path.join(SIG_DIR, sig_file))
                noise, sr = sf.read(os.path.join(NOISE_DIR, noise_file))
                mixed = mix_audio_with_snr(signal, noise, snr)
                # reshape if the signal is stereo
                if mixed.shape[1] > 1:
                    mixed = mixed.mean(axis=1).reshape(-1, 1)
                sf.write(os.path.join(TARGET_DIR, f"{sig_file.replace('.wav', '')}_{noise_file.replace('.wav', '')}_{snr}.wav"), mixed, sr)
