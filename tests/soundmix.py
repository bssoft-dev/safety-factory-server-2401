import numpy as np
import soundfile as sf
import os
from pydub import AudioSegment

sr = 16000

trim_cases = {
    1: [[4, 10], [60, 66]],
    2: [[0, 6], [6, 12]],
    3: [[0, 6]],
    4: [[4, 10]],
    5: [[0, 6], [16, 22]],
    6: [[0, 6], [6, 12]]
}

SIG_DIR = "tests/sounds/before_noise"
NOISE_DIR = "tests/sounds/noise"
TRIM_NOISE_DIR = "tests/sounds/noise/noise_clip"
TARGET_DIR = "tests/sounds/after_noise"

def trim_audio_with_timestamps(audio_path: str, output_path: str, timestamps: list[int]):
    audio = AudioSegment.from_file(audio_path)
    audio = audio[timestamps[0]*1000:timestamps[1]*1000]
    audio.set_channels(1)
    audio.export(output_path, format="wav")

def mix_audio_with_snr(signal: np.ndarray, noise: np.ndarray, snr: float):
    # if the signal is shorter than the noise, cut the noise
    if len(signal) < len(noise):
        noise = noise[:len(signal)]
    
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

def trim_noise():
    os.makedirs(TRIM_NOISE_DIR, exist_ok=True)
    for case_id, timestamps in trim_cases.items():
        print(timestamps)
        for atime in timestamps:
            start = atime[0]
            end = atime[1]
            trim_audio_with_timestamps(os.path.join(NOISE_DIR, f"video{case_id}.wav"), os.path.join(TRIM_NOISE_DIR, f"video{case_id}_{start}_{end}.wav"), [start, end])

def mix_audio():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for sig_file in os.listdir(SIG_DIR):
        if not sig_file.endswith(".wav"):
            continue
        for noise_file in os.listdir(TRIM_NOISE_DIR):
            if not noise_file.endswith(".wav"):
                continue
            signal, sr = sf.read(os.path.join(SIG_DIR, sig_file))
            noise, sr = sf.read(os.path.join(TRIM_NOISE_DIR, noise_file))
            mixed = mix_audio_with_snr(signal, noise, -3)
            # reshape if the signal is stereo
            if mixed.shape[1] > 1:
                mixed = mixed.mean(axis=1).reshape(-1, 1)
            sf.write(os.path.join(TARGET_DIR, f"{sig_file.replace('.wav', '')}_{noise_file.replace('.wav', '')}.wav"), mixed, sr)

if __name__ == "__main__":
    # trim_noise() # 기존 버전은 파일명을 비디오 번호와 타임스탬프로 표시했으나 현재는 파일명을 임의로 바꿨음
    mix_audio()