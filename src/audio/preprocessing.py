import numpy as np
import noisereduce as nr
from scipy.io import wavfile
import wave
import numpy as np
import librosa


def load_audio(path: str) -> tuple[np.ndarray, int]:
    print_wave_analytics(path)
    data, sample_rate = librosa.load(path)
    return data, sample_rate


def clean_audio(data: np.ndarray, sr: int):
    no_noise = nr.reduce_noise(y=data, sr=sr)
    wavfile.write("debug/reduced_noise.wav", sr, no_noise)
    return no_noise


def print_wave_analytics(path: str):
    """Python's wave analytics"""
    wav = wave.open(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", wav.getframerate())
    print("Total samples (frames) = ", wav.getnframes())
    print("Duration = ", wav.getnframes() / wav.getframerate())


def print_sk_wave_analytics(path):
    """SciKit's wave analytics"""
    rate, data = wavfile.read(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", rate)
    print("Total samples (frames) = ", data.shape)
