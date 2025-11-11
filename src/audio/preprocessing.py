import numpy as np
import noisereduce as nr
from scipy import signal
from scipy.io import wavfile
import wave
import numpy as np
import librosa


def load_audio(path: str) -> tuple[np.ndarray, int]:
    print_wave_analytics(path)
    data, sample_rate = librosa.load(path)
    return data, sample_rate


def clean_audio(data: np.ndarray, sr: int):
    """
    1. Noise reduce
    2. Bandpass filter
        - Eliminate powerful high frequencies which are likely noise.
        - Save the audio at this point as we will use it later to check amplitudes.
    """
    no_noise = nr.reduce_noise(y=data, sr=sr)
    wavfile.write("debug/reduced_noise.wav", sr, no_noise)

    sos = signal.butter(5, [200, 5000], "bandpass", fs=sr, output="sos")
    filtered = signal.sosfilt(sos, no_noise)
    return filtered

def boost_low_end(data: np.ndarray,cutoff_freq: int = 100, amplification_factor: float = 3):
    # Define the cutoff frequency and amplification factor

    # Create a mask for low frequencies
    # Use abs(xf) to cover both positive and negative frequency components (for real signal symmetry)
    low_freq_mask = np.abs(data) < cutoff_freq

    # Amplify the selected frequencies
    yf_amplified = data.copy()
    yf_amplified[low_freq_mask] *= amplification_factor
    return yf_amplified


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
