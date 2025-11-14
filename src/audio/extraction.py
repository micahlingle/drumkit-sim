import numpy as np
from src.audio.segmentation import SIXTEENTH_MIN_LENGTH_SEC, calculate_rms
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def segments_to_features(
    data: np.ndarray,
    segments,
    sr: int,
    segment_length=SIXTEENTH_MIN_LENGTH_SEC,
    debug: bool = False,
):
    '''
    Convert audio segments to features
    '''
    ffts = segments_to_ffts(data, segments, sr, segment_length, debug)
    features = extract_peaks(ffts, n=8)
    return features


def extract_peaks(
    ffts: list[np.ndarray],
    n: int
):
    '''
    Convert each fft into a feature vector, where each feature vector contains the frequency of a peak and the amplitude.
    This simplifies the entire FFT (thousands of points) into the most important component frequencies which characterize
    the sound.

    Inputs
    - ffts, 2d array of [frequencies, amplitudes]
    - n: number of peaks to find for each FFT
    '''
    vectors = []
    for fft in ffts:
        # Find mean height of FFT y-values and use that as a baseline to find peaks.
        rms = calculate_rms(fft[:, 1])
        peaks, props = find_peaks(fft[:, 1], height=rms)
        top_n_peaks_indices = np.argsort(props["peak_heights"])[-n:]

        # Create feature vector of peak frequencies paired with amplitudes
        feature_vector = np.stack([fft[peaks[top_n_peaks_indices], 0], props["peak_heights"][top_n_peaks_indices]], axis=-1).flatten()
        vectors.append(feature_vector)
    return vectors

def segments_to_ffts(
    data: np.ndarray,
    segments,
    sr: int,
    segment_length=SIXTEENTH_MIN_LENGTH_SEC,
    debug: bool = False,
):
    sixteenth_min_length_samples = int(segment_length * sr)

    ffts = []
    for start, stop in segments:
        ffts.append(np.fft.fft(data[start:stop]))

    real_ffts = []
    for i, fft in enumerate(ffts):
        x = np.fft.fftfreq(sixteenth_min_length_samples, 1 / sr)
        # Only get positive x values
        freq_lower = 0
        freq_upper = sixteenth_min_length_samples // 2
        fft_x = x[freq_lower:freq_upper]
        fft_y = np.abs(fft[freq_lower:freq_upper])
        
        # TODO: Multiply by some function to scale up lower frequencies and soften higher

        fft_2d = np.stack([fft_x, fft_y], axis=-1)
        real_ffts.append(fft_2d)
        if debug:
            plt.clf()
            plt.plot(fft_x, fft_y)
            plt.ylabel("amplitude")
            plt.xlabel("Hz")
            plt.savefig(f"debug/fft{i}.png")
    return real_ffts
