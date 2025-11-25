import numpy as np
from src.audio.segmentation import SIXTEENTH_MIN_LENGTH_SEC, calculate_rms
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import librosa


def segments_to_features(
    data: np.ndarray,
    segments,
    sr: int,
    segment_length=SIXTEENTH_MIN_LENGTH_SEC,
    debug: bool = False,
) -> list[np.ndarray]:
    """
    Convert audio segments to features
    """
    # ffts = segments_to_ffts(data, segments, sr, segment_length, debug)
    # features = extract_peaks(ffts, n=8)
    # return features

    segments_to_ffts(data, segments, sr, segment_length, debug)
    feats = extract_librosa_features(data, segments, sr, debug)
    std = standardize(feats, debug)
    pca_feats = pca(std, 2)
    return pca_feats


def pca(features: np.ndarray, n: int) -> list[np.ndarray]:
    """
    Perform PCA on features
    """
    pca = PCA(n_components=n)
    return pca.fit_transform(features)


def standardize(features: list[np.ndarray], debug=False) -> np.ndarray:
    """
    Standardize using scikit StandardScaler
    """
    # Convert features to 2D array
    features = np.array(features)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(features)
    if debug:
        print(f"std feats: {standardized}")
    return standardized


def extract_librosa_features(
    data: np.ndarray,
    segments,
    sr: int,
    debug: bool = False,
) -> list[np.ndarray]:
    feats = []
    for i, bounds in enumerate(segments):
        start, stop = bounds
        audio_segment = data[start:stop]
        segment_length = stop - start

        # Add MFCCs for better timbral characterization
        mfccs = librosa.feature.mfcc(
            y=audio_segment, sr=sr, n_mfcc=3, n_fft=segment_length, center=False
        )

        centroids = librosa.feature.spectral_centroid(
            y=audio_segment, n_fft=segment_length, center=False
        )
        contrast = librosa.feature.spectral_contrast(
            y=audio_segment,
            n_fft=segment_length,
            win_length=segment_length,
            center=False,
        )
        zcross_rate = librosa.feature.zero_crossing_rate(
            y=audio_segment, frame_length=segment_length, center=False
        )
        polynomial_coefficients = librosa.feature.poly_features(
            y=audio_segment, n_fft=segment_length, center=False
        )

        if debug:
            print(f"segment {i}:")
            print(f"\tmfccs: {mfccs}")
            print(f"\tcentroids: {centroids}")
            print(f"\tcontrast: {contrast}")
            print(f"\tzcross_rate: {zcross_rate}")
            print(f"\tpolynomial_coefficients: {polynomial_coefficients}")

        # Combine all features
        features = np.concatenate(
            [
                mfccs.flatten(),
                centroids.flatten(),
                contrast.flatten(),
                zcross_rate.flatten(),
                polynomial_coefficients[0],
            ]
        )
        feats.append(features)
    return feats


def extract_peaks(ffts: list[np.ndarray], n: int) -> list[np.ndarray]:
    """
    Convert each fft into a feature vector, where each feature vector contains the frequency of a peak and the amplitude.
    This simplifies the entire FFT (thousands of points) into the most important component frequencies which characterize
    the sound.

    Inputs
    - ffts, 2d array of [frequencies, amplitudes]
    - n: number of peaks to find for each FFT
    """
    vectors = []
    for fft in ffts:
        # Find mean height of FFT y-values and use that as a baseline to find peaks.
        rms = calculate_rms(fft[:, 1])
        peaks, props = find_peaks(fft[:, 1], height=rms)
        top_n_peaks_indices = np.argsort(props["peak_heights"])[-n:]

        # Create feature vector of peak frequencies paired with amplitudes
        feature_vector = np.stack(
            [
                fft[peaks[top_n_peaks_indices], 0],
                props["peak_heights"][top_n_peaks_indices],
            ],
            axis=-1,
        ).flatten()
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

        fft_2d = np.stack([fft_x, fft_y], axis=-1)
        real_ffts.append(fft_2d)
        if debug:
            plt.clf()
            plt.plot(fft_x, fft_y)
            plt.ylabel("amplitude")
            plt.xlabel("Hz")
            plt.savefig(f"debug/fft{i}.png")
    return real_ffts
