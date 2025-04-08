# Micah Lingle
# DrumKitSim
# Sample reader

import os
import wave
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import feature, core
import numpy as np
import argparse as ap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
import noisereduce as nr
import matplotlib.pyplot as plt

debug = False

# Find paths to audio files
def get_paths(p = "./datasets/"):
    # File separators
    entries = os.listdir(p)
    for i in range(len(entries)):
        entries[i] = f"{p}{entries[i]}"
    paths = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        if (os.path.isdir(entry)):
            nested = os.listdir(entry)
            for j in range(len(nested)):
                nested[j] = f"{entry}/{nested[j]}"
                entries.append(nested[j])
        else:
            paths.append(entry)
        i += 1
    print("Recursively found files: ",paths)
    return paths
    
# python's wave analytics
def wave_analytics(path : str):
    wav = wave.open(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", wav.getframerate())
    print("Total samples (frames) = ", wav.getnframes())
    print("Duration = ", wav.getnframes()/wav.getframerate())

# SciKit's wave analytics
def sk_wave_analytics(path):
    rate, data = wavfile.read(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", rate)
    print("Total samples (frames) = ", data.shape)
    print(data)

def calculate_rms_noise(audio_data):
  '''
  Calculates the RMS noise of an audio signal.

  Args:
    audio_data: A NumPy array representing the audio signal.

  Returns:
    The RMS noise value.
  '''
  return np.sqrt(np.mean(audio_data**2))

# Let's do 0.104 seconds per hit--that's high enough resolution for 16th notes at 144bpm.
# Quarter note
MAX_BPM = 144
MIN_SECONDS_PER_BEAT = 60/MAX_BPM
SIXTEENTH_MIN_LENGTH_SEC = MIN_SECONDS_PER_BEAT/4

def preprocess_audio(data: np.ndarray, sr: int):
    '''
    1. Noise reduce
    2. Bandpass filter
        - Eliminate powerful high frequencies which are likely noise.
        - Save the audio at this point as we will use it later to check amplitudes.
    '''
    no_noise = nr.reduce_noise(y=data, sr=sr)
    wavfile.write("debug/reduced_noise.wav", sr, no_noise)

    sos = signal.butter(5, [200, 5000], 'bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, no_noise)
    return filtered

def segment(data: np.ndarray, sr: int, segment_length: float):
    '''
    Find peaks
    Return audio ranges in terms of samples
    '''
    noise_floor = calculate_rms_noise(data)
    # peak_indices, props = signal.find_peaks(filtered, height=noise_floor, width=sr/512)
    # print(len(peak_indices))
    # print(f'data: {data.shape}')
    # x = np.arange(0, data.shape[0], 1)
    # print(f'x: {x.size}')
    # y = np.zeros(data.shape)
    # y[peak_indices] = 1
    
    # Widths: convolution array length
    # Window size: size of window when evaluating if there is a peak or not
    # peak_indices = signal.find_peaks_cwt(data, widths=(sr/5), window_size=sr/16)
    # print(len(peak_indices))
    # x = np.arange(0, data.shape[0], 1)
    # print(f'x: {x.size}')
    # y = np.zeros(data.shape)
    # y[peak_indices] = 1

    # Convolution actually made librosa peak finding perform with more false positives on notes with long decay
    # kernel = np.linspace(1, 0, 5)
    # print(kernel)
    # smoothed = np.convolve(data, kernel, mode='same')
    
    sixteenth_min_length_samples = int(segment_length * sr)

    # Librosa peak finding
    min_thirtysecond_samples = sixteenth_min_length_samples / 2
    pre_max = min_thirtysecond_samples
    post_max = min_thirtysecond_samples
    pre_avg = min_thirtysecond_samples
    post_avg = min_thirtysecond_samples
    delta = noise_floor
    wait = min_thirtysecond_samples
    mask = False
    peak_indices = librosa.util.peak_pick(data, pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg, delta=delta, wait=wait, sparse=not mask)
    x = np.arange(0, data.shape[0], 1)
    y = np.zeros(data.shape)
    y[peak_indices] = 1

    if debug:
        print(f"Number of peaks: {len(peak_indices)}")
        plt.plot(x, y)
        plt.plot(x, data)
        plt.ylabel('amplitude')
        plt.xlabel('samples')
        plt.savefig("debug/amplitude_librosa.png")

    # # Get audio around the time of the peak
    # for peak_index in peak_indices:
    unit = int(sixteenth_min_length_samples / 10)
    ranges = []
    for peak_index in peak_indices:
        range_start = peak_index - unit
        range_end = peak_index + 9 * unit
        ranges.append((range_start, range_end))
    return peak_indices, ranges

def segments_to_ffts(data: np.ndarray, segments, sr: int, segment_length: float):
    sixteenth_min_length_samples = int(segment_length * sr)

    ffts = []
    for start, stop in segments:
        ffts.append(np.fft.fft(data[start:stop]))
    
    real_ffts = []
    for i, fft in enumerate(ffts):
        plt.clf()
        x = np.fft.fftfreq(sixteenth_min_length_samples, 1/sr)
        # Only get positive x values
        freq_lower = 0
        freq_upper = sixteenth_min_length_samples//2
        fft_x = x[freq_lower:freq_upper]
        fft_y = np.abs(fft[freq_lower:freq_upper])
        real_ffts.append(fft_y)
        if debug:
            plt.plot(fft_x, fft_y)
            plt.ylabel('amplitude')
            plt.xlabel('Hz')
            plt.savefig(f'debug/fft{i}.png')
    return real_ffts

def process_audio(data: np.ndarray, sr: int):
    '''
    1. Clean the audio
    2. Segment the audio
    3. Create FFTs
    '''

    # take absolute value to get amplitude rather than actual signal
    filtered = preprocess_audio(data, sr)
    amplitude_over_time = np.absolute(filtered)

    segment_length_sec = SIXTEENTH_MIN_LENGTH_SEC
    peaks, segments = segment(amplitude_over_time, sr, segment_length_sec)
    ffts = segments_to_ffts(filtered, segments, sr, segment_length_sec)
    return amplitude_over_time, peaks, ffts

def cluster(ffts, n):
    '''
    Cluster ffts where n represents number of clusters.
    '''
    model = GaussianMixture(n)
    labels = model.fit_predict(ffts)
    if debug:
        print(labels)

def main():

    parser = ap.ArgumentParser()
    parser.add_argument("--num-drums", type = int, default=3, help="Number of drums in the recording")
    parser.add_argument("--debug", action='store_true', help="")

    args = parser.parse_args()
    num_drums = args.num_drums
    global debug
    debug = args.debug

    # paths = get_paths()
    path = "./datasets/3sounds.wav"
    wave_analytics(path)

    print(path)
    
    data, sample_rate = librosa.load(path)

    # Use peaks and amplitudes to get amplitudes at peaks.
    # Use FFTs for clustering
    amplitude_audio, peaks, ffts = process_audio(data, sample_rate)
    cluster(ffts, num_drums)

    # normalized = no_noise / np.max(no_noise)
    # np.save("normalized", normalized)


if (__name__ == "__main__"):
    main()