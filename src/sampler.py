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
import soundfile as sf
import shutil

from midi import map_drums_to_midi, write_midi

debug = False


# Find paths to audio files
def get_paths(p="./datasets/"):
    # File separators
    entries = os.listdir(p)
    for i in range(len(entries)):
        entries[i] = f"{p}{entries[i]}"
    paths = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        if os.path.isdir(entry):
            nested = os.listdir(entry)
            for j in range(len(nested)):
                nested[j] = f"{entry}/{nested[j]}"
                entries.append(nested[j])
        else:
            paths.append(entry)
        i += 1
    print("Recursively found files: ", paths)
    return paths


# python's wave analytics
def wave_analytics(path: str):
    wav = wave.open(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", wav.getframerate())
    print("Total samples (frames) = ", wav.getnframes())
    print("Duration = ", wav.getnframes() / wav.getframerate())


# SciKit's wave analytics
def sk_wave_analytics(path):
    rate, data = wavfile.read(path)
    print(f"File name: {path}")
    print("Sampling (frame) rate = ", rate)
    print("Total samples (frames) = ", data.shape)
    print(data)


def calculate_rms(audio_data):
    """
    Calculates the RMS value of an audio signal.

    Args:
      audio_data: A NumPy array representing the audio signal.

    Returns:
      The RMS noise value.
    """
    return np.sqrt(np.mean(audio_data**2))


# Let's do 0.104 seconds per hit--that's high enough resolution for 16th notes at 144bpm.
# Quarter note
MAX_BPM = 144
MIN_SECONDS_PER_BEAT = 60 / MAX_BPM
SIXTEENTH_MIN_LENGTH_SEC = MIN_SECONDS_PER_BEAT / 4


def preprocess_audio(data: np.ndarray, sr: int):
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


def segment(data: np.ndarray, sr: int, segment_length: float):
    """
    Find peaks
    Return audio ranges in terms of samples
    """
    noise_floor = calculate_rms(data)
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
    peak_indices = librosa.util.peak_pick(
        data,
        pre_max=pre_max,
        post_max=post_max,
        pre_avg=pre_avg,
        post_avg=post_avg,
        delta=delta,
        wait=wait,
        sparse=not mask,
    )
    x = np.arange(0, data.shape[0], 1)
    y = np.zeros(data.shape)
    y[peak_indices] = 1

    if debug:
        print(f"Number of peaks: {len(peak_indices)}")
        plt.plot(x, y)
        plt.plot(x, data)
        plt.ylabel("amplitude")
        plt.xlabel("samples")
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
        x = np.fft.fftfreq(sixteenth_min_length_samples, 1 / sr)
        # Only get positive x values
        freq_lower = 0
        freq_upper = sixteenth_min_length_samples // 2
        fft_x = x[freq_lower:freq_upper]
        fft_y = np.abs(fft[freq_lower:freq_upper])
        real_ffts.append(fft_y)
        if debug:
            plt.clf()
            plt.plot(fft_x, fft_y)
            plt.ylabel("amplitude")
            plt.xlabel("Hz")
            plt.savefig(f"debug/fft{i}.png")
    return real_ffts


def process_audio(
    data: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int, int], list[np.ndarray]]:
    """
    1. Clean the audio
    2. Segment the audio
    3. Create FFTs

    Inputs
    - data: 1-dimensional audio file
    - sr: sample rate

    Outputs
    - filtered: cleaned audio after preprocessing
    - amplitude_over_time: absolute value of filtered audio
    - peaks: peak indices of amplitude_over_time
    - segments: indices (in samples) surrounding the peaks
    - ffts: y-values for amplitude of each frequency. The number of samples
    """

    # take absolute value to get amplitude rather than actual signal
    filtered = preprocess_audio(data, sr)
    amplitude_over_time = np.absolute(filtered)

    segment_length_sec = SIXTEENTH_MIN_LENGTH_SEC
    peaks, segments = segment(amplitude_over_time, sr, segment_length_sec)
    ffts = segments_to_ffts(filtered, segments, sr, segment_length_sec)
    return filtered, amplitude_over_time, peaks, segments, ffts


def cluster(ffts, n):
    """
    Cluster ffts where n represents number of clusters.
    """
    model = GaussianMixture(n)
    labels = model.fit_predict(ffts)
    if debug:
        print(labels)
    return labels


def save_all_audio_by_label(
    path, num_drums, segments, labels, peaks, filtered, sample_rate
):
    """
    Save audio to temp folder for playback/interaction
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for i in range(num_drums):
        os.mkdir(f"{path}/{i}")
    for segment, label, peak in zip(segments, labels, peaks):
        start, stop = segment
        sf.write(f"{path}/{label}/{peak}.wav", filtered[start:stop], sample_rate)


def get_velocities(amplitude_audio, segments):
    """
    1. Calculate RMS values for each segment
    2. Normalize

    Returns a list of velocity integers for each inputted segment.
    Note MIDI velocity is 0-127
    """
    amplitudes = np.zeros((len(segments)))
    for i, segment in enumerate(segments):
        start, stop = segment
        rms = calculate_rms(amplitude_audio[start:stop])
        amplitudes[i] = rms

    # Normalize and scale wrt max MIDI velocity
    normalized = amplitudes / np.max(amplitudes)
    normalized *= 127
    velocity_integers = normalized.astype(np.uint8)
    return velocity_integers.tolist()


def main():

    parser = ap.ArgumentParser()
    parser.add_argument("bpm", default=60, help="Beats per minute")
    parser.add_argument(
        "time_signature",
        default=(4, 4),
        help="Time signature. Only denominator really matters",
    )
    parser.add_argument(
        "num_drums", type=int, default=3, help="Number of drums in the recording"
    )
    parser.add_argument("--debug", action="store_true", help="")

    args = parser.parse_args()
    bpm = int(args.bpm)
    times = args.time_signature.split("/")
    time_signature = (int(times[0]), int(times[1]))
    num_drums = int(args.num_drums)

    global debug
    debug = args.debug
    if debug:
        os.makedirs("debug", exist_ok=True)

    # paths = get_paths()
    path = "./datasets/3sounds.wav"
    wave_analytics(path)

    data, sample_rate = librosa.load(path)
    filtered, amplitude_audio, peaks, segments, ffts = process_audio(data, sample_rate)
    print(peaks)
    labels = cluster(ffts, num_drums)
    tmp_path = "temp"
    save_all_audio_by_label(
        tmp_path, num_drums, segments, labels, peaks, filtered, sample_rate
    )
    drum_to_midi_map = map_drums_to_midi(num_drums, tmp_path)
    # Use peaks and amplitudes to get amplitudes at peaks.
    velocities = get_velocities(amplitude_audio, segments)
    write_midi(
        peaks.tolist(),
        labels,
        velocities,
        drum_to_midi_map,
        sample_rate,
        bpm,
        time_signature,
    )


if __name__ == "__main__":
    main()
