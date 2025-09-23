import numpy as np
import matplotlib.pyplot as plt
import librosa

# Let's do 0.104 seconds per hit--that's high enough resolution for 16th notes at 144bpm.
# Quarter note
MAX_BPM = 144
MIN_SECONDS_PER_BEAT = 60 / MAX_BPM
SIXTEENTH_MIN_LENGTH_SEC = MIN_SECONDS_PER_BEAT / 4


def segment_audio(cleaned_data, sr) -> tuple[np.ndarray, tuple[int, int], list[int]]:
    amplitude_over_time = np.absolute(cleaned_data)
    peaks, segments = segment(amplitude_over_time, sr, SIXTEENTH_MIN_LENGTH_SEC)
    velocities = get_velocities(amplitude_over_time, segments)
    return peaks, segments, velocities


def segment(
    data: np.ndarray, sr: int, segment_length: float, debug=False
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Find peaks
    Return audio ranges in terms of samples
    """
    noise_floor = calculate_rms(data)

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

    unit = int(sixteenth_min_length_samples / 10)
    ranges = []
    for peak_index in peak_indices:
        range_start = peak_index - unit
        range_end = peak_index + 9 * unit
        ranges.append((range_start, range_end))
    return peak_indices, ranges


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


def calculate_rms(audio_data):
    """
    Calculates the RMS value of an audio signal.

    Args:
      audio_data: A NumPy array representing the audio signal.

    Returns:
      The RMS noise value.
    """
    return np.sqrt(np.mean(audio_data**2))
