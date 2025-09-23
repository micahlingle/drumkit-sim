import numpy as np
from src.audio.segmentation import SIXTEENTH_MIN_LENGTH_SEC
import matplotlib.pyplot as plt


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
        real_ffts.append(fft_y)
        if debug:
            plt.clf()
            plt.plot(fft_x, fft_y)
            plt.ylabel("amplitude")
            plt.xlabel("Hz")
            plt.savefig(f"debug/fft{i}.png")
    return real_ffts
