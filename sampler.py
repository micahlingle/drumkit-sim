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


def segment_audio(data: np.ndarray, sr: int):
    '''
    Audio segmentation pipeline.
    - The open question is: how can we split up two segments which are < segment_duration
      apart from each other? We could just lower the segment duration to an infinitesimal number.
    - 2 ideas on how to do run the segmentation
       1. Compress and RMS split
       2. Find peaks

    # Way 1
    1. Noise reduce
    2. Bandpass filter
        - Eliminate powerful high frequencies which are likely noise.
        - Save the audio at this point as we will use it later to check amplitudes.
    3. Compress
        - Because Librosa's split function not only 
    4. Split
        - Librosa has a topdb parameter which acts as the noise threshold.
          It effectively gates everything below this threshold. This is helpful
        - We want to 

    # Way 2
    1. Noise reduce
    2. Bandpass filter
    3. Find peaks
    '''

    no_noise = nr.reduce_noise(y=data, sr=sr)
    np.save("no_noise", no_noise)
    wavfile.write("temp/reduced_noise.wav", sr, no_noise)

    # Bandpass filter to remove high frequency noise
    sos = signal.butter(5, [200, 5000], 'bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, no_noise)

    # # Librosa automatically accounts for RMS/MSE
    # intervals = librosa.effects.split(filtered)
    # print(intervals)

    # for i, interval in enumerate(intervals):
    #     start, end = interval
    #     wavfile.write(f"temp/bandpass_heavy{i}.wav", sr, no_noise[start:end])

    # take absolute value
    amplitude_over_time = np.absolute(filtered)
    # noise_floor = calculate_rms_noise(amplitude_over_time)
    # peak_indices, props = signal.find_peaks(filtered, height=noise_floor, width=sr/512)
    # print(len(peak_indices))
    # print(f'data: {data.shape}')
    # x = np.arange(0, data.shape[0], 1)
    # print(f'x: {x.size}')
    # y = np.zeros(data.shape)
    # y[peak_indices] = 1
    
    # Widths: convolution array length
    # Window size: size of window when evaluating if there is a peak or not
    peak_indices = signal.find_peaks_cwt(amplitude_over_time, widths=(sr/5), window_size=sr/16)
    print(len(peak_indices))
    x = np.arange(0, data.shape[0], 1)
    print(f'x: {x.size}')
    y = np.zeros(data.shape)
    y[peak_indices] = 1

    plt.plot(x, y)
    plt.ylabel('amplitude')
    plt.xlabel('samples')
    plt.savefig("sample_plot.png")

def main():

    parser = ap.ArgumentParser()
    parser.add_argument("--sample-rate", type = int, default=48000)
    parser.add_argument("--num-drums", type = int, default=3)
    # Default of 200ms because average is 242 based on my testing.
    parser.add_argument("--segment-duration", type = int, default=200)

    args = parser.parse_args()
    desired_rate = args.sample_rate
    num_drums = args.num_drums


    # paths = get_paths()
    path = "./datasets/snaps.wav"
    wave_analytics(path)

    print(path)
    
    data, sr = librosa.load(path, sr = desired_rate)
    np.save("data", data)

    segment_audio(data, sr)

    # normalized = no_noise / np.max(no_noise)
    # np.save("normalized", normalized)



if (__name__ == "__main__"):
    main()