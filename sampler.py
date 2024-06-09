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


# Find paths to audio files
def getPaths(p = "./datasets/"):
	# Add Mac support... does it work with .wav to begin with?
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
	
def aggAnalytics(path):
	
	waveAnalytics(path)		# open the first file only for debug
	#skWaveAnalytics(path)
	
# python's wave analytics
def waveAnalytics(path : str):
	wav = wave.open(path)
	print(f"File name: {path}")
	print("Sampling (frame) rate = ", wav.getframerate())
	print("Total samples (frames) = ", wav.getnframes())
	print("Duration = ", wav.getnframes()/wav.getframerate())

# SciKit's wave analytics
def skWaveAnalytics(path):
	rate, data = wavfile.read(path)
	print(f"File name: {path}")
	print("Sampling (frame) rate = ", rate)
	print("Total samples (frames) = ", data.shape)
	print(data)

	
def main():

	parser = ap.ArgumentParser()
	parser.add_argument("--sample-rate", type = int, default=48000)
	parser.add_argument("--num-drums", type = int, default=3)

	args = parser.parse_args()
	desired_rate = args.sample_rate

	# Drums, and no drum.
	num_drums = args.num_drums + 1

	paths = getPaths()
	path = paths[0]
	waveAnalytics(path)

	print(path)
	
	data, sr = librosa.load(path, sr = desired_rate)
	np.save("data", data)

	no_noise = nr.reduce_noise(y=data, sr=sr)
	np.save("no_noise", no_noise)
	wavfile.write("reduced_noise.wav", sr, no_noise)

	# # find average threshold for percussive element decibel
	# mse_array = feature.rms(y=no_noise, frame_length=2048, hop_length=512) ** 2
	# mse_db_array = core.power_to_db(mse_array.squeeze())
	# print(mse_db_array)
	# mse_db_scalar = np.percentile(mse_db_array, .5)
	# print(mse_db_scalar)

	# Bandpass filter
	sos = signal.butter(5, [200, 5000], 'bandpass', fs=sr, output='sos')
	# sos = signal.butter(10, [20, 10000], 'bandpass', fs=sr, output='sos')
	filtered = signal.sosfilt(sos, no_noise)

	# intervals = librosa.effects.split(no_noise, top_db=mse_db_scalar)
	intervals = librosa.effects.split(filtered)
	print(intervals)

	for i, interval in enumerate(intervals):
		start, end = interval
		wavfile.write(f"bandpass_heavy{i}.wav", sr, no_noise[start:end])

	# normalized = no_noise / np.max(no_noise)
	# np.save("normalized", normalized)




if (__name__ == "__main__"):
	

	main()