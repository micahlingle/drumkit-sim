# Micah Lingle
# DrumKitSim
# Sample reader

import os
import wave
from scipy.io import wavfile
import librosa
import numpy as np
import argparse as ap

desiredRate = 22050

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

# Use librosa to get frequency bands
def getLibr(path):
	wav, sr = librosa.load(path, sr = desiredRate)
	wav = rmRmsNoise(wav)
	print(len(wav))
	for f in wav:
		print(f)

	
def rmRmsNoise(wav):
	# Compute rms amplitude
	rms = np.sqrt(np.average(np.square(wav))) 
	print(f"rms is {rms}")
	# Based on absolute value of amplitude, zero out quiet bits
	abs = np.abs(wav)
	return np.where(abs > rms, wav, 0)
	
def main():
	paths = getPaths()
	waveAnalytics(paths[0])

	print(paths[0])
	getLibr(paths[0])

if (__name__ == "__main__"):
	parser = ap.ArgumentParser()
	parser.add_argument("sample_rate", type = int)
	args = parser.parse_args()
	print(args.sample_rate)
	desiredRate = args.sample_rate

	main()