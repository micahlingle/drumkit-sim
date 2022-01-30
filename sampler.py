# Micah Lingle
# DrumKitSim
# Sample reader

import os
import wave
from scipy.io import wavfile
import librosa
import numpy as np

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
	
def aggAnalytics(paths):
	
	waveAnalytics(paths[0])		# open the first file only for debug
	skWaveAnalytics(paths[0])
	
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
	print("Sampling (frame) rate = ", rate)
	print("Total samples (frames) = ", data.shape)
	print(data)

# Use librosa to get frequency bands
def getLibr(path):
	wav, sr = librosa.load(path, sr = desiredRate)
	
def rmRmsNoise(wav, sr):
	# Compute rms amplitude
	rms = np.sqrt(np.average(np.square(wav)))
	# Based on absolute value of amplitude, zero out quiet bits
	abs = np.abs(wav)
	return np.where(abs > rms, wav, 0)
	
def main():
	paths = getPaths()
	aggAnalytics(paths)

def test():
	main()

if (__name__ == "__main__"):
	test()