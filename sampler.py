# Micah Lingle
# DrumKitSim
# Sample reader

import os
import wave
from scipy.io import wavfile
import librosa
import numpy as np
import argparse as ap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

desiredRate = 22050
numDrums = 0

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
	wav = wav / np.max(wav)

	print(len(wav))
	for f in wav:
		print(f, end=",")
	print(" ")

	# Account for missed sample, interpolate for missing data
	for f in range(len(wav)):
		if (f != 0 and f != len(wav) - 1):
			if (wav[f-1] != 0
					and wav[f+1] != 0
					and wav[f] == 0):
				wav[f] = .5 * (wav[f-1] + wav[f+1])

	ft = np.fft.fft(wav)
	ft = np.real(ft)
	ft = wav
	ft = ft.reshape(-1, 1)

	km = KMeans(n_clusters=numDrums)
	y = km.fit_predict(ft)
	print("KMEANS")
	for i in y:
		print(i, end=", ")
	print("\n\n ")

	gm = GaussianMixture(n_components=numDrums)
	y = gm.fit_predict(ft)
	print("GMM")
	for i in y:
		print(i, end=", ")
	print("\n\n ")

	db = DBSCAN(eps=.04, min_samples=int(desiredRate * .1))
	y = db.fit_predict(ft)
	print("DBSCAN")
	for i in y:
		print(i, end=", ")
	print("\n\n ")


	# https://stackoverflow.com/questions/52616617/python-mean-shift-clustering-of-complex-number-numpy-array 
	# eig = np.linalg.eigh(wav)

	
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
	parser.add_argument("num_drums", type = int)

	args = parser.parse_args()
	desiredRate = args.sample_rate
	numDrums = args.num_drums + 1

	main()