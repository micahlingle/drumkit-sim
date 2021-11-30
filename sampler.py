# Micah Lingle
# DrumKitSim
# Sample reader

import os
import wave
from scipy.io import wavfile




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
	#print("Recursively found files: ",paths)
	return paths
	
def openFile(paths):
	
	wav = wave.open(paths[0])
	# python's wave analytics
	print(f"File name: {paths[0]}")
	print("Sampling (frame) rate = ", wav.getframerate())
	print("Total samples (frames) = ", wav.getnframes())
	print("Duration = ", wav.getnframes()/wav.getframerate())
	# scikit's wavfile analytics
	rate, data = wavfile.read(paths[0])
	print("Sampling (frame) rate = ", rate)
	print("Total samples (frames) = ", data.shape)
	print(data)
	
def analyzeFile():
	#wav = wave.open()
	print("Nothing")
	
def main():
	paths = getPaths()
	openFile(paths)

def test():
	main()

if (__name__ == "__main__"):
	test()