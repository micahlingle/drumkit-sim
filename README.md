# DrumKitSim
The goal of this project is to simulate playing a drumkit through programming-based means while still retaining a natural, swung sound. 
This will be accomplished by analyzing user-inputted audio recordings of users drumming on any number of distinct surfaces which have unique
timbres. Each drum sound hit will then be recognized, isolated, and normalized. Tone and tambre will then be recognized by performing
clustering analysis on spectrograms; each distinct surface being hit in the recording will be mapped to a virtual, simulated drum. Then, 
inserting drum samples depending on cluster labelings and recorded-sample amplitudes, a new drum track will be created which is able
to accurately recreate the recorded demo with respect to both time and dynamic emphases. 

# Updates
#### 2/16
Gaussian Mixture Model currently performs the best, even before FFT. 
Trials to be run
* Using vs not using FFT
* More GMM
* Soft-clustering techniques
* Different eps for DBSCAN
* Using CNN to process a Mel Spectrogram

# Resources 
* Pytorch TorchAudio, transforms.MelSpectrogram
* np.fft