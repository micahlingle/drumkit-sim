## Studio Drumming, Reimagined

The goal of this project is to simulate playing a drumkit through programming-based means while still retaining a natural, swung sound. This will be accomplished by analyzing user-inputted audio recordings of users drumming on any number of distinct surfaces which have unique timbres. Each drum impact soundbyte will then be recognized, isolated, and normalized. Tone and tambre will then be recognized by performing clustering analysis on spectograms; each distinct surface being hit in the recording will be mapped to a virtual, simulated drum. Then, inserting drum samples depending on cluster labelings and recorded-sample amplitudes, a new drum track will be created which is able to accurately recreate the recorded demo with respect to both time and dynamic emphases.

### No studio required

Having a studio or expensive microphone setup is no problem. A recording must simply be inputted to the program to bring studio-quality sound and sampling to life in audio tracks.
You can play drums anywhere - even on the kitchen sink. 

## Methodology

In order to perform clustering on sounds, soundbytes must be represented in a numerical fashion. Because clustering should be performed based on the timbre of sounds, spectograms will be created to represent tonal frequency variations. 
Then, spectograms will be analyzed by a CNN. Instead of the network performing classification or generating a numerical result, outputs (in vector form) will be taken from the convolutional layers before fully-connected layers are reached. 
These vectors may then be used as input into a traditional soft-clustering algorithm. Thus, vectors, therefore images, therefore sounds are classified.

### Progress

This project is due for completion in early 2022. 
