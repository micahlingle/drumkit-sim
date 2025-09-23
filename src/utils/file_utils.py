import os
import shutil
import soundfile as sf


def save_all_audio_by_label(
    path, num_drums, segments, labels, peaks, filtered, sample_rate
):
    """
    Save audio to temp folder for playback/interaction
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    for i in range(num_drums):
        os.mkdir(f"{path}/{i}")
    for segment, label, peak in zip(segments, labels, peaks):
        start, stop = segment
        sf.write(f"{path}/{label}/{peak}.wav", filtered[start:stop], sample_rate)


def get_paths(p="./datasets/"):
    """Find paths to audio files"""
    # File separators
    entries = os.listdir(p)
    for i in range(len(entries)):
        entries[i] = f"{p}{entries[i]}"
    paths = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        if os.path.isdir(entry):
            nested = os.listdir(entry)
            for j in range(len(nested)):
                nested[j] = f"{entry}/{nested[j]}"
                entries.append(nested[j])
        else:
            paths.append(entry)
        i += 1
    print("Recursively found files: ", paths)
    return paths
