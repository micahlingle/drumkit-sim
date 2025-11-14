# Micah Lingle
# DrumKitSim

from src.utils import cli
from src.audio import preprocessing, segmentation, extraction
from src.clustering import clustering
from src.midi import midi


def main():
    cli.parse_args()
    args = cli.cli_args

    # Preprocess data
    data, sample_rate = preprocessing.load_audio(args.path)
    cleaned_audio = preprocessing.clean_audio(data, sample_rate)

    # Create a data set by segmenting audio
    peaks, segments, velocities = segmentation.segment_audio(cleaned_audio, sample_rate)

    # Extract features from the data set
    ffts = extraction.segments_to_features(cleaned_audio, segments, sample_rate)

    # Group the objects in the data set by clustering
    labels = clustering.cluster(ffts, args.num_drums)

    # Launch interactive classification of drum sounds for the user
    midi.classify_audio_and_export_midi(
        args.num_drums,
        segments,
        labels,
        peaks,
        cleaned_audio,
        sample_rate,
        velocities,
        args.bpm,
        args.time_signature,
    )


if __name__ == "__main__":
    main()
