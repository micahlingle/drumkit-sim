import pytest
from src.audio import preprocessing, segmentation, extraction
from src.clustering import clustering


class TestParams:
    def __init__(
        self,
        path: str,
        num_drums: int,
        expected_peaks: int,
        expected_ground_truth_labels: list[int],
    ):
        """
        peaks_bounds: range of acceptable number of peaks found by segmentation
        ground_truth_labels: label for each segment
        """
        self.path = path
        self.num_drums = num_drums
        self.expected_peaks = expected_peaks
        self.expected_ground_truth_labels = expected_ground_truth_labels


def normalize_labels(labels: list[int]):
    """
    Because clustering numbering is arbitrary, we need to convert from a list of labels
    to an ordered list of labels. We want to know exactly which sample is messing up the
    clustering algorithm.

    In:
    [1, 1, 0, 0, 2, 2]

    Return:
    [0, 0, 1, 1, 2, 2]
    """

    # Get order of appearance of labels
    order_of_appearance = []
    for label in labels:
        if label not in order_of_appearance:
            order_of_appearance.append(label)

    new_labels = [0 for _ in range(len(labels))]
    for nth_label_idx, nth_label in enumerate(order_of_appearance):
        for j, label in enumerate(labels):
            if label == nth_label:
                new_labels[j] = nth_label_idx

    return new_labels


def validate_audio_file(params: TestParams):

    # Preprocess data
    data, sample_rate = preprocessing.load_audio("datasets/3sounds.wav")
    cleaned_audio = preprocessing.clean_audio(data, sample_rate)

    # Create a data set by segmenting audio
    peaks, segments, velocities = segmentation.segment_audio(cleaned_audio, sample_rate)
    assert len(peaks) == params.expected_peaks

    # Extract features from the data set
    # Group the objects in the data set by clustering
    ffts = extraction.segments_to_ffts(cleaned_audio, segments, sample_rate)
    labels = clustering.cluster(ffts, 3)

    y_hat = normalize_labels(labels)
    y = normalize_labels(params.expected_ground_truth_labels)
    assert y_hat == y

def test_3sounds():
    params = TestParams(
        path="datasets/3sounds.wav",
        num_drums=3,
        expected_peaks=6,
        expected_ground_truth_labels=[0, 0, 1, 1, 2, 2],
    )
    validate_audio_file(params)