import pytest
from src.audio import preprocessing, segmentation, extraction
from src.clustering import clustering
from src.utils import validation_utils


class TestParams:
    def __init__(
        self,
        path: str,
        num_drums: int,
        expected_peaks: int,
        expected_peaks_accuracy: float,
        expected_labels: list[int],
        expected_label_accuracy: float,
    ):
        """
        peaks_bounds: range of acceptable number of peaks found by segmentation
        ground_truth_labels: label for each segment
        """
        self.path = path
        self.num_drums = num_drums
        self.expected_peaks = expected_peaks
        self.expected_peaks_accuracy = expected_peaks_accuracy
        self.expected_labels = expected_labels
        self.expected_label_accuracy = expected_label_accuracy


def validate_audio_file(params: TestParams):
    # Preprocess data
    data, sample_rate = preprocessing.load_audio(params.path)
    cleaned_audio = preprocessing.clean_audio(data, sample_rate)

    # Create a data set by segmenting audio
    peaks, segments, _ = segmentation.segment_audio(cleaned_audio, sample_rate, debug=True)
    peaks_difference = abs(params.expected_peaks - len(peaks))
    peaks_accuracy = (params.expected_peaks - peaks_difference) / params.expected_peaks
    assert (
        peaks_accuracy >= params.expected_peaks_accuracy
    ), f"\tNum Peaks: {len(peaks)}\n\tExpected Num Peaks: {params.expected_peaks}"

    # Extract features from the data set
    # Group the objects in the data set by clustering
    ffts = extraction.segments_to_features(data, segments, sample_rate, debug=True)
    labels = clustering.cluster(ffts, params.num_drums, debug=True)

    # Validate clustering happened into expected groups
    y_hat = validation_utils.normalize_labels(labels)
    y = validation_utils.normalize_labels(params.expected_labels)
    accuracy = validation_utils.calculate_label_prediction_accuracy(y_hat, y)
    assert (
        accuracy >= params.expected_label_accuracy
    ), f"\tPredictions: {y_hat}\n\tExpected Labels: {y}"


def test_3sounds():
    params = TestParams(
        path="datasets/3sounds.wav",
        num_drums=3,
        expected_peaks=6,
        expected_peaks_accuracy=0.9,
        expected_labels=[0, 0, 1, 1, 2, 2],
        expected_label_accuracy=0.99,
    )
    validate_audio_file(params)


def validate_params(params: TestParams):
    """
    The number of peaks and labels expected should be equal.
    The number of drums can be indicated by the indices in the test.
    """
    assert params.expected_peaks == len(params.expected_labels)
    maximum_label = 0
    for item in params.expected_labels:
        current_val = 0
        if isinstance(item, set):
            current_val = max(item)
        else:
            current_val = item
        if current_val > maximum_label:
            maximum_label = current_val
    assert params.num_drums == maximum_label + 1


def test_snaps():
    params = TestParams(
        path="datasets/snaps.wav",
        num_drums=1,
        expected_peaks=3,
        expected_peaks_accuracy=0.6,
        expected_labels=[0, 0, 0],
        expected_label_accuracy=0.75,
    )
    validate_params(params)
    validate_audio_file(params)


def test_TightSnaps():
    params = TestParams(
        path="datasets/TightSnaps.wav",
        num_drums=1,
        expected_peaks=12,
        expected_peaks_accuracy=0.8,
        expected_labels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        expected_label_accuracy=0.8,
    )
    validate_params(params)
    validate_audio_file(params)


def test_2tapsLong():
    params = TestParams(
        path="datasets/2tapsLong.wav",
        num_drums=2,
        expected_peaks=27,
        expected_peaks_accuracy=0.9,
        expected_labels=[
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ],
        expected_label_accuracy=0.9,
    )
    validate_params(params)
    validate_audio_file(params)


def test_double_stops():
    expected_labels = []
    # table, glass, stomp
    individual_hits = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    for _ in range(2):
        expected_labels.extend(individual_hits)
    # {stomp glass}, {stomp glass}, {stomp table}, glass
    double_stops = [{1, 2}, {1, 2}, {0, 2}, 1]
    for _ in range(4):
        expected_labels.extend(double_stops)

    params = TestParams(
        path="datasets/Double_stops_3.wav",
        num_drums=3,
        expected_peaks=18 + 16,
        expected_peaks_accuracy=0.8,
        expected_labels=expected_labels,
        expected_label_accuracy=0.8,
    )
    validate_params(params)
    validate_audio_file(params)
