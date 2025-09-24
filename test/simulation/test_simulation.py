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
    peaks, segments, _ = segmentation.segment_audio(cleaned_audio, sample_rate)
    peaks_difference = abs(params.expected_peaks - len(peaks))
    peaks_accuracy = (params.expected_peaks - peaks_difference) / params.expected_peaks
    assert (
        peaks_accuracy >= params.expected_peaks_accuracy
    ), f"\tNum Peaks: {len(peaks)}\n\tExpected Num Peaks: {params.expected_peaks}"

    # Extract features from the data set
    # Group the objects in the data set by clustering
    ffts = extraction.segments_to_ffts(cleaned_audio, segments, sample_rate)
    labels = clustering.cluster(ffts, params.num_drums)

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
    assert params.expected_peaks == len(params.expected_labels)
    assert params.num_drums == max(params.expected_labels) + 1


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
