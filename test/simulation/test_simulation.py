import pytest
from src.audio import preprocessing, segmentation, extraction
from src.clustering import clustering
from src.utils import validation_utils
from typing import Optional

class AudioParams:
    def __init__(
        self,
        path: str,
        expected_peaks: int,
        expected_peaks_accuracy: float,
        expected_labels: list[int],
        expected_label_accuracy: float,
    ):
        self.path = path
        self.expected_peaks = expected_peaks
        self.expected_peaks_accuracy = expected_peaks_accuracy
        self.expected_labels = expected_labels
        self.expected_label_accuracy = expected_label_accuracy

class TestParams:
    def __init__(
        self,
        num_drums: int,
        audio_train: AudioParams,
        audio_test: Optional[AudioParams] = None
    ):
        self.num_drums = num_drums
        self.audio_test = audio_test
        self.audio_train = audio_train


def validate_test_params(params: TestParams):
    assert params.num_drums > 0
    for audio_obj in (params.audio_train, params.audio_test):
        if audio_obj == None:
            continue
        validate_audio_params(audio_obj)


def validate_audio_params(params: AudioParams):
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


def train_and_test(params: TestParams):
    model = clustering.Model(params.num_drums)
    training = True
    for audio_obj in (params.audio_train, params.audio_test):
        if audio_obj == None:
            continue

        # Preprocess data
        data, sample_rate = preprocessing.load_audio(audio_obj.path)
        cleaned_audio = preprocessing.clean_audio(data, sample_rate)

        # Create a data set by segmenting audio
        peaks, segments, _ = segmentation.segment_audio(cleaned_audio, sample_rate, debug=True)
        peaks_difference = abs(audio_obj.expected_peaks - len(peaks))
        peaks_accuracy = (audio_obj.expected_peaks - peaks_difference) / audio_obj.expected_peaks
        assert (
            peaks_accuracy >= audio_obj.expected_peaks_accuracy
        ), f"\tNum Peaks: {len(peaks)}\n\tExpected Num Peaks: {audio_obj.expected_peaks}"

        # Extract features from the data set
        features = extraction.segments_to_features(data, segments, sample_rate, debug=True)

        # Only train the model once
        if training:
            training = False
            model.train(features, debug=True)
        labels = model.test(features, debug=True)

        # Validate clustering happened into expected groups
        y_hat = validation_utils.normalize_labels(labels)
        y = validation_utils.normalize_labels(audio_obj.expected_labels)
        accuracy = validation_utils.calculate_label_prediction_accuracy(y_hat, y)
        assert (
            accuracy >= audio_obj.expected_label_accuracy
        ), f"\tPredictions: {y_hat}\n\tExpected Labels: {y}"


def test_3sounds():
    params_train = AudioParams(
        path="datasets/3sounds.wav",
        expected_peaks=6,
        expected_peaks_accuracy=0.9,
        expected_labels=[0, 0, 1, 1, 2, 2],
        expected_label_accuracy=0.99,
    )
    params = TestParams(num_drums=3, audio_train=params_train)
    validate_test_params(params)
    train_and_test(params)


def test_snaps():
    params_train = AudioParams(
        path="datasets/snaps.wav",
        expected_peaks=3,
        expected_peaks_accuracy=0.6,
        expected_labels=[0, 0, 0],
        expected_label_accuracy=0.75,
    )
    params = TestParams(num_drums=1, audio_train=params_train)
    validate_test_params(params)
    train_and_test(params)


def test_TightSnaps():
    params_train = AudioParams(
        path="datasets/TightSnaps.wav",
        expected_peaks=12,
        expected_peaks_accuracy=0.8,
        expected_labels=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        expected_label_accuracy=0.8,
    )
    params = TestParams(num_drums=1, audio_train=params_train)
    validate_test_params(params)
    train_and_test(params)


def test_2tapsLong():
    params_train = AudioParams(
        path="datasets/2tapsLong.wav",
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
    params = TestParams(num_drums=2, audio_train=params_train)
    validate_test_params(params)
    train_and_test(params)


def test_double_stops():
    expected_labels = []
    # table, glass, stomp
    individual_hits = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    for _ in range(2):
        expected_labels.extend(individual_hits)
    # {stomp glass}, {stomp glass}, {table glass}, glass
    double_stops = [{1, 2}, {1, 2}, {0, 1}, 1]
    for _ in range(4):
        expected_labels.extend(double_stops)

    params_train = AudioParams(
        path="datasets/Double_stops_3.wav",
        expected_peaks=18 + 16,
        expected_peaks_accuracy=0.8,
        expected_labels=expected_labels,
        expected_label_accuracy=0.8,
    )
    params = TestParams(num_drums=3, audio_train=params_train)
    validate_test_params(params)
    train_and_test(params)
