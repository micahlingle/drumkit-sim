from src.utils import validation_utils


def test_normalize_labels():
    output = validation_utils.normalize_labels([2, 1, 3, 4, 5, 5, 6, 1, 3])
    assert output == [0, 1, 2, 3, 4, 4, 5, 1, 2]


def test_normalize_labels_with_sets():
    output = validation_utils.normalize_labels([{2, 1, 3}, 4, {5, 6, 1, 3}])
    assert output == [{0, 1, 2}, 3, {4, 5, 1, 2}]


def test_calculate_label_prediction_accuracy():
    y_hat = [0, 1, 2, 3]
    y = [0, 1, 2]
    output = validation_utils.calculate_label_prediction_accuracy(y_hat, y)
    assert output == 0.75
