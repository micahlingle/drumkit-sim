from src.utils import transformation_utils

def test_normalize_labels():
    output = transformation_utils.normalize_labels([2, 1, 3, 4, 5, 5, 6, 1, 3])
    assert output == [0, 1, 2, 3, 4, 4, 5, 1, 2]