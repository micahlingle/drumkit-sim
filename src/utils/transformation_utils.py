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
