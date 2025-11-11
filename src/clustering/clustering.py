from sklearn.mixture import GaussianMixture
import numpy as np


def cluster(ffts: list[np.ndarray], n: int, threshold=0.1, similarity= 0.2, random_seed=42, covariance_type="diag", debug=False) -> list[int | set[int]]:
    """
    Cluster ffts where n represents number of clusters.
    """
    model = GaussianMixture(n, covariance_type=covariance_type, random_state=random_seed)
    print(f"len: {len(ffts)}")
    # # normalize by logging?
    # ffts = np.log10(ffts)
    print(ffts[0])


    # Train on TK% of the data so that predictions will not be overfit.
    model.fit(ffts[:int(len(ffts) * .5)])
    p = model.predict_proba(ffts)
    if debug:
        print(f"p: {p}")

    fuzzy_assignments = []
    # Iterate over each row (data point) in the probabilities matrix
    for prob_row in p:
        # Get the indices of components where the probability is above the threshold
        # np.where returns a tuple, we take the first element (the array of indices)
        cluster_indices = np.where(prob_row >= threshold)[0]
        # If thresholds are within similarity to each other, then place them in a set and append the set to fuzzy_assignments.
        if len(cluster_indices) > 1:
            fuzzy_assignments.append(set(cluster_indices))

        # Otherwise, append the index of the component to fuzzy_assignments
        elif len(cluster_indices) == 1:
            fuzzy_assignments.append(cluster_indices[0])

    if debug:
        print(f"labels: {fuzzy_assignments}")
    return fuzzy_assignments
