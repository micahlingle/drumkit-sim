from sklearn.mixture import GaussianMixture
import numpy as np


class Model:

    def __init__(self, n):
        self._model = GaussianMixture(
            n_components=n, covariance_type="diag", random_state=42
        )

    def train(self, features: list[np.ndarray], debug=False):
        if debug:
            print(f"len: {len(features)}")
            print(features[0])
        self._model.fit(features)

    def test(
        self, features: list[np.ndarray], threshold=0.1, debug=False
    ) -> list[int | set[int]]:
        """
        Cluster features where n represents number of clusters.
        """

        p = self._model.predict_proba(features)
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
