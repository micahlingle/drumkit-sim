from sklearn.mixture import GaussianMixture


def cluster(ffts, n, debug=False):
    """
    Cluster ffts where n represents number of clusters.
    """
    model = GaussianMixture(n)
    labels = model.fit_predict(ffts)
    if debug:
        print(labels)
    return labels
