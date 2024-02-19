import json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .general_algo import GeneralAlgo


class ClusterAlgo(GeneralAlgo):

    _AVAILABLE_ALGO = {
        "K-Means": KMeans,
        "DBSCAN": DBSCAN,
    }

    def __init__(self) -> None:
        super().__init__()
        self.labels = None

    def apply(self, X_reduced) -> np.ndarray:
        """ 
        Apply the clustering algorithm on X_reduced

        Args:
            X_reduced (np.ndarray): Reduced features of X

        Returns:
            np.ndarray: clustering labels
        """
        if not self._algo:
            raise RuntimeError("Algorithm not set. Call 'set_algo' first.")
        self._algo.fit(X_reduced)
        self.labels = self._algo.labels_
        return self.labels
