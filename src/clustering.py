import logging
import sys

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from .general_algo import GeneralAlgo

logger = logging.getLogger(__name__)


class ClusterAlgo(GeneralAlgo):

    _AVAILABLE_ALGO = {
        "K-Means": KMeans,
        "DBSCAN": DBSCAN,
    }

    def __init__(self) -> None:
        super().__init__()
        self.labels: np.ndarray = None

    def apply(self, X_reduced) -> np.ndarray:
        """
        Apply the clustering algorithm to the reduced features.

        Args:
            X_reduced (np.ndarray): Reduced features

        Returns:
            np.ndarray: clustering labels
        """
        if not self._algo:
            logger.exception("Algorithm not set. Call 'set_algo' first.")
            sys.exit(1)
        self._algo.fit(X_reduced)
        self.labels = self._algo.labels_
        return self.labels
