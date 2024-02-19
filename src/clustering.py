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
        self._method = None
        self._algo = None
        self.labels = None

    @property
    def method(self):
        return self._method
    
    @method.setter
    def method(self, _):
        raise AttributeError("Directly modification of method disabled.")

    def set_algo(self, method: str, configs):
        if method not in self._AVAILABLE_ALGO:
            raise ValueError(f"Unknown clustering algorithm: {method}.")
        self._method = method
        self._algo = self._AVAILABLE_ALGO[method](**configs[method])
        return self
    
    def display_configs(self):
        """
        Display the current configuration settings of the algorithm
        """
        if self._algo:
            print(json.dumps(self._algo.get_params(), indent=4))
        else:
            print("No algorithm configured.")

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
