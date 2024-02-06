import json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans


class ClusterAlgo():
    def __init__(self) -> None:
        self.method = None
        self.algo = None
        self.labels = None

    def set_algo(self, method: str, configs):
        valid_methods = {"KMEANS": KMeans, "DBSCAN": DBSCAN}
        assert method in valid_methods, f"Unknown clustering algorithm: {method}."
        self.method = method
        self.algo = valid_methods[method](**configs[method])

        return self
    
    def display_configs(self):
        """
        Display the current configuration settings of the algorithm
        """
        if self.algo:
            print(json.dumps(self.algo.get_params(), indent=4))
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
        assert self.algo is not None, "Please set the algorithm first."
        self.algo.fit(X_reduced)
        self.labels = self.algo.labels_
        return self.labels
