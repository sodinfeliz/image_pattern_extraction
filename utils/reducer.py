import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


class DimReducer():
    def __init__(self) -> None:
        self.method = None
        self._algo = None
        self.configs = dict()

    def display_configs(self):
        if self._algo:
            print(json.dumps(self._algo.get_params(), indent=4))
        else:
            print("No algorithm configured.")

    def set_algo(self, method, configs):
        valid_methods = {"TSNE": TSNE, "UMAP": UMAP}
        assert method in valid_methods, f"Unknown reduction method: {method}."
        self.method = method
        self._algo = valid_methods[method](**configs[method])
        return self
    
    def apply(self, X: np.ndarray, init_dim: int=100) -> np.ndarray:
        """
        Apply the dimensional reduction algorithm to the `X`

        Args:
            X (np.ndarray): input data (n_samples, in_features)
            init_dim (int): ...

        Returns:
            np.ndarray: (n_samples, out_features)
        """
        assert self._algo is not None, "Please set algorithm first."
        if self.method == "TSNE":
            X = PCA(n_components=min(len(X), init_dim)).fit_transform(X)
        X_reduced = self._algo.fit_transform(X)
        return X_reduced
