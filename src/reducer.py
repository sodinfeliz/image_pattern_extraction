import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .prompt import select_prompt


class DimReducer():

    _AVAILABLE_ALGO = [
        "t-SNE",
        "UMAP"
    ]

    def __init__(self) -> None:
        self.method = None
        self._algo = None
        self.reduction_methods = {
            "t-SNE": TSNE,
            "UMAP": UMAP
        }
        self.configs = dict()

    def display_configs(self):
        if self._algo:
            print(json.dumps(self._algo.get_params(), indent=4))
        else:
            print("No algorithm configured.")

    def set_algo(self, method, configs):
        assert method in self._AVAILABLE_ALGO, f"Unknown reduction method: {method}."
        self.method = method
        self._algo = self.reduction_methods[method](**configs[method])
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
        if self.method == "t-SNE":
            X = PCA(n_components=min(len(X), init_dim)).fit_transform(X)
        X_reduced = self._algo.fit_transform(X)
        return X_reduced
    
    @classmethod
    def prompt(cls):
        return select_prompt(
            "Select the dimensionality reduction algorithm:",
            choices=cls._AVAILABLE_ALGO
        )
