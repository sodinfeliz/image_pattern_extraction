import sys
import logging

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .general_algo import GeneralAlgo

logger = logging.getLogger(__name__)


class ReduceAlgo(GeneralAlgo):

    _AVAILABLE_ALGO = {
        "t-SNE": TSNE,
        "UMAP": UMAP
    }
    _ALGO_NAME = "reduction"

    def __init__(self) -> None:
        super().__init__()

    def apply(self, X: np.ndarray, init_dim: int=100) -> np.ndarray:
        """
        Apply the dimensional reduction algorithm to the `X`

        Args:
            X (np.ndarray): input data (n_samples, in_features)
            init_dim (int): ...

        Returns:
            np.ndarray: (n_samples, out_features)
        """
        if not self._algo:
            logger.exception("Please set the algorithm first.")
            sys.exit(1)
        if isinstance(self._algo, TSNE) and X.shape[1] > init_dim:
            X = PCA(n_components=min(len(X), init_dim)).fit_transform(X)
        X_reduced = self._algo.fit_transform(X)
        return X_reduced
