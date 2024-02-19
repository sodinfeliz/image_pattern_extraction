import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from .prompt import (
    select_prompt,
    confirm_prompt,
    autocomplete_prompt,
    text_prompt,
)

class DimReducer():

    _AVAILABLE_ALGO = {
        "t-SNE": TSNE,
        "UMAP": UMAP
    }

    def __init__(self) -> None:
        self._method = None
        self._algo = None
        self.configs = dict()

    @property
    def method(self):
        if self._algo:
            return self._algo.__class__.__name__
        return None

    def display_configs(self):
        if self._algo:
            print(json.dumps(self._algo.get_params(), indent=4))
        else:
            print("No algorithm configured.")

    def set_algo(self, method: str, configs: dict):
        if method not in self._AVAILABLE_ALGO:
            raise ValueError(f"Unknown reduction method: '{method}'. " +
                             f"Available methods are: {list(self._AVAILABLE_ALGO.keys())}.")
        algo_calss = self._AVAILABLE_ALGO[method]
        self._algo = algo_calss(**configs.get(method, {}))
        return self
    
    def update_algo_config(self, configs: dict):
        if not self._algo:
            raise RuntimeError("Algorithm not set. Call 'set_algo' first.")
        self._algo.set_params(**configs)
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
        if not self._algo:
            raise RuntimeError("Please set the algorithm first.")
        if isinstance(self._algo, TSNE) and X.shape[1] > init_dim:
            X = PCA(n_components=min(len(X), init_dim)).fit_transform(X)
        X_reduced = self._algo.fit_transform(X)
        return X_reduced
    
    @classmethod
    def prompt(cls, configs: dict):
        method = select_prompt(
            "Select the dimensionality reduction algorithm:",
            choices=list(cls._AVAILABLE_ALGO.keys())
        )

        finished = False
        all_configs = cls._AVAILABLE_ALGO[method]().get_params()
        all_configs.update(configs[method])

        if confirm_prompt("Would you like to fine-tune the parameters?"):
            while not finished:
                selected = autocomplete_prompt("Select the parameter:", choices=all_configs.keys())
                cur_val = all_configs[selected]
                vtype = type(cur_val)

                while True:
                    user_input = text_prompt(f"Set '{selected}' [{vtype.__name__}: {cur_val}]:")
                    try:
                        if user_input and user_input.lower() != "none":
                            all_configs[selected] = vtype(user_input)
                        # Validate by setting params
                        cls._AVAILABLE_ALGO[method]().set_params(**all_configs)
                        break # Exit the loop if successfully updated and validated
                    except ValueError as e:
                        print(f"Invalid value for {selected}: {e}")
                    except TypeError as e:
                        print(f"Invalid type for {selected}: {e}")

                configs[method][selected] = all_configs[selected]
                finished = not confirm_prompt("Continue adjusting parameters?")
        else:
            print("Skipping parameter tuning.")

        return method
