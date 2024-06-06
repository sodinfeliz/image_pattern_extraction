import json

from .prompt import autocomplete_prompt, confirm_prompt, select_prompt, text_prompt


class GeneralAlgo:

    _AVAILABLE_ALGO = {}
    _ALGO_NAME = ""

    def __init__(self) -> None:
        self._algo = None

    @property
    def method(self):
        if self._algo:
            return self._algo.__class__.__name__
        return None

    def set_algo(self, method: str, configs: dict):
        if method not in self._AVAILABLE_ALGO:
            raise ValueError(
                f"Unknown {self._ALGO_NAME} method: '{method}'. "
                + f"Available methods are: {list(self._AVAILABLE_ALGO.keys())}."
            )
        algo_class = self._AVAILABLE_ALGO[method]
        self._algo = algo_class(**configs.get(method, {}))
        return self

    def display_configs(self):
        """
        Display the current configuration settings of the algorithm
        """
        if self._algo:
            print(json.dumps(self._algo.get_params(), indent=4))
        else:
            print("No algorithm configured.")

    def update_algo_config(self, configs: dict):
        if not self._algo:
            raise RuntimeError("Algorithm not set. Call 'set_algo' first.")
        self._algo.set_params(**configs)
        return self

    @classmethod
    def prompt(cls, message: str, configs: dict):
        method = select_prompt(message, choices=list(cls._AVAILABLE_ALGO.keys()))

        algo_instance = cls._AVAILABLE_ALGO[method]()
        algo_configs = algo_instance.get_params()
        algo_configs.update(configs.get(method, {}))

        if confirm_prompt("Would you like to fine-tune the parameters?"):
            finished = False
            while not finished:
                selected_param = autocomplete_prompt(
                    "Select the parameter:", choices=algo_configs.keys()
                )
                cur_val = algo_configs[selected_param]
                value_type = type(cur_val)

                while True:
                    user_input = text_prompt(
                        f"Set '{selected_param}' [{value_type.__name__}: {cur_val}]:"
                    )
                    try:
                        if user_input and user_input.lower() != "none":
                            algo_configs[selected_param] = value_type(user_input)
                        # Validate by setting params
                        algo_instance.set_params(**algo_configs)
                        break  # Exit the loop if successfully updated and validated
                    except (ValueError, TypeError) as e:
                        print(f"Error updating {selected_param}: {e}")

                configs[method][selected_param] = algo_configs[selected_param]
                finished = not confirm_prompt("Continue adjusting parameters?")
        else:
            print("Skipping parameter tuning.")

        return method
