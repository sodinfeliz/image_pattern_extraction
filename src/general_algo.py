from .prompt import (
    text_prompt,
    autocomplete_prompt,
    confirm_prompt,
    select_prompt
)


class GeneralAlgo:

    _AVAILABLE_ALGO = {}

    @classmethod
    def prompt(cls, message: str, configs: dict):
        method = select_prompt(message, choices=list(cls._AVAILABLE_ALGO.keys()))

        algo_instance = cls._AVAILABLE_ALGO[method]()
        algo_configs = algo_instance.get_params()
        algo_configs.update(configs.get(method, {}))

        if confirm_prompt("Would you like to fine-tune the parameters?"):
            finished = False
            while not finished:
                selected_param = autocomplete_prompt("Select the parameter:", choices=algo_configs.keys())
                cur_val = algo_configs[selected_param]
                vtype = type(cur_val)

                while True:
                    user_input = text_prompt(f"Set '{selected_param}' [{vtype.__name__}: {cur_val}]:")
                    try:
                        if user_input and user_input.lower() != "none":
                            algo_configs[selected_param] = vtype(user_input)
                        # Validate by setting params
                        algo_instance.set_params(**algo_configs)
                        break # Exit the loop if successfully updated and validated
                    except (ValueError, TypeError) as e:
                        print(f"Error updating {selected_param}: {e}")

                configs[method][selected_param] = algo_configs[selected_param]
                finished = not confirm_prompt("Continue adjusting parameters?")
        else:
            print("Skipping parameter tuning.")

        return method

