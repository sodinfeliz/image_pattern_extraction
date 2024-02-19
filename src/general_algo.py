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

