import questionary
from prompt_toolkit.shortcuts import CompleteStyle


def select_prompt(message: str, choices: list):
    return questionary.select(
        message,
        choices=choices,
        pointer='\u27A4',
    ).ask()


def confirm_prompt(message: str, default: bool=True):
    return questionary.confirm(
        message,
        default=default
    ).ask()


def autocomplete_prompt(message: str, choices: list[str]):
    return questionary.autocomplete(
        message,
        choices=choices,
        complete_style=CompleteStyle.MULTI_COLUMN
    ).ask()


def text_prompt(message: str, validate=None, default=""):
    return questionary.text(
        message,
        default=default,
        validate=validate,
        validate_while_typing=False
    ).ask()


def input_prompt(data_dir: str):
    return questionary.path(
        "Input directory:",
        get_paths=lambda: [data_dir],
        only_directories=True,
    ).ask()


def output_prompt(maximum: int):
    maximum = int(maximum)
    n_smallest = text_prompt(
        f"Output top [1-{maximum}] images:",
        validate=lambda text: text.isdigit() and 1 <= int(text) <= maximum,
    )
    return int(n_smallest)
