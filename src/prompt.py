import questionary

from src.utils import list_all_directories


def select_prompt(message: str, choices: list):
    return questionary.select(
        message,
        choices=choices,
        pointer='\u27A4',
    ).ask()


def directory_prompt(data_dir: str):
    return questionary.path(
        "Input directory:",
        get_paths=lambda: [data_dir],
        only_directories=True,
    ).ask()


def output_prompt(maximum: int):
    maximum = int(maximum)
    n_smallest = questionary.text(
        f"Output top [1-{maximum}] images:",
        validate=lambda text: text.isdigit() and 1 <= int(text) <= maximum,
        validate_while_typing=False,
    ).ask()
    return int(n_smallest)


if __name__ == "__main__":
    path = directory_prompt(data_dir='./data')
    print(path)
