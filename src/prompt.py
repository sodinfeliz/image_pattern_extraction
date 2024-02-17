import questionary

from src.utils import list_all_directories


def select_prompt(message: str, choices: list):
    return questionary.select(
        message,
        choices=choices,
        pointer='\u27A4'
    ).ask()


def directory_prompt(data_dir: str):
    # TODO Let user directly input the dir name
    return select_prompt(
        "Select one of the directory name:",
        choices=list_all_directories(data_dir)
    )


def extraction_prompt(backbone_choices: list[str]):
    return select_prompt(
        "Select the backbone of the extraction model:",
        choices=backbone_choices
    )


def reduction_prompt(reduction_choices: list[str]):
    return select_prompt(
        "Select the dimensionality reduction algorithm:",
        choices=reduction_choices
    )


def clustering_prompt():
    return select_prompt(
        "Select the clustering algorithm:",
        choices=["K-Means", "DBSCAN"]
    )


def output_prompt(maximum: int):
    maximum = int(maximum)
    n_smallest = questionary.text(
        f"Output top [1-{maximum}] images:",
        validate=lambda text: text.isdigit() and 1 <= int(text) <= maximum,
        validate_while_typing=False
    ).ask()
    return int(n_smallest)


if __name__ == "__main__":
    path = directory_prompt(data_dir='./data')
    print(path)
