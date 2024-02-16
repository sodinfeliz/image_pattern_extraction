import os
import questionary
import pyinputplus as pyip

from src.utils import list_all_directories


def directory_prompt(data_dir: str):
    # def validate(dirname: str):    
    #     if dirname != 'list' and not os.path.exists(os.path.join(data_dir, dirname)):
    #         raise Exception("Directory doesn't exist.")

    # dirname = pyip.inputCustom(
    #     customValidationFunc=validate,
    #     prompt="Enter directory name, or type 'list' to see available directories: ")
    
    # if dirname == 'list':
    #     l = list_all_directories("./data")
    #     dirname = pyip.inputMenu(l, numbered=True)

    dirname = questionary.select(
        "Select one of the directory name:",
        choices=list_all_directories(data_dir)
    ).ask()

    return dirname


def extraction_prompt():
    backbone = questionary.select(
        "Select the backbone of the extraction model:",
        choices=["ResNet", "EfficientNet"]
    ).ask()
    return backbone.lower()


def reduction_prompt():
    reduction_method = questionary.select(
        "Select the dimensionality reduction algorithm:",
        choices=["t-SNE", "UMAP"]
    ).ask()
    return reduction_method


def clustering_prompt():
    cluster_method = questionary.select(
        "Select the clustering algorithm:",
        choices=["K-Means", "DBSCAN"]
    ).ask()
    return cluster_method


def output_prompt(maximum: int):
    maximum = int(maximum)
    def validate_input(text):
        if text.isdigit():
            num = int(text)
            return 1 <= num <= maximum
        return False

    n_smallest = questionary.text(
        f"Output top [1-{maximum}] images:",
        validate=validate_input,
        validate_while_typing=False
    ).ask()
    
    return int(n_smallest)


if __name__ == "__main__":
    path = directory_prompt(data_dir='./data')
    print(path)
