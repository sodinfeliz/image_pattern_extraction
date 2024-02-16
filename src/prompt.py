import os
import pyinputplus as pyip

from src.utils import list_all_directories


def directory_prompt(data_dir: str):
    def validate(dirname: str):    
        if dirname != 'list' and not os.path.exists(os.path.join(data_dir, dirname)):
            raise Exception("Directory doesn't exist.")

    dirname = pyip.inputCustom(
        customValidationFunc=validate,
        prompt="Enter directory name, or type 'list' to see available directories: ")
    
    if dirname == 'list':
        l = list_all_directories("./data")
        dirname = pyip.inputMenu(l, numbered=True)

    return dirname


def extraction_prompt():
    backbone = pyip.inputMenu(
        choices=["ResNet", "EfficientNet"],
        numbered=True,
        prompt="Select the backbone of the extraction model: \n"
    ).lower()
    return backbone


def reduction_prompt():
    reduction_method = pyip.inputMenu(
        choices=["t-SNE", "UMAP"], 
        numbered=True,
        prompt="Select the dimensionality reduction algorithm: \n"
    )
    return reduction_method


def clustering_prompt():
    cluster_method = pyip.inputMenu(
        choices=["K-Means", "DBSCAN"], 
        numbered=True,
        prompt="Select the clustering algorithm: \n"
    )
    return cluster_method


def output_prompt(maximum: int):
    n_smallest = pyip.inputInt(
        greaterThan=0, max=int(maximum),
        prompt=f"Output top [1-{maximum}] images: "
    )
    return n_smallest


if __name__ == "__main__":
    path = directory_prompt(data_dir='./data')
    print(path)
