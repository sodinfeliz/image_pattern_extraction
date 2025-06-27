import argparse
import copy
import json
import logging
import logging.config
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import toml
from rich import print
from rich.progress import Progress, TaskID

from src import DrawResult
from src.files import first_subdirectory, open_directory
from src.ml_utils import ClusterAlgo, FeatureExtractor, ReduceAlgo
from src.prompt import input_prompt, output_prompt, select_prompt

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up the logging configuration."""
    logging_config_file = Path("configs/logging.json")
    try:
        with open(logging_config_file) as f_in:
            logging_config = json.load(f_in)
    except FileNotFoundError:
        logging.exception(
            f"Can't find the logging configuration file: {logging_config_file}"
        )
        sys.exit(1)
    logging.config.dictConfig(logging_config)


class MainProcess:

    _STEP_DESC = {
        1: "input",
        2: "extraction",
        3: "reduction",
        4: "clustering",
        5: "output",
    }

    def __init__(self, config_path: Union[str, Path]):
        self.config_path: Path = Path(config_path)
        self.step_methods: dict = {
            "input": self.input_step,
            "extraction": self.extraction_step,
            "reduction": self.reduction_step,
            "clustering": self.clustering_step,
            "output": self.output_step,
        }
        self.extractor: Optional[FeatureExtractor] = None
        self.step: int = 1  # step starts from 1
        self.stop_process: bool = False
        self._load_configs()

    #############################
    # Configuration related
    #############################

    def _load_configs(self):
        """Set the configurations from the user configuration file"""
        try:
            with open(self.config_path, "r") as file:
                self.original_configs = toml.load(file)  # for rollback
                self.configs = copy.deepcopy(self.original_configs)

            self.data_dir = Path(self.configs["global_settings"]["data_dir"])
            self.result_dir = Path(self.configs["global_settings"]["result_dir"])
        except FileNotFoundError:
            self._exit_process(f"Can't find the configuration file: {self.config_path}", rollback=False)
        except KeyError as error:
            self._exit_process(f"KeyError: The key '{error.args[0]}' is missing from global_settings in configuration file", rollback=False)

        self._hline(" Load configurations ")
        print("Configuration file: ", self.config_path)

    def _update_configs(self):
        """Update the user configurations from the current settings."""
        try:
            with open(self.config_path, "w") as file:
                toml.dump(self.configs, file)
        except FileNotFoundError:
            self._exit_process(f"Can't find the configuration file: {str(self.config_path)}")
        except Exception as e:
            self._exit_process(f"Error updating the configuration file: {e}")

    def _rollback_configs(self):
        """Rollback the user configurations to the previous settings."""
        try:
            with open(self.config_path, "w") as file:
                toml.dump(self.original_configs, file)
        except Exception as e:
            self._exit_process(f"Error when rolling back the configuration file: {e}", rollback=False)

    #############################
    # Main process related
    #############################

    def start(self):
        """Start the main process loop."""
        try:
            while not self.stop_process and self.step <= len(self._STEP_DESC):
                step_desc = self._STEP_DESC[self.step]
                method = self.step_methods.get(step_desc)
                try:
                    self._hline(f" Step {self.step}: {step_desc.capitalize()} ")
                    method()
                    self._proceed()
                except TypeError:
                    self.stop_process = True
                    logger.exception("Undefined step.")
            else:
                if self.stop_process:
                    self._hline(" Process stopped ")
                    self._rollback_configs()
                else:
                    self._hline(" Process completed ")
        except Exception as e:
            self._exit_process(f"An error occurred during the process: {e}")

    def _proceed(self):
        """Handles the navigation between steps."""
        if self.stop_process:
            return

        step_desc = self._STEP_DESC[self.step]

        if step_desc in ["extraction", "output"]:  # Skip the prompt for these steps
            response = "Next"
        else:
            response = self._prompt_next_action(step_desc)

        if response == "Next":
            self.step += 1
        elif response == "Back":
            self.step = max(self.step - 1, 1)
        elif response == "Exit":
            self.stop_process = True
        elif response == "Repeat":
            pass
        else:
            logger.exception(f"Undefined response: '{response}'.")
            self.stop_process = True

    def _prompt_next_action(self, step_desc: str) -> str:
        """Prompts the user for the next action after a step is completed."""
        print(
            f"\n[bold dodger_blue1]{step_desc.capitalize()}[/bold dodger_blue1] step completed. ",
            end="",
        )
        print("What would you like to do next?")

        choices = ["Next", "Repeat", "Exit"]
        if step_desc != "input":
            choices.insert(2, "Back")

        return select_prompt("Select the next action:", choices=choices)

    def _hline(self, text: str) -> None:
        """Print a horizontal line with the given text."""
        print(f"\n{text:=^80}\n")

    def input_step(self):
        """Input step: Select the data directory."""
        if (first_dirname := first_subdirectory(self.data_dir)) is None:
            self._exit_process("There's no available image data.", rollback=False)

        self.dirname = input_prompt(
            data_dir=self.data_dir,
            exit_keys=self.configs["global_settings"]["exit_keys"],
        )
        if self.dirname in self.configs["global_settings"]["exit_keys"]:
            self.stop_process = True
            return

        if not self.dirname:
            self.dirname = first_dirname

        self.src_path = self.data_dir / self.dirname
        print(f"Input data path: {self.src_path}")

        self.dst_path = self.result_dir / self.dirname
        shutil.rmtree(self.dst_path, ignore_errors=True)
        self.dst_path.mkdir(parents=True)

    def extraction_step(self):
        """Extraction step: Extract features from the input images."""
        self.backbone = FeatureExtractor.prompt()
        if self.extractor is None:
            self.extractor = FeatureExtractor(
                configs=self.configs["extractor"], backbone=self.backbone
            )
        else:
            self.extractor.set_encoder(self.backbone)
        self.X, self.image_paths = self.extractor.extract(path=self.src_path)

    def reduction_step(self):
        """Reduction step: Reduce the dimensionality of the extracted features."""
        self.reduction_method = ReduceAlgo.prompt(
            message="Select the dimensionality reduction algorithm:",
            configs=self.configs["reduction"],
        )
        self._update_configs()

        print("\nStart reducing dimensionality ... ", end="")
        self.reducer = ReduceAlgo().set_algo(
            method=self.reduction_method, configs=self.configs["reduction"]
        )
        self.X_reduced = self.reducer.apply(self.X)
        self.reduced_dim = self.X_reduced.shape[1]
        print("[bold chartreuse3]completed[/bold chartreuse3]")

        self.df = pd.DataFrame(self.X_reduced)
        self.df.columns = [f"x{i+1}" for i in range(self.reduced_dim)]

        DrawResult.draw_reduction(self.df, reduction_method=self.reduction_method)

    def clustering_step(self):
        """Clustering step: Cluster the reduced features."""
        self.cluster_method = ClusterAlgo.prompt(
            message="Select the clustering algorithm:",
            configs=self.configs["clustering"],
        )
        self._update_configs()

        print("\nStart clustering ... ", end="")
        self.cluster_algo = ClusterAlgo().set_algo(
            method=self.cluster_method, configs=self.configs["clustering"]
        )
        self.df["cluster"] = self.cluster_algo.apply(self.X_reduced)
        print("[bold chartreuse3]completed[/bold chartreuse3]")

        self.df_mean = self.df.groupby("cluster").mean()
        self.df_mean.columns = [f"mean_x{i+1}" for i in range(self.reduced_dim)]

        # Separate the outliers from the clusters if DBSCAN is used
        # DBSCAN assigns outliers to cluster -1
        self.df_outlier = None
        if self.cluster_method == "DBSCAN" and -1 in self.df_mean.index:
            self.df_outlier = self.df[self.df["cluster"] == -1]
            self.df_filter = self.df[self.df["cluster"] != -1]
            self.df_mean.drop(-1, inplace=True)
        else:
            self.df_filter = self.df.copy()

        if self.df_outlier is not None:
            print("Outliers count: ", len(self.df_outlier))

        print("Clustering mean: ")
        print(self.df_mean)

        DrawResult.draw_clustering(
            self.df_filter,
            self.df_mean,
            reduction_method=self.reduction_method,
            cluster_method=self.cluster_method,
        )

    def output_step(self):
        """Output step: Output the results."""

        self.df_merge = pd.merge(
            self.df_filter, self.df_mean, how="inner", on="cluster"
        )
        # Calculate squared distance across all dimensions
        dist_square = pd.Series(0, index=self.df_merge.index)
        for i in range(self.reduced_dim):
            dist_square += (self.df_merge[f"x{i+1}"] - self.df_merge[f"mean_x{i+1}"]) ** 2
        self.df_merge["dist_square"] = dist_square

        # Assign the original index to the merged dataframe
        self.df_merge.index = self.df_filter.index

        smallest_cluster = int(self.df_merge.groupby("cluster")["x1"].count().min())
        n_smallest = output_prompt(smallest_cluster)

        df_rank = self.df_merge.reset_index().rename(
            columns={"index": "original_index"}
        )
        df_rank = df_rank.sort_values(["cluster", "dist_square"])
        df_rank = df_rank.groupby("cluster").head(n_smallest)
        df_rank["rank"] = df_rank.groupby("cluster").cumcount()
        df_rank = df_rank.pivot(
            index="cluster", columns="rank", values="original_index"
        )

        # draw top n images
        DrawResult.draw_top_n_output(
            df_rank, image_paths=self.image_paths, dst_path=self.dst_path
        )

        # copy images to the cluster directory
        cluster_num = len(df_rank)
        points_per_cluster = len(df_rank.columns)

        with Progress() as progress:
            task_id: TaskID = progress.add_task(
                description="[cyan]Copying images: ",
                total=cluster_num * points_per_cluster,
            )
            for i in range(cluster_num):
                cluster_dir = self.dst_path / f"cluster_{i}"
                cluster_dir.mkdir()
                for j in range(points_per_cluster):
                    src = self.image_paths[df_rank.iloc[i, j]]
                    dst = cluster_dir / src.name
                    shutil.copy(src, dst)
                    progress.update(task_id, advance=1)

        # copy outliers to the outlier directory
        if self.df_outlier is not None:
            outlier_dir = self.dst_path / "outliers"
            outlier_dir.mkdir()
            for i in range(len(self.df_outlier)):
                src = self.image_paths[self.df_outlier.index[i]]
                dst = outlier_dir / src.name
                shutil.copy(src, dst)

        # override user configuration file
        with open(self.config_path, "w") as file:
            toml.dump(self.configs, file)

        # copy the configuration file to the result directory
        shutil.copy(self.config_path, self.dst_path / "config.toml")

        # output the final result
        df_final = self.df.copy()
        df_final["image_path"] = self.image_paths
        df_final.to_csv(self.dst_path / "result.csv", index=False)

        open_directory(self.dst_path)

    #############################
    # Exit related
    #############################

    def _exit_process(
        self,
        error_message: Optional[str] = None,
        rollback: bool = True,
    ):
        """Exit the process with an error."""
        if error_message:
            logger.exception(error_message)
        if rollback:
            self._rollback_configs()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/user-config.toml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # copy the default configuration file
        shutil.copy(
            Path(__file__).parent / "configs" / "config.toml",
            config_path,
        )

    setup_logging()
    process = MainProcess(config_path=config_path)
    process.start()
