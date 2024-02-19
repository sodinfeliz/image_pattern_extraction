import os
import yaml
import shutil
import pandas as pd
import argparse
from rich import print

from src.utils import (
    open_directory,
    list_all_directories
)
from src import (
    ClusterAlgo,
    ReduceAlgo,
    FeatureExtractor,
)
from src.draw import DrawResult
from src.prompt import (
    directory_prompt,
    output_prompt,
    select_prompt,
)


def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


class MainProcess:

    _STEP_DESC = {
        1: "input",
        2: "extraction",
        3: "reduction",
        4: "clustering",
        5: "output"
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.step_methods = {
            "input": self.input_step,
            "extraction": self.extraction_step,
            "reduction": self.reduction_step,
            "clustering": self.clustering_step,
            "output": self.output_step
        }
        self.extractor = None
        self.step = 1
        self.load_configs()

    def load_configs(self):
        if not os.path.exists(self.config_path):
            raise FileExistsError(f"Can't find the configuration file: {self.config_path}")

        self.configs = load_config(self.config_path)
        self.data_dir = self.configs['global_settings']['data_dir']
        self.result_dir = self.configs['global_settings']['result_dir']

    def start(self):
        """ Start the main process loop. """ 
        while self.step <= len(self._STEP_DESC):
            step_desc = self._STEP_DESC[self.step]
            method = self.step_methods.get(step_desc)
            if method:
                method()
                self.proceed()
            else:
                raise AttributeError("Undefined step.") 

    def proceed(self):
        """ Handles the navigation between steps. """
        step_desc = self._STEP_DESC[self.step]

        if step_desc in ["extraction", "output"]:
            response = "Next"
        else:
            response = self.prompt_next_action(step_desc)

        self.hline()
        if response == "Next":
            self.step += 1
        elif response == "Back":
            self.step = max(self.step-1, 1)

    def prompt_next_action(self, step_desc):
        """ Prompts the user for the next action after a step is completed. """
        print(f"\n[bold dodger_blue1]{step_desc.capitalize()}[/bold dodger_blue1] step completed. ", end="")
        print("What would you like to do next?")
        return select_prompt(
            "Select 'Next' to proceed, 'Repeat' to redo, or 'Back' to return to the previous step:",
            choices=['Next', 'Repeat', 'Back']
        )

    def hline(self, symbol='=', count=94):
        """ Prints a horizontal line. """
        print("\n" + symbol*count + "\n")

    def input_step(self):
        if len(list_all_directories(self.data_dir)) == 0:
            raise Exception("There's no available image data.")
        
        self.dirname = directory_prompt(data_dir=self.data_dir)
        self.src_path = os.path.join(self.data_dir, self.dirname)
        print(f"Input data path: {self.src_path}")

        self.dst_path = os.path.join(self.result_dir, self.dirname)
        if os.path.exists(self.dst_path):
            shutil.rmtree(self.dst_path)
        os.mkdir(self.dst_path)
        
    def extraction_step(self):
        self.backbone = FeatureExtractor.prompt()
        if self.extractor is None:
            self.extractor = FeatureExtractor(
                configs=self.configs['extractor'],
                backbone=self.backbone)
        else:
            self.extractor.set_model(self.backbone)
        self.X, self.image_names = self.extractor.extract(path=self.src_path)

    def reduction_step(self):
        self.reduction_method = ReduceAlgo.prompt(
            message="Select the dimensionality reduction algorithm:",
            configs=self.configs['reduction'])
        print("\nStart reducing dimensionality ... ", end="")
        self.reducer = ReduceAlgo().set_algo(
            method=self.reduction_method, 
            configs=self.configs['reduction'])
        self.X_reduced = self.reducer.apply(self.X)
        print("[bold chartreuse3]completed[/bold chartreuse3]")

        self.df = pd.DataFrame(self.X_reduced)
        self.df.columns = ['x', 'y']
        DrawResult.draw_reduction(self.df, self.reduction_method)

    def clustering_step(self):
        self.cluster_method = ClusterAlgo.prompt(
            message="Select the clustering algorithm:",
            configs=self.configs['clustering'])
        print("\nStart clustering ... ", end="")
        self.cluster_algo = ClusterAlgo().set_algo(
            method=self.cluster_method, 
            configs=self.configs['clustering'])
        self.df['cluster'] = self.cluster_algo.apply(self.X_reduced)
        print("[bold chartreuse3]completed[/bold chartreuse3]")

        self.df_mean = self.df.groupby('cluster').mean()
        self.df_mean.columns = ['mean_x', 'mean_y']

        self.df_outlier = None
        if self.cluster_method == 'DBSCAN' and -1 in self.df_mean.index:
            self.df_outlier = self.df[self.df['cluster'] == -1]
            self.df_filter = self.df[self.df['cluster'] != -1]
            self.df_mean.drop(-1, inplace=True)
        else:
            self.df_filter = self.df.copy()

        print("Clustering mean: ")
        print(self.df_mean)

        DrawResult.draw_clustering(
            self.df_filter, self.df_mean,
            reduction_method=self.reduction_method,
            cluster_method=self.cluster_method
        )

    def output_step(self):
        self.df_merge = pd.merge(self.df_filter, self.df_mean, how='inner', on='cluster')
        self.df_merge['dist_square'] = (self.df_merge['x'] - self.df_merge['mean_x'])**2 + \
                                       (self.df_merge['y'] - self.df_merge['mean_y'])**2
        self.df_merge.index = self.df_filter.index
        n_smallest = output_prompt(maximum=self.df_merge.groupby('cluster')['x'].count().min())

        df_rank = self.df_merge.reset_index().rename(columns={'index': 'original_index'})
        df_rank = df_rank.sort_values(['cluster', 'dist_square'])
        df_rank = df_rank.groupby('cluster').head(n_smallest)
        df_rank['rank'] = df_rank.groupby('cluster').cumcount()
        df_rank = df_rank.pivot(index='cluster', columns='rank', values='original_index')

        DrawResult.draw_top_n_output(
            df_rank, self.image_names, 
            src_path=self.src_path, 
            dst_path=self.dst_path
        )

        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.configs, file, sort_keys=False)

        open_directory(self.dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='user-config.yaml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    process = MainProcess(config_path=args.config)
    process.start()
