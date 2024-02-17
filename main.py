import os
import yaml
import shutil
import pandas as pd
import questionary
from termcolor import colored

import src.utils as utils
from src import ClusterAlgo, DimReducer, FeatureExtrator
from src.draw import DrawResult
from src.prompt import (directory_prompt, extraction_prompt, 
                        reduction_prompt, clustering_prompt, output_prompt)


def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


class MainProcess():

    _STEP_DESC = {
        1: "input",
        2: "extraction",
        3: "reduction",
        4: "clustering",
        5: "output"
    }
    _ALL_STEPS = len(_STEP_DESC)

    def __init__(self):
        self.configs = load_config('config.yaml')
        self.data_dir = self.configs['global_settings']['data_dir']
        self.result_dir = self.configs['global_settings']['result_dir']
        self.extractor = None
        self.step = 1

    def start(self):
        """ Start the process. """ 
        if self.step > MainProcess._ALL_STEPS: return
        match MainProcess._STEP_DESC[self.step]:
            case "input":
                self.input_step()
            case "extraction":
                self.extraction_step()
            case "reduction":
                self.reduction_step()
            case "clustering":
                self.clustering_step()
            case "output":
                self.output_step()
            case _:
                raise AttributeError("Undefined step.") 
        self.proceed()
        self.start()

    def proceed(self):
        if MainProcess._STEP_DESC[self.step] in ["extraction", "output"]:
            response = "Next"
        else:
            print(f"\n{MainProcess._STEP_DESC[self.step].capitalize()} step completed. " + 
                   "What would you like to do next?")
            response = questionary.select(
                "Select 'Next' to proceed, 'Repeat' to redo, or 'Back' to return to the previous step:",
                choices=['Next', 'Repeat', 'Back']
            ).ask()

        self.hline()
        if response == "Next":
            self.step += 1
        elif response == "Back":
            self.step = max(self.step-1, 1)

    def hline(self):
        print("\n============================================================================================\n")

    def input_step(self):
        if len(utils.list_all_directories(self.data_dir)) == 0:
            raise Exception("There's no available image data.")
        
        self.dirname = directory_prompt(data_dir=self.data_dir)
        self.src_path = os.path.join(self.data_dir, self.dirname)
        print(f"Input data path: {self.src_path}")

        self.dst_path = os.path.join(self.result_dir, self.dirname)
        if os.path.exists(self.dst_path):
            shutil.rmtree(self.dst_path)
        os.mkdir(self.dst_path)
        
    def extraction_step(self):
        self.backbone = extraction_prompt()
        if self.extractor is None:
            self.extractor = FeatureExtrator(
                configs=self.configs['extractor'],
                backbone=self.backbone)
        else:
            self.extractor.set_model(self.backbone)
        self.X, self.image_names = self.extractor.extract(path=self.src_path)

    def reduction_step(self):
        self.reduction_method = reduction_prompt()
        print("\nStart reducing dimensionality ... ", end="")
        self.reducer = DimReducer().set_algo(
            method=self.reduction_method, 
            configs=self.configs['reduction'])
        self.X_reduced = self.reducer.apply(self.X)
        print(colored("completed", "green"))

        self.df = pd.DataFrame(self.X_reduced)
        self.df.columns = ['x', 'y']
        DrawResult.draw_reduction(self.df, self.reduction_method)

    def clustering_step(self):
        self.cluster_method = clustering_prompt()
        print("\nStart clustering ... ", end="")
        self.cluster_algo = ClusterAlgo().set_algo(
            method=self.cluster_method, 
            configs=self.configs['clustering'])
        self.df['cluster'] = self.cluster_algo.apply(self.X_reduced)
        print(colored("completed", "green"))

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
        utils.open_directory(self.dst_path)


if __name__ == "__main__":
    process = MainProcess()
    process.start()

