import os
import yaml
import cv2
import shutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import ClusterAlgo, DimReducer, FeatureExtrator


def load_config(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_image_to_fixed_width(path: str, width: int=256):
    im = cv2.imread(path)
    ratio = width / im.shape[1]
    resized_im = cv2.resize(im, (width, int(ratio * im.shape[0])))
    return resized_im


def main(dest_dir: str):
    configs = load_config('config.yaml')

    data_dir = configs['global_settings']['data_dir']
    result_dir = configs['global_settings']['result_dir']
    dst_path = os.path.join(result_dir, dest_dir)
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)

    if not os.path.exists((src_path := os.path.join(data_dir, dest_dir))):
        raise FileExistsError(f"Can't find {src_path}.")

    backbone = "resnet"
    reduction_method = "UMAP"
    cluster_method = "DBSCAN"

    
    extractor = FeatureExtrator(configs=configs['extractor'])
    X, images = extractor.extract(path=src_path)

    reducer = DimReducer().set_algo(reduction_method, configs['reduction'])
    X_reduced = reducer.apply(X)
    cluster_algo = ClusterAlgo().set_algo(method=cluster_method, configs=configs['clustering'])
    labels = cluster_algo.apply(X_reduced)
    
    ##### TEMP
    X_reduced = pd.DataFrame(X_reduced)
    X_reduced.columns = ['x', 'y']
    X_reduced['cluster'] = labels

    df_mean = X_reduced.groupby('cluster').mean()
    df_mean.columns = ['mean_x', 'mean_y']

    df_outlier = None
    if cluster_method == 'DBSCAN' and -1 in df_mean.index:
        df_outlier = X_reduced[X_reduced['cluster'] == -1]
        df_filter = X_reduced[X_reduced['cluster'] != -1]
        df_mean.drop(-1, inplace=True)
    else:
        df_filter = X_reduced.copy()
    
    df_merge = pd.merge(df_filter, df_mean, how='inner', on='cluster')
    df_merge['dist_square'] = (df_merge['x'] - df_merge['mean_x'])**2 + \
                              (df_merge['y'] - df_merge['mean_y'])**2
    df_merge.index = df_filter.index

    n_smallest = 5
    n_smallest = int(min(df_merge.groupby('cluster')['x'].count().min(), n_smallest))

    df_sorted = df_merge.reset_index().rename(columns={'index': 'original_index'})
    df_sorted = df_sorted.sort_values(['cluster', 'dist_square'])
    df_sorted = df_sorted.groupby('cluster').head(n_smallest)
    df_sorted['rank'] = df_sorted.groupby('cluster').cumcount()

    result = df_sorted.pivot(index='cluster', columns='rank', values='original_index')
    r, c = len(result), n_smallest
    fig_im = make_subplots(rows=r, cols=c)

    for i in range(r):
        for j in range(c):
            index = result.loc[i][j]
            if np.isnan(index):
                im = np.empty((0,0))
            else:
                im_path = os.path.join(src_path, images[int(index)])
                im = load_image_to_fixed_width(im_path)[..., ::-1]
            fig_im.add_trace(
                go.Image(z=im),
                row=i+1, col=j+1
            )

    fig_im.update_layout(
        width=n_smallest * 200, 
        height=r*150 + 100,
        template='plotly',
        title={
            'text': f"Top {n_smallest} Images Nearest to the Cluster's Mean",
            'x': 0.5, 'xanchor': 'center',
            'font': {'size': 20}
        },
    )

    fig_im.update_xaxes(showticklabels=False)
    fig_im.update_yaxes(showticklabels=False)
    fig_im.write_image(os.path.join(dst_path, f'top_{n_smallest}_samples.png'))


if __name__ == "__main__":
    dest_dir = '4DM24DK1B84_9CM112046N10_UniformLight'
    main(dest_dir)

