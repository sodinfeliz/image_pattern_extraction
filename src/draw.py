import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_image_to_fixed_width(path: str, width: int=256):
    im = cv2.imread(path)
    ratio = width / im.shape[1]
    resized_im = cv2.resize(im, (width, int(ratio * im.shape[0])))
    return resized_im


class DrawResult():
    
    PLOTLY_THEME: str = "plotly_dark"

    @classmethod
    def draw_reduction(
        cls,
        df: pd.DataFrame,
        *,
        reduction_method: str,
    ) -> None:
        """ 
        Draw the reduction result, and show on the browser.
        
        Args:
            df (pd.DataFrame): dataframe with reduced features
            reduction_method (str): name of the reduction method
        """
        trace0 = go.Scatter(
            x=df['x'], 
            y=df['y'],
            mode='markers',
            marker=dict(size=4),
            customdata=df.index,
            hovertemplate="Index: %{customdata}"
        )

        layout = go.Layout(
            xaxis=dict(title=''),
            template=cls.PLOTLY_THEME,
            width=700, height=600,
            title={
                'text': f'{reduction_method} w/o Clustering',
                'x': 0.5, 'xanchor': 'center',
                'font': {'size': 24}
            }
        )

        fig = go.Figure(data=[trace0], layout=layout)
        fig.show()

    @classmethod
    def draw_clustering(
        cls,
        df_filter: pd.DataFrame,
        df_mean: pd.DataFrame,
        *,
        reduction_method: str,
        cluster_method: str
    ):
        """ 
        Draw the clustering result, and show on the browser.
        
        Args:
            df_filter (pd.DataFrame): dataframe with cluster labels
            df_mean (pd.DataFrame): dataframe with cluster means
            reduction_method (str): name of the reduction method
            cluster_method (str): name of the clustering method
        """
        trace0 = go.Scatter(
            x=df_filter['x'], 
            y=df_filter['y'],
            name='cluster',
            mode='markers',
            marker=dict(
                size=4,
                color=df_filter['cluster'],
                colorscale='Viridis',
            ),
            customdata=df_filter.index,
            hovertemplate="Index: %{customdata}"
        )

        trace1 = go.Scatter(
            x=df_mean['mean_x'], 
            y=df_mean['mean_y'],
            name='mean',
            mode='markers+text',
            marker=dict(
                size=7,
                color='red',
                symbol='x',
            ),
            text=df_mean.index.astype(str),
            textposition='top center',
            textfont=dict(
                size=15,
            ),
            customdata=df_mean.index,
            hovertemplate="<br>".join([
                "Mean X: %{x}",
                "Mean Y: %{y}",
                "Cluster: %{customdata}"
            ])
        )

        layout = go.Layout(
            xaxis=dict(title=''),
            template=cls.PLOTLY_THEME,
            width=700, height=600,
            title={
                'text': f'{reduction_method} with {cluster_method}',
                'x': 0.5, 'xanchor': 'center',
                'font': {'size': 24}
            }
        )

        fig = go.Figure(data=[trace0, trace1], layout=layout)
        fig.show()

    @classmethod
    def draw_top_n_output(
        cls,
        df_rank: pd.DataFrame,
        *,
        image_names: list,
        src_path: str,
        dst_path: str,
        summary_count: int=5,
    ):
        """ 
        Draw the top 'summary_count' images nearest to the cluster's mean.
        Then save the output image to the destination path.
        
        Args:
            df_rank (pd.DataFrame): rank of the top n images
            image_names (list): list of image names
            src_path (str): source path of the images
            dst_path (str): destination path of the output image
        """
        r, c = len(df_rank), len(df_rank.columns)
        c = min(c, summary_count)

        fig_im = make_subplots(rows=r, cols=c)
        for i in range(r):
            for j in range(c):
                index = df_rank.loc[i][j]
                if np.isnan(index):
                    im = np.empty((0,0))
                else:
                    im_path = os.path.join(src_path, image_names[int(index)])
                    im = load_image_to_fixed_width(im_path)[..., ::-1]
                fig_im.add_trace(
                    go.Image(z=im),
                    row=i+1, col=j+1
                )
        
        fig_im.update_layout(
            width=c * 200, 
            height=r*150 + 100,
            template=cls.PLOTLY_THEME,
            title={
                'text': f"Top {c} Images Nearest to the Cluster's Mean",
                'x': 0.5, 'xanchor': 'center',
                'font': {'size': 20}
            },
        )

        fig_im.update_xaxes(showticklabels=False)
        fig_im.update_yaxes(showticklabels=False)
        fig_im.write_image(os.path.join(dst_path, f'top_{c}_samples.png'))
