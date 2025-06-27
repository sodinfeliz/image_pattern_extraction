from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore  # (plotly-stubs only supports Python 3.10+)
from plotly.subplots import make_subplots  # type: ignore


class DrawResult:

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

        For the features more than 2, the first two features are used.

        Args:
            df (pd.DataFrame): dataframe with reduced features
            reduction_method (str): name of the reduction method
        """
        trace0 = go.Scatter(
            x=df["x1"],
            y=df["x2"],
            mode="markers",
            marker=dict(size=4),
            customdata=df.index,
            hovertemplate="Index: %{customdata}",
        )

        layout = go.Layout(
            xaxis=dict(title=""),
            template=cls.PLOTLY_THEME,
            width=700,
            height=600,
            title={
                "text": f"{reduction_method} w/o Clustering",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24},
            },
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
        cluster_method: str,
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
            x=df_filter["x1"],
            y=df_filter["x2"],
            name="cluster",
            mode="markers",
            marker=dict(
                size=4,
                color=df_filter["cluster"],
                colorscale="Viridis",
            ),
            customdata=df_filter.index,
            hovertemplate="Index: %{customdata}",
        )

        trace1 = go.Scatter(
            x=df_mean["mean_x1"],
            y=df_mean["mean_x2"],
            name="mean",
            mode="markers+text",
            marker=dict(
                size=7,
                color="red",
                symbol="x",
            ),
            text=df_mean.index.astype(str),
            textposition="top center",
            textfont=dict(
                size=15,
            ),
            customdata=df_mean.index,
            hovertemplate="<br>".join(
                ["Mean X: %{x}", "Mean Y: %{y}", "Cluster: %{customdata}"]
            ),
        )

        layout = go.Layout(
            xaxis=dict(title=""),
            template=cls.PLOTLY_THEME,
            width=700,
            height=600,
            title={
                "text": f"{reduction_method} with {cluster_method}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 24},
            },
        )

        fig = go.Figure(data=[trace0, trace1], layout=layout)
        fig.show()

    @classmethod
    def draw_top_n_output(
        cls,
        df_rank: pd.DataFrame,
        *,
        image_paths: List[Path],
        dst_path: Path,
        summary_count: int = 5,
    ):
        """
        Draw the top 'summary_count' images nearest to the cluster's mean.
        Then save the output image to the destination path.

        Args:
            df_rank (pd.DataFrame): rank of the top n images
            image_paths (list[Path]): list of image paths
            dst_path (Path): destination path of the output image
            summary_count (int): number of images to be shown
        """
        r, c = len(df_rank), len(df_rank.columns)
        c = min(c, summary_count)

        fig_im = make_subplots(rows=r, cols=c)
        for i in range(r):
            for j in range(c):
                index = df_rank.loc[i][j]
                if np.isnan(index):
                    im = np.empty((0, 0))
                else:
                    im_path = image_paths[int(index)]
                    im = cls.load_image_to_fixed_width(str(im_path))[..., ::-1]
                fig_im.add_trace(go.Image(z=im), row=i + 1, col=j + 1)

        fig_im.update_layout(
            width=c * 200,
            height=r * 150 + 100,
            template=cls.PLOTLY_THEME,
            title={
                "text": f"Top {c} Images Nearest to the Cluster's Mean",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
        )

        fig_im.update_xaxes(showticklabels=False)
        fig_im.update_yaxes(showticklabels=False)
        fig_im.write_image(dst_path / f"top_{c}_samples.png")

    @staticmethod
    def load_image_to_fixed_width(path: str, maximum_width: int = 256):
        im = cv2.imread(path)
        if im.shape[1] > maximum_width:
            ratio = maximum_width / im.shape[1]
            im = cv2.resize(im, (maximum_width, int(ratio * im.shape[0])))
        return im
