import os
import sys
import copy
import plotly.graph_objs as go
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.decomposition import KernelPCA, PCA
import getpass
from pathlib import Path

N_SPLITS = 5

# TODO: Create function that returns color, either for a single one, or for a list. Might be much easier...
color_dict = {
    "Co": "navy",
    "Rh": "sienna",
    "Ir": "olive",
    "Ni": "teal",
    "Pd": "purple",
    "Pt": "green",
    "Cu": "blue",
    "Ag": "brown",
    "Au": "orange",
    "Cd": "black",
    "Os": "peachpuff",
    "Fe": "peru",
    "Zn": "salmon",
    "Ru": "palegreen",
}

# TODO: Cannot put PCA/KernelPCA here, of course, because they don't implement predict.
NEED_TO_STANDARDIZE = (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    SVR,
)

METALS = [
    "Au",
    "Ir",
    "Ni",
    "Pt",
    "Ru",
    "Ag",
    "Cd",
    "Cu",
    "Os",
    "Pd",
    "Rh",
    "Zn",
    "Co",
    "Fe",
]

CAVITIES = [
    # C (purely)
    "C_din3_c3",
    "C_din4_c4",
    "C_rol1_din2_c3",
    "C_rol2_din2_c4",
    "C_rol3",
    # n
    "N_din3",
    "N_din3_c1",
    "N_din3_c2",
    "N_din4",
    "N_din4_c1",
    "N_din4_c2_d1",
    "N_din4_c2_d2",
    "N_din4_c2_s1",
    "N_din4_c2_s2",
    "N_din4_c3",
    "N_rol1_din2",
    "N_rol1_din2_c1_a",
    "N_rol1_din2_c1_b",
    "N_rol1_din2_c2_s1",
    "N_rol1_din2_c2_s2",
    "N_rol2_din2",
    "N_rol2_din2_a",
    "N_rol2_din2_b",
    "N_rol2_din2_c2_d1",
    "N_rol2_din2_c2_s1",
    "N_rol2_din2_c2_s2",
    "N_rol2_din2_c2_s3",
    "N_rol2_din2_c3_a",
    "N_rol2_din2_c3_b",
    "N_rol3",
    "N_rol3_c1",
    "N_rol3_c2",
    # p
    "P_din3",
    "P_din3_c1",
    "P_din3_c2",
    "P_din4",
    "P_din4_c1",
    "P_din4_c2_d1",
    "P_din4_c2_d2",
    "P_din4_c2_s1",
    "P_din4_c2_s2",
    "P_din4_c3",
    "P_rol1_din2",
    "P_rol1_din2_c1_a",
    "P_rol1_din2_c1_b",
    "P_rol1_din2_c2_s1",
    "P_rol1_din2_c2_s2",
    "P_rol2_din2",
    "P_rol2_din2_a",
    "P_rol2_din2_b",
    "P_rol2_din2_c2_d1",
    "P_rol2_din2_c2_s1",
    "P_rol2_din2_c2_s2",
    "P_rol2_din2_c2_s3",
    "P_rol2_din2_c3_a",
    "P_rol2_din2_c3_b",
    "P_rol3",
    "P_rol3_c1",
    "P_rol3_c2",
    # s
    "S_din3",
    "S_din3_c1",
    "S_din3_c2",
    "S_din4",
    "S_din4_c1",
    "S_din4_c2_d1",
    "S_din4_c2_d2",
    "S_din4_c2_s1",
    "S_din4_c2_s2",
    "S_din4_c3",
    "S_rol1_din2",
    "S_rol1_din2_c1_a",
    "S_rol1_din2_c1_b",
    "S_rol1_din2_c2_s1",
    "S_rol1_din2_c2_s2",
    "S_rol2_din2",
    "S_rol2_din2_a",
    "S_rol2_din2_b",
    "S_rol2_din2_c2_d1",
    "S_rol2_din2_c2_s1",
    "S_rol2_din2_c2_s2",
    "S_rol2_din2_c2_s3",
    "S_rol2_din2_c3_a",
    "S_rol2_din2_c3_b",
    "S_rol3",
    "S_rol3_c1",
    "S_rol3_c2",
]

ridge_dict = {
    "alpha": 1.0,
    "fit_intercept": True,
    "normalize": False,
    "copy_X": True,
    "max_iter": int(float("1e5")),
    "tol": float("1e-3"),
    "solver": "auto",
    "random_state": 0,
}

lasso_dict = {
    "alpha": float("1e-5"),
    "fit_intercept": True,
    "normalize": False,
    "precompute": False,
    "copy_X": True,
    "max_iter": int(float("1e5")),
    "tol": float("1e-3"),
    "warm_start": False,
    "positive": False,
    "random_state": 0,
    "selection": "cyclic",
}

en_dict = {
    "alpha": float("1e-3"),
    "copy_X": True,
    "fit_intercept": True,
    "l1_ratio": 1 - float("1e-3"),
    "max_iter": int(float("1e6")),
    "normalize": False,
    "positive": False,
    "precompute": True,
    "random_state": 0,
    "selection": "cyclic",
    "tol": float("1e-3"),
    "warm_start": True,
}

rf_dict = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "criterion": "squared_error",
    "max_depth": 8,
    "max_features": 0.4,
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    # "min_impurity_split": None,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 128,
    "n_jobs": -1,
    "oob_score": False,
    "random_state": 0,
    "verbose": 0,
    "warm_start": False,
}

svr_dict = {
    "C": 1.0,
    "cache_size": 200,
    "coef0": 0.0,
    "degree": 3,
    "epsilon": 0.1,
    "gamma": "scale",
    "kernel": "rbf",
    "max_iter": -1,
    "shrinking": True,
    "tol": 0.001,
    "verbose": False,
}

# Default figure layout for energy-figures
energy_layout = go.Layout(
    # Update global layout
    width=600,
    height=600,
    font=dict(family="Arial", color="black", size=26),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    hoverlabel={"namelength": -1},
    # title=dict(text=plot_title, x=0.5, ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(
        xanchor="right",
        x=1,
        yanchor="bottom",
        y=0,
        bgcolor="rgba(0,0,0,0.1)",  # bordercolor='rgba(0,0,0,0.4)',
        font_size=26,
        tracegroupgap=2,
    ),
    xaxis=dict(
        title="E<sub>DFT</sub> / eV",
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickfont_size=26,
        tickformat=".1f",
        tickwidth=3,
        ticklen=6,
    ),
    yaxis=dict(
        title="E<sub>pred</sub> / eV",
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        ticks="outside",
        tickfont_size=26,
        tickformat=".1f",
        tickwidth=3,
        ticklen=6,
    ),
)

pca_layout = go.Layout(
    width=600,
    height=600,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=50,
    ),
    font=dict(size=20, family="Arial", color="black"),
    hoverlabel={"namelength": -1},
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(
        title="PC 1",
        tickfont_size=20,
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="black",
    ),
    yaxis=dict(
        title="PC 2",
        tickfont_size=20,
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="black",
    ),
)

pca_weight_layout = go.Layout(
    width=1000,
    height=800,
    font=dict(size=14, family="Arial", color="black"),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis=dict(
        showline=True,
        mirror=True,
        showgrid=False,
        zeroline=True,
        showticklabels=False,
        linecolor="black",
    ),
    yaxis=dict(
        showline=True,
        mirror=True,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        linecolor="black",
    ),
    xaxis2=dict(
        showline=True,
        mirror=True,
        showgrid=False,
        zeroline=False,
        tickfont_size=16,
        linecolor="black",
    ),
    yaxis2=dict(
        showline=True,
        mirror=True,
        showgrid=False,
        zeroline=False,
        tickfont_size=11,
        linecolor="black",
    ),
)

pca_transform_layout = go.Layout(
    width=1000,
    height=800,
    font=dict(size=20, family="Arial", color="black"),
    margin=go.layout.Margin(
        l=0,
        r=0,
        b=0,
        t=50,
    ),
    hoverlabel={"namelength": -1},
    title=dict(
        text="Data projection along PC1 and PC2",
        x=0.5,
    ),
    legend=dict(
        orientation="h",
        bgcolor="rgba(0,0,0,0.1)",
        xanchor="left",
        x=0,
        yanchor="top",
        y=1,
    ),
    # Add ranges here
    xaxis=dict(
        title="PC 1",
        tickfont_size=20,
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="black",
    ),
    yaxis=dict(
        title="PC 2",
        tickfont_size=20,
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linecolor="black",
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

# proj_layout = go.Layout(
#     width=900,
#     height=800,  # font=dict(size=20, family='Arial', color='black'),
#     margin=go.layout.Margin(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     xaxis=dict(title="PC1"),
#     yaxis=dict(title="PC2"),
# )

# coeff_bar_layout = go.Layout(
#     width=500,
#     height=520,
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         range=[0, 1],
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,  # autorange='reversed',
#         ticks="outside",
#         tickfont_size=24,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=False,
#         showgrid=False,
#         zeroline=False,
#         autorange="reversed",
#         ticks="outside",
#         tickfont_size=24,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# error_bar_layout = go.Layout(
#     width=800,
#     height=300,
#     barmode="group",
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     showlegend=False,
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# main_bar_layout = go.Layout(
#     width=500,
#     height=520,
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=False,
#         linewidth=3,
#         linecolor="black",
#         mirror=False,  # range=[0, 1],
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,  # autorange='reversed',
#         ticks="outside",
#         tickfont_size=24,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=False,
#         linewidth=3,
#         linecolor="black",
#         mirror=False,
#         showgrid=False,
#         zeroline=False,
#         autorange="reversed",
#         ticks="outside",
#         tickfont_size=24,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# si_bar_layout = go.Layout(
#     width=500,
#     height=800,
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,  # autorange='reversed',
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# en_parsim_layout = go.Layout(
#     # Update global layout
#     width=600,
#     height=600,
#     font=dict(family="Arial", color="black", size=26),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     # title=dict(text=plot_title, x=0.5, ),
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",  # bordercolor='rgba(0,0,0,0.4)',
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         title="E<sub>DFT</sub> / eV",
#         title_font_size=30,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=26,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
#     yaxis=dict(
#         title="E<sub>EN</sub> / eV",
#         title_font_size=30,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=26,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# rf_parsim_layout = go.Layout(
#     # Update global layout
#     width=600,
#     height=600,
#     font=dict(family="Arial", color="black", size=26),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     # title=dict(text=plot_title, x=0.5, ),
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",  # bordercolor='rgba(0,0,0,0.4)',
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         title="E<sub>DFT</sub> / eV",
#         title_font_size=30,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=26,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
#     yaxis=dict(
#         title="E<sub>RF</sub> / eV",
#         title_font_size=30,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=26,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# sfs_layout = go.Layout(
#     width=800,
#     height=250,
#     hovermode="x unified",
#     font=dict(family="Arial", color="black"),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         font_size=24,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         title_font_size=36,
#         showticklabels=True,
#         range=[0, 31],
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="",
#         tickfont_size=32,
#         tickwidth=3,
#         ticklen=6,
#         hoverformat="d",
#     ),
#     yaxis=dict(
#         title_font_size=36,
#         showticklabels=True,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="",
#         tickfont_size=32,
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
# )

# layout_h = go.Layout(
#     width=300,
#     height=800,
#     barmode="group",
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     showlegend=False,
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         range=[0, 1],
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,  # autorange='reversed',
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )

# layout_v = go.Layout(
#     width=800,
#     height=300,
#     barmode="group",
#     font=dict(family="Arial", color="black", size=20),
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0,
#     ),
#     hoverlabel={"namelength": -1},
#     paper_bgcolor="white",
#     plot_bgcolor="white",
#     showlegend=False,
#     legend=dict(
#         xanchor="right",
#         x=1,
#         yanchor="bottom",
#         y=0,
#         bgcolor="rgba(0,0,0,0.1)",
#         font_size=26,
#         tracegroupgap=2,
#     ),
#     xaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         tickangle=0,  # autorange='reversed',
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#         hoverformat=".3f",
#     ),
#     yaxis=dict(
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         range=[0, 1],
#         showgrid=False,
#         zeroline=False,
#         ticks="outside",
#         tickfont_size=20,
#         tickformat=".1f",
#         tickwidth=3,
#         ticklen=6,
#     ),
# )
