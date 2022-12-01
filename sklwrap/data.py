import plotly.graph_objs as go
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR

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

# # Default figure layout for energy-figures
# energy_layout = go.Layout(
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
#         title="E<sub>pred</sub> / eV",
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


regr_layout = go.Layout(
    width=597,
    height=597,
    font=dict(family="Arial", color="black"),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
    ),
    hoverlabel={"namelength": -1},
    paper_bgcolor="white",
    plot_bgcolor="white",
    legend=dict(
        xanchor="right",
        x=1,
        yanchor="bottom",
        y=0,
        bgcolor="rgba(0,0,0,0.1)",
        font_size=26,
        tracegroupgap=2,
    ),
    xaxis=dict(
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
        ticks="outside",
        tickfont_size=26,
        tickformat=".1f",
        tickwidth=3,
        ticklen=6,
    ),
    yaxis=dict(
        title_font_size=30,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
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
