import copy
import itertools as it
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .data import *


def plotly_to_image(
    plotly_fig: go.Figure,
    path_elements: list[str],
    figure_name: str,
    save_types: list = ["png", "svg", "html"],
    paper: bool = False,
):
    """This function takes a plotly figure object and writes it as svg, png, and html files to disk. In particular, it takes care of removing the `non-scaling-stroke` vector-effect in the svg, and, if need be, creates an svg without annotations for postprocessing in Inkscape.

    Args:
        plotly_fig (go.Figure): Plotly.go Figure object to be saved.
        path_elements (list[str]): List of path elements for the save location of the Figure.
        figure_name (str): Base filename for the saved Figure.
        save_types (list, optional): Which formats shall be saved. Defaults to ["png", "svg", "html"].
        paper (bool, optional): If an additional version of the svg without ticks or annotations should be saved. Defaults to False.

    Returns:
        None
    """

    # ! This could also be done outside...
    main_path = os.path.join(*path_elements)

    for save_type in save_types:
        if save_type == "svg":
            # ! All the extra work for svgs to remove vector-effect, and create figures without annotations for the paper.
            svg_filename = os.path.join(main_path, "{}.svg".format(figure_name))
            # str.replace() returns a new string, old string is not changed.
            temp_filename = svg_filename.replace(".svg", "_temp.svg")

            plotly_fig.write_image(temp_filename, engine="kaleido")
            with open(temp_filename, "rt") as fin:
                with open(svg_filename, "wt") as fout:
                    for line in fin:
                        fout.write(
                            line.replace(
                                "vector-effect: non-scaling-stroke",
                                "vector-effect: none",
                            )
                        )
            os.remove(temp_filename)

            if paper is True:
                svg_filename = svg_filename.replace(".svg", "_paper.svg")
                paper_plotly_fig = copy.deepcopy(plotly_fig)
                for anno in paper_plotly_fig["layout"]["annotations"]:
                    anno["text"] = ""
                paper_layout = go.Layout(
                    xaxis=dict(ticks="", showticklabels=False, showgrid=False),
                    yaxis=dict(ticks="", showticklabels=False, showgrid=False),
                    title_text="",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                _ = paper_plotly_fig.update_layout(paper_layout)

                paper_plotly_fig.write_image(temp_filename, engine="kaleido")

                with open(temp_filename, "rt") as fin:
                    with open(svg_filename, "wt") as fout:
                        for line in fin:
                            fout.write(
                                line.replace(
                                    "vector-effect: non-scaling-stroke",
                                    "vector-effect: none",
                                )
                            )
                os.remove(temp_filename)

        # Here, we still create it from path_elements, as defining main_path does not provide much gain/advantage.
        elif save_type == "html":
            plotly_fig.write_html(
                os.path.join(main_path, "{}.{}".format(figure_name, save_type))
            )

        else:  # png or pdf
            plotly_fig.write_image(
                os.path.join(main_path, "{}.{}".format(figure_name, save_type)),
                engine="kaleido",
            )

    return None


def plot_regr(
    regr_dict,
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    color_setup=None,  # This should be a dict with the key being the column name and the value being a color mapping:
    regr_layout=None,
    *args,
    **kwargs,
):

    # function_arguments = locals()
    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # Include threshold for deviation somewhere
    # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
    # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
    # TODO: All plotly layout options using kwargs: `showticklabels`, `showlegend`, `label_column`

    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # ! At least color column needs to be set to something for the loop over the values later on.
    # ? Currently, using kwargs or the first column of the df to get a column to do the looping over.
    if color_setup is None:
        color_setup = {kwargs.get("split_col", df_func.columns[0]): {}}

    color_column = list(color_setup.keys())[0]
    color_mapping = list(color_setup.values())[0]

    # ! Sort df so that legend items appear ordered -> Check that this does not mess up the ordering from input to pred values.
    df_func = df_func.sort_values(by=color_column)

    regr_figs = []

    bool_columns = [col for col in df_func.columns if "train" in col]
    pred_columns = [col for col in df_func.columns if "pred" in col]

    for cv_id in range(len(error_dict["rmse_trains"])):

        cv_id_bool_column = bool_columns[int(cv_id)]
        cv_id_pred_column = pred_columns[int(cv_id)]

        split_bool_array = df_func[cv_id_bool_column].values

        df_train = df_func[split_bool_array].copy(deep=True)
        df_test = df_func[np.logical_not(split_bool_array)].copy(deep=True)

        # Instantiate figure
        regr_fig = go.Figure()

        # Add annotation with R^2 and RMSEs
        # TODO: Implement this better

        # region
        if which_error == "mean":
            error_list = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]
        elif which_error == "best":
            error_list = [
                np.max(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.min(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.min(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]
        elif which_error == "full_mean":
            error_list = [
                np.mean(error_dict["rsquared_fulls"]),
                np.std(error_dict["rsquared_fulls"]),
                np.mean(error_dict["rmse_fulls"]),
                np.std(error_dict["rmse_fulls"]),
                np.mean(error_dict["mae_fulls"]),
                np.std(error_dict["mae_fulls"]),
            ]
        elif which_error == "full_best":
            error_list = [
                np.max(error_dict["rsquared_fulls"]),
                np.std(error_dict["rsquared_fulls"]),
                np.min(error_dict["rmse_fulls"]),
                np.std(error_dict["rmse_fulls"]),
                np.min(error_dict["mae_fulls"]),
                np.std(error_dict["mae_fulls"]),
            ]
        else:
            raise KeyError("Invalid error specification used.")
        # endregion

        error_text = "R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
            *error_list
        )

        _ = regr_fig.add_annotation(
            xanchor="left",
            yanchor="top",
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            align="left",
            text=error_text,
            font_size=26,
            font_family="Arial",
            showarrow=False,
            bgcolor="rgba(0,0,0,0.1)",
        )
        # region

        # Plot energy data points
        # ! Difference between train and test via marker symbol
        # ! Different colors based on a column

        # print("color_column", color_column)
        if show_train is True:
            for column_value in df_train[color_column].unique():
                column_train_df = df_train.loc[df_train[color_column] == column_value]
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_train_df["y"],
                        y=column_train_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=8,
                            symbol=0,
                            opacity=1,
                            color=color_mapping.get(column_value, "black"),
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        # text=train_plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_train_legend", True),
                    ),
                )

        if show_test is True:
            for column_value in df_test[color_column].unique():
                column_test_df = df_test.loc[df_test[color_column] == column_value]
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_test_df["y"],
                        y=column_test_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=8,
                            symbol=4,
                            opacity=1,
                            color=color_mapping.get(column_value, "black"),
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        # text=test_plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_test_legend", True),
                    ),
                )

        # endregion

        if set_range is None:
            all_values = (
                df_train["y"].tolist()
                + df_train[cv_id_pred_column].tolist()
                + df_test["y"].tolist()
                + df_test[cv_id_pred_column].tolist()
            )
            all_values = list(map(float, all_values))

            full_range = [min(all_values), max(all_values)]
            range_ext = (
                full_range[0] - 0.075 * np.ptp(full_range),
                full_range[1] + 0.075 * np.ptp(full_range),
            )
        else:
            range_ext = set_range

        # Add ideal fit line to plot
        _ = regr_fig.add_trace(
            go.Scatter(
                x=range_ext,
                y=range_ext,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
        )

        # Update global layout

        if regr_layout is not None:
            _ = regr_fig.update_layout(regr_layout)

        axes_layout = go.Layout(
            xaxis=dict(range=range_ext), yaxis=dict(range=range_ext)
        )
        _ = regr_fig.update_layout(axes_layout)

        regr_figs.append(regr_fig)

    return regr_figs


def plot_errors(
    error_dict, x_values, plot_measures, annot_text, x_title, showlegend=True
):
    """
    Plot RMSE (train and test) and R^2 for ML model from the result of varying a parameter.
    """

    annot_texts = ["<br>".join(sorted(annot_text)) for annot_text in annot_text]

    error_fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces for error measures.
    for iplot_measure, plot_measure in enumerate(plot_measures):
        if "rmse" in plot_measure:
            line_color = "blue"
            line_name = "RMSE ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y1"
        elif "mae" in plot_measure:
            line_color = "green"
            line_name = "MAE ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y1"
        elif "rsquared" in plot_measure:
            line_color = "red"
            line_name = "R<sup>2</sup> ({})".format(plot_measure.split("_")[1][:-1])
            y_axis = "y2"
        if "test" in plot_measure:
            dash = "dash"
        else:
            dash = None

        _ = error_fig.add_trace(
            go.Scatter(
                x=list(x_values),
                y=error_dict[
                    plot_measure
                ],  # [round(_, 2) for _ in error_dict[plot_measure]] # TODO: Round numbers shown on hover (but not for plotting).
                mode="lines",
                name=line_name,
                line=dict(
                    color=line_color,
                    width=3,
                    dash=dash,
                ),
                showlegend=showlegend,
                hoverinfo="x+y",
                yaxis=y_axis,
            ),
            secondary_y="2" in y_axis,
        )

        if min(x_values) < 0.05:
            x_range = [0, max(x_values)]
        else:
            x_range = [min(x_values), max(x_values)]

        # Add invisible trace to allow for dynamic annotation of descriptors
        if iplot_measure == len(plot_measures) - 1:
            _ = error_fig.add_trace(
                go.Scatter(
                    # x=list(range(1, error_array.shape[0])),
                    x=list(x_values),
                    y=error_dict[plot_measure],
                    mode="lines",
                    name="RMSE (train)",
                    line=dict(color="blue", width=0.001),
                    showlegend=False,
                    text=annot_texts,
                    hoverinfo="text",
                ),
            )

    error_layout = go.Layout(
        width=809,
        height=500,
        font=dict(family="Arial", color="black"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            xanchor="center",
            x=0.5,
            yanchor="top",
            y=1,
            font_size=26,
            bgcolor="rgba(0,0,0,0.1)",
        ),
        hoverlabel={"namelength": -1},
        hovermode="x unified",
        xaxis=dict(
            title=x_title,
            title_font_size=30,
            range=x_range,
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
        yaxis=dict(
            title="RMSE / eV",
            title_font_size=30,
            title_font_color="blue",  # range=[0, rmse_max*1.25],
            showline=True,
            linewidth=3,
            linecolor="black",
            color="blue",  # mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",  # tick0=0, dtick=0.1,
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
        yaxis2=dict(
            title="R<sup>2</sup>",
            title_font_size=30,
            title_font_color="red",  # range=[0, 1],
            showline=True,
            linewidth=3,
            linecolor="black",
            color="red",  # mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",  # tick0=0, dtick=0.2,
            ticks="outside",
            tickfont_size=26,
            tickwidth=3,
            ticklen=6,
        ),
    )

    _ = error_fig.update_layout(error_layout)

    return error_fig
