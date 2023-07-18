import copy
import os
from typing import List

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from .data import *

# Get all the named CSS colors
# named_colors = [
#     "#1f77b4",  # muted blue
#     "#ff7f0e",  # safety orange
#     "#2ca02c",  # cooked asparagus green
#     "#d62728",  # brick red
#     "#9467bd",  # muted purple
#     "#8c564b",  # chestnut brown
#     "#e377c2",  # raspberry yogurt pink
#     "#7f7f7f",  # middle gray
#     "#bcbd22",  # curry yellow-green
#     "#17becf",  # blue-teal
# ] * 10

# Light24 Palette

# named_colors = [
#     '#1f77b4',
#     '#ff7f0e',
#     '#2ca02c',
#     '#d62728',
#     '#9467bd'
# ]

named_colors = [
    '#FD3216',
    '#00FE35',
    '#6A76FC',
    '#FED4C4',
    '#FE00CE', 
    '#0DF9FF',
    '#F6F926',
    '#FF9616',
    '#479B55',
    '#EEA6FB',
    '#DC587D',
    '#D626FF',
    '#6E899C',
    '#00B5F7',
] * 10
default_blue = "#636EFA"
# named_colors.extend(
#     [_ for _ in list(webcolors.CSS3_NAMES_TO_HEX.keys()) if "white" not in _]
# )
# print(named_colors)


named_symbols = [
        'circle',
        'square',
        'diamond',
        'cross',
        'pentagon'
]


# # G10 color pallete
# named_colors = [
#  '#0099C6',
#  '#DD4477', 
#  '#66AA00', 
#  '#B82E2E', 
#  '#316395',
# ] * 10

def plotly_to_image(
    plotly_fig: go.Figure,
    path_elements: List[str],
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
    color_column=None,  # ! Make None as default, and possible. Currently breaks
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    color_mapping: dict = {},
    regr_layout=None,
    text_column=None,
    *args,
    **kwargs,
):
    """
    Generate scatter plots for regression results.

    Args:
        regr_dict (dict): A dictionary containing regression results and data.
        color_column (str): The name of the column in the dataset to use for coloring the data points.
        show_train (bool, optional): Whether to show the training data points. Defaults to True.
        show_test (bool, optional): Whether to show the test data points. Defaults to True.
        set_range (tuple, optional): The range of values to be displayed on the x and y axes. If None, the range is determined automatically. Defaults to None.
        which_error (str, optional): The type of error to display in the annotation. Possible values: "mean", "best", "full_mean", "full_best", "all". Defaults to "mean".
        color_mapping (dict, optional): A dictionary mapping column values to color names or hex values. If None, default colors are used. Defaults to None.
        regr_layout (dict, optional): Additional layout options for the plotly figure. Defaults to None.
        text_column (str, optional): The name of the column in the dataset to use for text labels on the data points. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of plotly figures representing the regression scatter plots.

    Raises:
        KeyError: If an invalid error specification is used for `which_error`.

    Notes:
        - The `regr_dict` parameter should contain the following keys: 'error_dict', 'df_in'.
        - The 'error_dict' should contain error metrics for train and test sets, e.g., 'rsquared_trains', 'rsquared_tests', 'rmse_trains', 'rmse_tests', etc.
        - The 'df_in' should contain the dataset used for regression, including the columns specified in `color_column` and `text_column` if applicable.
    """
    # function_arguments = locals()

    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # Include threshold for deviation somewhere
    # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
    # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
    # TODO: All plotly layout options using kwargs: `showticklabels`, `showlegend`, `label_column`

    # # Print the color names and their corresponding hex values
    # for color_name, hex_value in named_colors.items():
    #     print(f"{color_name}: {hex_value}")
    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # if color_setup is None:
    #     text_colum
    #     color_setup = {}

    # ! At least color column needs to be set to something for the loop over the values later on.
    # ? Currently, using kwargs or the first column of the df to get a column to do the looping over.

    # color_setup = {k:v for k,v in }
    # print(color_setup)

    # color_column = list(color_setup.keys())[0]
    # color_mapping = list(color_setup.values())[0]

    # ! Sort df so that legend items appear ordered -> Check that this does not mess up the ordering from input to pred values.
    if color_column is None:
        # ! Cheap hack so that the column with the least number of values is used.
        color_column = get_color_column(df_func)

    # TODO: Make verbose here to print if
    df_func = df_func.dropna(subset=[color_column])
    color_column_values = sorted(list(set(list(df_func[color_column].values))))
    if color_mapping is None:
        color_mapping = {
            k: v
            for k, v in list(
                zip(color_column_values, named_colors[0 : len(color_column_values)])
            )
        }

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
        if which_error != "all":
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
        # ! Change decimal to 3 
        else:
            error_list_train = [
                np.mean(error_dict["rsquared_trains"]),
                np.std(error_dict["rsquared_trains"]),
                np.mean(error_dict["rmse_trains"]),
                np.std(error_dict["rmse_trains"]),
                np.mean(error_dict["mae_trains"]),
                np.std(error_dict["mae_trains"]),
            ]
            error_text = "Train:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
                *error_list_train
            )
            error_list_test = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]

            error_text += "<br>Test:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
                *error_list_test
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
                if text_column is not None:
                    plot_text = column_train_df[text_column]
                else:
                    plot_text = [""] * column_train_df.shape[0]
                ###
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_train_df["y"],
                        y=column_train_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=8,
                            symbol=0,
                            opacity=1,
                            color=color_mapping.get(column_value, default_blue),
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_train_legend", True),
                    ),
                )

        if show_test is True:
            for column_value in df_test[color_column].unique():
                column_test_df = df_test.loc[df_test[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_test_df[text_column]
                else:
                    plot_text = "" * column_test_df.shape[0]
                ###
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
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_test_legend", False),
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



def plot_regr_color_symbol(
    regr_dict,
    color_column=None,
    symbol_column=None,
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    color_mapping=None,
    regr_layout=None,
    text_column=None,
    *args,
    **kwargs,
):
    """
    Generate scatter plots for regression results.

    Args:
        regr_dict (dict): A dictionary containing regression results and data.
        color_column (str): The name of the column in the dataset to use for coloring the data points.
        show_train (bool, optional): Whether to show the training data points. Defaults to True.
        show_test (bool, optional): Whether to show the test data points. Defaults to True.
        set_range (tuple, optional): The range of values to be displayed on the x and y axes. If None, the range is determined automatically. Defaults to None.
        which_error (str, optional): The type of error to display in the annotation. Possible values: "mean", "best", "full_mean", "full_best", "all". Defaults to "mean".
        color_mapping (dict, optional): A dictionary mapping column values to color names or hex values. If None, default colors are used. Defaults to None.
        regr_layout (dict, optional): Additional layout options for the plotly figure. Defaults to None.
        text_column (str, optional): The name of the column in the dataset to use for text labels on the data points. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of plotly figures representing the regression scatter plots.

    Raises:
        KeyError: If an invalid error specification is used for `which_error`.

    Notes:
        - The `regr_dict` parameter should contain the following keys: 'error_dict', 'df_in'.
        - The 'error_dict' should contain error metrics for train and test sets, e.g., 'rsquared_trains', 'rsquared_tests', 'rmse_trains', 'rmse_tests', etc.
        - The 'df_in' should contain the dataset used for regression, including the columns specified in `color_column` and `text_column` if applicable.
    """
    # function_arguments = locals()
    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # Include threshold for deviation somewhere
    # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
    # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
    # TODO: All plotly layout options using kwargs: `showticklabels`, `showlegend`, `label_column`

    # # Print the color names and their corresponding hex values
    # for color_name, hex_value in named_colors.items():
    #     print(f"{color_name}: {hex_value}")
    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # if color_setup is None:
    #     text_colum
    #     color_setup = {}

    # ! At least color column needs to be set to something for the loop over the values later on.
    # ? Currently, using kwargs or the first column of the df to get a column to do the looping over.

    # color_setup = {k:v for k,v in }
    # print(color_setup)

    # color_column = list(color_setup.keys())[0]
    # color_mapping = list(color_setup.values())[0]

    # ! Sort df so that legend items appear ordered -> Check that this does not mess up the ordering from input to pred values.
    if color_column is not None:
        # TODO: Make verbose here to print if
        df_func = df_func.dropna(subset=[color_column])
        color_column_values = sorted(list(set(list(df_func[color_column].values))))
        color_mapping = {
            k: v
            for k, v in list(
                zip(color_column_values, named_colors[0 : len(color_column_values)])
            )
        }

        df_func = df_func.sort_values(by=color_column)

    if symbol_column is not None:
        # TODO: Make verbose here to print if
        df_func = df_func.dropna(subset=[symbol_column])
        symbol_column_values = sorted(list(set(list(df_func[symbol_column].values))))
        symbol_mapping = {
            k: v
            for k, v in list(
                zip(symbol_column_values, named_symbols[0 : len(symbol_column_values)])
            )
        }

        print("Symbol Mapping:", symbol_mapping)  # Check the mapping

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
        if which_error != "all":
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
        # ! Change decimal to 3
        else:
            error_list_train = [
                np.mean(error_dict["rsquared_trains"]),
                np.std(error_dict["rsquared_trains"]),
                np.mean(error_dict["rmse_trains"]),
                np.std(error_dict["rmse_trains"]),
                np.mean(error_dict["mae_trains"]),
                np.std(error_dict["mae_trains"]),
            ]
            error_text = "Train:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
                *error_list_train
            )
            error_list_test = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]

            error_text += "<br>Test:<br>R<sup>2</sup> = {:.2f} &#177; {:.2f}<br>RMSE = {:.2f} &#177; {:.2f}<br>MAE = {:.2f} &#177; {:.2f}".format(
                *error_list_test
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

        # Plot energy data points
        # ! Difference between train and test via marker symbol
        # ! Different colors based on a column

        # print("color_column", color_column)
        if show_train is True:
            for column_value in df_train[color_column].unique():
                column_train_df = df_train.loc[df_train[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_train_df[text_column]
                else:
                    plot_text = [""] * column_train_df.shape[0]

                train_symbols = [
                    symbol_mapping.get(_, "star")
                    for _ in column_train_df[symbol_column].values
                ]

                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_train_df["y"],
                        y=column_train_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=10,
                            symbol=train_symbols,
                            opacity=1,
                            color=color_mapping.get(column_value, "black"),
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_train_legend", True),
                    ),
                )

        if show_test is True:
            for column_value in df_test[color_column].unique():
                column_test_df = df_test.loc[df_test[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_test_df[text_column]
                else:
                    plot_text = [""] * column_test_df.shape[0]

                test_symbols = [
                    symbol_mapping.get(_, "star")
                    for _ in column_test_df[symbol_column].values
                ]

                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_test_df["y"],
                        y=column_test_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=10,
                            symbol=test_symbols,
                            opacity=1,
                            color=color_mapping.get(column_value, "black"),
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_test_legend", True),
                    ),
                )

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


def plot_regr_train_test(
    regr_dict,
    color_column,
    show_train=True,
    show_test=True,
    set_range=None,
    which_error="mean",
    color_mapping=None,
    regr_layout=None,
    text_column=None,
    *args,
    **kwargs,
):
    """
    Generate scatter plots for regression results.

    Args:
        regr_dict (dict): A dictionary containing regression results and data.
        color_column (str): The name of the column in the dataset to use for coloring the data points.
        show_train (bool, optional): Whether to show the training data points. Defaults to True.
        show_test (bool, optional): Whether to show the test data points. Defaults to True.
        set_range (tuple, optional): The range of values to be displayed on the x and y axes. If None, the range is determined automatically. Defaults to None.
        which_error (str, optional): The type of error to display in the annotation. Possible values: "mean", "best", "full_mean", "full_best", "all". Defaults to "mean".
        color_mapping (dict, optional): A dictionary mapping column values to color names or hex values. If None, default colors are used. Defaults to None.
        regr_layout (dict, optional): Additional layout options for the plotly figure. Defaults to None.
        text_column (str, optional): The name of the column in the dataset to use for text labels on the data points. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of plotly figures representing the regression scatter plots.

    Raises:
        KeyError: If an invalid error specification is used for `which_error`.

    Notes:
        - The `regr_dict` parameter should contain the following keys: 'error_dict', 'df_in'.
        - The 'error_dict' should contain error metrics for train and test sets, e.g., 'rsquared_trains', 'rsquared_tests', 'rmse_trains', 'rmse_tests', etc.
        - The 'df_in' should contain the dataset used for regression, including the columns specified in `color_column` and `text_column` if applicable.
    """
    # function_arguments = locals()
    # print(type(kwargs))
    # print(type(function_arguments))
    # print(function_arguments.keys())

    # Include threshold for deviation somewhere
    # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
    # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
    # TODO: All plotly layout options using kwargs: `showticklabels`, `showlegend`, `label_column`

    # # Print the color names and their corresponding hex values
    # for color_name, hex_value in named_colors.items():
    #     print(f"{color_name}: {hex_value}")
    error_dict = regr_dict["error_dict"]

    df_func = regr_dict["df_in"].copy(deep=True)

    # if color_setup is None:
    #     text_colum
    #     color_setup = {}

    # ! At least color column needs to be set to something for the loop over the values later on.
    # ? Currently, using kwargs or the first column of the df to get a column to do the looping over.

    # color_setup = {k:v for k,v in }
    # print(color_setup)

    # color_column = list(color_setup.keys())[0]
    # color_mapping = list(color_setup.values())[0]

    # ! Sort df so that legend items appear ordered -> Check that this does not mess up the ordering from input to pred values.
    if color_column is not None:
        # TODO: Make verbose here to print if
        df_func = df_func.dropna(subset=[color_column])
        color_column_values = sorted(list(set(list(df_func[color_column].values))))
        color_mapping = {
            k: v
            for k, v in list(
                zip(color_column_values, named_colors[0 : len(color_column_values)])
            )
        }

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
        if which_error != "all":
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
        # ! Change decimal to 3 
        else:
            error_list_train = [
                np.mean(error_dict["rsquared_trains"]),
                np.std(error_dict["rsquared_trains"]),
                np.mean(error_dict["rmse_trains"]),
                np.std(error_dict["rmse_trains"]),
                np.mean(error_dict["mae_trains"]),
                np.std(error_dict["mae_trains"]),
            ]
            error_text = "Train:<br>R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
                *error_list_train
            )
            error_list_test = [
                np.mean(error_dict["rsquared_tests"]),
                np.std(error_dict["rsquared_tests"]),
                np.mean(error_dict["rmse_tests"]),
                np.std(error_dict["rmse_tests"]),
                np.mean(error_dict["mae_tests"]),
                np.std(error_dict["mae_tests"]),
            ]

            error_text += "<br>Test:<br>R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
                *error_list_test
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
                if text_column is not None:
                    plot_text = column_train_df[text_column]
                else:
                    plot_text = [""] * column_train_df.shape[0]
                ###
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_train_df["y"],
                        y=column_train_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=12,
                            symbol=0,
                            opacity=1, #initially opacity to 1
                            color='rgba(0,156,156, 0.7)',  # Blue color with 50% transparency
                            #color=color_mapping.get(column_value, "black"),
                            line=dict(
                             color='black',  # Rigid red color for the circle perimeter
                            width=1 ) # Adjust the width of the circle perimeter
                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_train_legend", True),
                    ),
                )

        if show_test is True:
            for column_value in df_test[color_column].unique():
                column_test_df = df_test.loc[df_test[color_column] == column_value]
                if text_column is not None:
                    plot_text = column_test_df[text_column]
                else:
                    plot_text = "" * column_test_df.shape[0]
                ###
                _ = regr_fig.add_trace(
                    go.Scatter(
                        x=column_test_df["y"],
                        y=column_test_df[cv_id_pred_column],
                        mode="markers",
                        marker=dict(
                            size=12, #initially 8
                            symbol=0, # initially 4 for cross
                            opacity=1, #initially opacity to 1
                            color='rgba(156,0,0, 0.7)', #change to red the test set
                            #color=color_mapping.get(column_value, "black"),
                            line=dict(
                             color='black',  # Rigid red color for the circle perimeter
                            width=1 ) # Adjust the width of the circle perimeter

                        ),
                        hoverinfo="text+x+y",
                        name="{}".format(column_value),
                        text=plot_text,
                        legendgroup="{}".format(column_value),
                        showlegend=kwargs.get("show_test_legend", False),
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
    Plot error measures (RMSE, MAE, R^2) for a machine learning model, obtained by varying a parameter.

    Args:
        error_dict (dict): Dictionary containing the error measures.
        x_values (list): List of x-axis values for the parameter being varied.
        plot_measures (list): List of error measures to plot (e.g., ['rmse_train', 'rmse_test', 'mae_train', 'rsquared_test']).
        annot_text (list): List of annotation texts corresponding to each data point on the plot.
        x_title (str): Title for the x-axis.
        showlegend (bool, optional): Whether to show the legend. Defaults to True.

    Returns:
        error_fig (plotly.graph_objects.Figure): Plotly figure object containing the error plot.
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
                x=list(x_values.astype(float).round(2)),
                # y=error_dict[
                #     plot_measure
                # ],
                y=np.array(error_dict[plot_measure]).astype(float).round(2),
                # TODO: Round numbers shown on hover (but not for plotting).
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
                    x=list(x_values.astype(float).round(2)),
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
