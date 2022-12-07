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
        elif save_type == "png":
            plotly_fig.write_image(
                os.path.join(*path_elements, "{}.{}".format(figure_name, save_type)),
                engine="kaleido",
            )

        elif save_type == "pdf":
            plotly_fig.write_image(
                os.path.join(*path_elements, "{}.{}".format(figure_name, save_type)),
                engine="kaleido",
            )

        elif save_type == "html":
            plotly_fig.write_html(
                os.path.join(*path_elements, "{}.{}".format(figure_name, save_type))
            )

    return None


# region
# def plot_regr(
#     result_dict,
#     error_dict,
#     show_train=True,
#     show_test=True,
#     set_range=None,
#     column="metal",
#     cv_id=None,
#     which_error="mean",
#     showticklabels=True,
#     showlegend=True,
# ):
#     # Include threshold for deviation somewhere
#     # TODO: When doing LOOCV, the excluded metal (group) is not added to the legend.
#     # TODO: Add possibility to provide specific colors. Had to remove that from the metal-color dict so that the code works general.
#     try:
#         if cv_id is None:
#             cv_id = np.argmin(error_dict["rmse_tests"])

#         train_df = pd.DataFrame(
#             {
#                 column: result_dict["m_trains"][cv_id],
#                 "plot_label": result_dict["l_trains"][cv_id],
#                 "y_calc": result_dict["y_trains"][cv_id],
#                 "y_pred": result_dict["y_train_preds"][cv_id],
#             }
#         )

#         test_df = pd.DataFrame(
#             {
#                 column: result_dict["m_tests"][cv_id],
#                 "plot_label": result_dict["l_tests"][cv_id],
#                 "y_calc": result_dict["y_tests"][cv_id],
#                 "y_pred": result_dict["y_test_preds"][cv_id],
#             }
#         )

#     except IndexError:
#         # Tried to create df with metal and label before and then add y_calc and y_pred with if, but assignment of new
#         # columns gives ValueError due to supposed different length (1) of passed array to new df-column
#         train_df = pd.DataFrame(
#             {
#                 column: result_dict["m_trains"][cv_id],
#                 "plot_label": result_dict["l_trains"][cv_id],
#                 "y_calc": result_dict["y_trains"][cv_id],
#                 "y_pred": result_dict["y_train_preds"][cv_id],
#             }
#         )

#         test_df = pd.DataFrame(
#             {
#                 column: result_dict["m_tests"][cv_id],
#                 "plot_label": result_dict["l_tests"][cv_id],
#                 "y_calc": result_dict["y_tests"][cv_id],
#                 "y_pred": result_dict["y_test_preds"][cv_id],
#             }
#         )

#     # TODO: is this really needed? Could just remove so that column_value is not a required input.
#     # if column_value is not None:
#     #     train_df = train_df.loc[train_df[column] == column_value]
#     #     test_df = test_df.loc[test_df[column] == column_value]

#     # Plot energies
#     ener_fig = go.Figure()

#     # Add annotation with R^2 and RMSEs
#     # TODO: Implement this better
#     if which_error == "mean":
#         _ = ener_fig.add_annotation(
#             xanchor="left",
#             yanchor="top",
#             xref="paper",
#             yref="paper",
#             x=0,
#             y=1,
#             align="left",
#             text="R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
#                 np.mean(error_dict["rsquared_tests"]),
#                 np.std(error_dict["rsquared_tests"]),
#                 np.mean(error_dict["rmse_tests"]),
#                 np.std(error_dict["rmse_tests"]),
#                 np.mean(error_dict["mae_tests"]),
#                 np.std(error_dict["mae_tests"]),
#             ),
#             font_size=26,
#             font_family="Arial",
#             showarrow=False,
#             bgcolor="rgba(0,0,0,0.1)",
#         )

#     elif which_error == "best":
#         _ = ener_fig.add_annotation(
#             xanchor="left",
#             yanchor="top",
#             xref="paper",
#             yref="paper",
#             x=0,
#             y=1,
#             align="left",
#             text="R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
#                 np.max(error_dict["rsquared_tests"]),
#                 np.std(error_dict["rsquared_tests"]),
#                 np.min(error_dict["rmse_tests"]),
#                 np.std(error_dict["rmse_tests"]),
#                 np.min(error_dict["mae_tests"]),
#                 np.std(error_dict["mae_tests"]),
#             ),
#             font_size=26,
#             font_family="Arial",
#             showarrow=False,
#             bgcolor="rgba(0,0,0,0.1)",
#         )

#     elif which_error == "full_mean":
#         _ = ener_fig.add_annotation(
#             xanchor="left",
#             yanchor="top",
#             xref="paper",
#             yref="paper",
#             x=0,
#             y=1,
#             align="left",
#             text="R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
#                 np.mean(error_dict["rsquared_fulls"]),
#                 np.std(error_dict["rsquared_fulls"]),
#                 np.mean(error_dict["rmse_fulls"]),
#                 np.std(error_dict["rmse_fulls"]),
#                 np.mean(error_dict["mae_fulls"]),
#                 np.std(error_dict["mae_fulls"]),
#             ),
#             font_size=26,
#             font_family="Arial",
#             showarrow=False,
#             bgcolor="rgba(0,0,0,0.1)",
#         )

#     elif which_error == "full_best":
#         _ = ener_fig.add_annotation(
#             xanchor="left",
#             yanchor="top",
#             xref="paper",
#             yref="paper",
#             x=0,
#             y=1,
#             align="left",
#             text="R<sup>2</sup> = {:.3f} &#177; {:.3f}<br>RMSE = {:.3f} &#177; {:.3f}<br>MAE = {:.3f} &#177; {:.3f}".format(
#                 np.max(error_dict["rsquared_fulls"]),
#                 np.std(error_dict["rsquared_fulls"]),
#                 np.min(error_dict["rmse_fulls"]),
#                 np.std(error_dict["rmse_fulls"]),
#                 np.min(error_dict["mae_fulls"]),
#                 np.std(error_dict["mae_fulls"]),
#             ),
#             font_size=26,
#             font_family="Arial",
#             showarrow=False,
#             bgcolor="rgba(0,0,0,0.1)",
#         )

#     else:
#         raise ValueError("Invalid error specification used.")

#     # TODO: Add here general functionality to provide color dict or not. Just hackish solution by copy-pasting all the code.
#     if column == "metal":
#         # Plot energy data points
#         test_legend = True
#         if show_train is True:
#             test_legend = False
#             for _column_value in train_df[column].unique():
#                 # Add trace for training data
#                 metal_train_df = train_df.loc[train_df[column] == _column_value]

#                 # TODO: Added this print as concatenation of the plot label and deviation as follows did not work due to str and bytes data type.
#                 # for i, j in zip(metal_train_df['plot_label'].to_list(), ['<br>Deviation: {}eV'.format(round(dev, 3)) for dev in np.abs(metal_train_df['y_calc'] - metal_train_df['y_pred'])]):
#                 #     print(i, j)

#                 train_plot_text = [
#                     i + j
#                     for i, j in zip(
#                         metal_train_df["plot_label"].to_list(),
#                         [
#                             "<br>Deviation: {}eV".format(round(dev, 3))
#                             for dev in np.abs(
#                                 metal_train_df["y_calc"] - metal_train_df["y_pred"]
#                             )
#                         ],
#                     )
#                 ]
#                 _ = ener_fig.add_trace(
#                     go.Scatter(
#                         x=metal_train_df["y_calc"],
#                         y=metal_train_df["y_pred"],
#                         mode="markers",
#                         marker=dict(
#                             size=8,
#                             symbol=0,
#                             opacity=1,
#                             color=color_dict.get(_column_value, "blue"),
#                         ),
#                         hoverinfo="text+x+y",
#                         name=_column_value,
#                         text=train_plot_text,
#                         legendgroup=_column_value,
#                         showlegend=showlegend,
#                     ),
#                 )

#         if show_test is True:
#             # Add testing data in second loop so that it is on top
#             for _column_value in test_df[column].unique():
#                 # Add trace for testing data
#                 metal_test_df = test_df.loc[test_df[column] == _column_value]

#                 test_plot_text = [
#                     i + j
#                     for i, j in zip(
#                         metal_test_df["plot_label"].to_list(),
#                         [
#                             "<br>Deviation: {}eV".format(round(dev, 3))
#                             for dev in np.abs(
#                                 metal_test_df["y_calc"] - metal_test_df["y_pred"]
#                             )
#                         ],
#                     )
#                 ]

#                 _ = ener_fig.add_trace(
#                     go.Scatter(
#                         x=metal_test_df["y_calc"],
#                         y=metal_test_df["y_pred"],
#                         mode="markers",
#                         marker=dict(
#                             size=11,
#                             symbol="x",
#                             opacity=1,
#                             color=color_dict.get(_column_value, "blue"),
#                             line=dict(
#                                 width=0.5,
#                                 color="rgba(255, 255, 255, 0.5)",
#                             ),
#                         ),
#                         hoverinfo="text+x+y",
#                         name=_column_value,
#                         text=test_plot_text,
#                         legendgroup=_column_value,
#                         showlegend=test_legend & showlegend,
#                     ),
#                 )
#     else:
#         # Plot energy data points
#         test_legend = True
#         if show_train is True:
#             test_legend = False
#             for _column_value in train_df[column].unique():
#                 # Add trace for training data
#                 metal_train_df = train_df.loc[train_df[column] == _column_value]

#                 train_plot_text = [
#                     i + j
#                     for i, j in zip(
#                         metal_train_df["plot_label"].to_list(),
#                         [
#                             "<br>Deviation: {}eV".format(round(dev, 3))
#                             for dev in np.abs(
#                                 metal_train_df["y_calc"] - metal_train_df["y_pred"]
#                             )
#                         ],
#                     )
#                 ]
#                 _ = ener_fig.add_trace(
#                     go.Scatter(
#                         x=metal_train_df["y_calc"],
#                         y=metal_train_df["y_pred"],
#                         mode="markers",
#                         marker=dict(size=8, symbol=0, opacity=1),
#                         hoverinfo="text+x+y",
#                         name=_column_value,
#                         text=train_plot_text,
#                         legendgroup=_column_value,
#                         showlegend=showlegend,
#                     ),
#                 )

#         if show_test is True:
#             # Add testing data in second loop so that it is on top
#             for _column_value in test_df[column].unique():
#                 # Add trace for testing data
#                 metal_test_df = test_df.loc[test_df[column] == _column_value]

#                 test_plot_text = [
#                     i + j
#                     for i, j in zip(
#                         metal_test_df["plot_label"].to_list(),
#                         [
#                             "<br>Deviation: {}eV".format(round(dev, 3))
#                             for dev in np.abs(
#                                 metal_test_df["y_calc"] - metal_test_df["y_pred"]
#                             )
#                         ],
#                     )
#                 ]

#                 _ = ener_fig.add_trace(
#                     go.Scatter(
#                         x=metal_test_df["y_calc"],
#                         y=metal_test_df["y_pred"],
#                         mode="markers",
#                         marker=dict(
#                             size=11,
#                             symbol="x",
#                             opacity=1,
#                             line=dict(
#                                 width=0.5,
#                                 color="rgba(255, 255, 255, 0.5)",
#                             ),
#                         ),
#                         hoverinfo="text+x+y",
#                         name=_column_value,
#                         text=test_plot_text,
#                         legendgroup=_column_value,
#                         showlegend=test_legend,
#                     ),
#                 )

#     # # fig_for_range = ener_fig.full_figure_for_development(warn=False)
#     # # x_range = fig_for_range.layout.xaxis.range
#     # # y_range = fig_for_range.layout.xaxis.range
#     # big_range = [min(x_range+y_range), max(x_range+y_range)]
#     all_values = (
#         train_df["y_calc"].tolist()
#         + train_df["y_pred"].tolist()
#         + test_df["y_calc"].tolist()
#         + test_df["y_pred"].tolist()
#     )
#     full_range = [min(all_values), max(all_values)]
#     range_ext = (
#         full_range[0] - 0.075 * np.ptp(full_range),
#         full_range[1] + 0.075 * np.ptp(full_range),
#     )
#     # x_range_ext = (x_range[0] - 0.075*np.ptp(x_range), x_range[1] + 0.075*np.ptp(x_range))
#     # y_range_ext = (y_range[0] - 0.075*np.ptp(y_range), y_range[1] + 0.075*np.ptp(y_range))
#     if set_range is not None:
#         range_ext = set_range

#     # Update global layout
#     ener_layout = go.Layout(
#         width=597,
#         height=597,
#         font=dict(family="Arial", color="black"),
#         margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#         ),
#         hoverlabel={"namelength": -1},
#         # title=dict(text=plot_title, x=0.5, ),
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         legend=dict(
#             xanchor="right",
#             x=1,
#             yanchor="bottom",
#             y=0,
#             bgcolor="rgba(0,0,0,0.1)",  # bordercolor='rgba(0,0,0,0.4)',
#             font_size=26,
#             tracegroupgap=2,
#         ),
#         xaxis=dict(
#             title_font_size=30,
#             range=range_ext,
#             showticklabels=showticklabels,  # title='E<sub>DFT</sub> / eV',
#             showline=True,
#             linewidth=3,
#             linecolor="black",
#             mirror=True,
#             showgrid=False,
#             zeroline=False,
#             gridcolor="rgba(0,0,0,0.3)",
#             ticks="outside",
#             tickfont_size=26,
#             tickformat=".1f",
#             tickwidth=3,
#             ticklen=6,
#         ),
#         yaxis=dict(
#             title_font_size=30,
#             range=range_ext,
#             showticklabels=showticklabels,  # lreg_energy_fig_set1_ncoordmos
#             showline=True,
#             linewidth=3,
#             linecolor="black",
#             mirror=True,
#             showgrid=False,
#             zeroline=False,
#             gridcolor="rgba(0,0,0,0.3)",
#             ticks="outside",
#             tickfont_size=26,
#             tickformat=".1f",
#             tickwidth=3,
#             ticklen=6,
#         ),
#     )

#     _ = ener_fig.update_layout(ener_layout)

#     # Add ideal fit line to plot
#     _ = ener_fig.add_trace(
#         go.Scatter(
#             x=range_ext,
#             y=range_ext,
#             mode="lines",
#             line=dict(color="black", width=2, dash="dash"),
#             hoverinfo="skip",
#             showlegend=False,
#         ),
#     )

#     return ener_fig
# endregion


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
                x=x_values,
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
                    x=x_values,
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


def prim_feat_hist(dict_in):
    hist_fig = go.Figure()
    hist_layout = go.Layout(
        width=1000,
        height=600,
        font=dict(size=20, family="Arial", color="black"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        xaxis=dict(
            title="Primary Feature",
            title_font_size=20,
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",
            ticks="outside",
            tickfont_size=12,
            tickwidth=3,
            ticklen=6,
            categoryorder="mean descending",
        ),
        yaxis=dict(
            title="Coeff",
            title_font_size=20,
            showline=True,
            linewidth=3,
            linecolor="black",
            mirror=True,
            showgrid=False,
            zeroline=False,
            gridcolor="rgba(0,0,0,0.3)",
            ticks="outside",
            tickfont_size=18,
            tickwidth=3,
            ticklen=6,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        barmode="stack",
        legend=dict(
            yanchor="top", y=1, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.1)"
        ),
    )

    _ = hist_fig.add_trace(
        go.Bar(
            name="test",
            x=list(dict_in.keys()),
            y=list(dict_in.values()),
            text=["{:.3f}".format(val) for val in list(dict_in.values())],
            textposition="inside",
            textfont=dict(size=20, family="Arial", color="black"),
        ),
    )

    _ = hist_fig.update_layout(hist_layout)
    return hist_fig


def corr_analysis(df_in, features, thresh=0.95, verbose=False, create_image=True):
    # Visualize variable correlations

    corr_df = df_in[features].corr().abs()

    if thresh is None:
        thresh_corr_df = corr_df[(corr_df > 1 - float("1e-9"))]
    else:
        thresh_corr_df = corr_df[(corr_df > thresh) & (corr_df != 1)]

    # plot_corr_df = thresh_corr_df.where(np.tril(np.ones(thresh_corr_df.shape)).astype(bool))

    return_dict = {"corr_df": thresh_corr_df}

    if create_image is False:
        return return_dict

    # Define hovertext for x unified hovermode that for any descriptor all correlations are shown.
    if len(features) < 200:
        hovertext = []
        for illist, llist in enumerate(thresh_corr_df.values.tolist()):
            temp = ""
            for ielement, element in enumerate(llist):
                if not math.isnan(element):
                    temp += "{}: {}<br>".format(features[ielement], round(element, 3))
            hovertext.append([temp[:-4]] * ielement)
        hovertext = np.array(hovertext)  # .transpose()

    # Plot correlation heatmap
    corr_fig = go.Figure()

    _ = corr_fig.add_trace(
        go.Heatmap(
            x=corr_df.columns,
            y=corr_df.columns,
            z=thresh_corr_df.values,
            text=np.around(thresh_corr_df.values, decimals=2),
            hoverinfo="x+y+z",
            xgap=0.2,
            ygap=0.2,
            colorbar=dict(thickness=20, ticklen=3, y=0.501, len=1.026),
        )
    )

    full_corr_layout = go.Layout(
        width=1000,
        height=1000,
        font=dict(size=16, family="Arial", color="black"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        # title=dict(text='Correlation > {}'.format(thresh), x=0.5),
        # hovermode='x unified',
        xaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont_size=14,
            linecolor="black",
            showline=True,
            mirror=True,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont_size=14,
            linecolor="black",
            showline=True,
            mirror=True,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _ = corr_fig.update_layout(full_corr_layout)
    return_dict["corr_fig"] = corr_fig

    corr_pairs, corr_set = [], []

    # Print out correlated features for each feature if verbosity is set.
    if verbose is True:
        print("Correlated descriptors:\n")
        for desc_comb in it.combinations(corr_df.columns, 2):
            corr = corr_df[desc_comb[0]][desc_comb[1]]
            if corr > thresh:
                corr_pairs.append(sorted([*desc_comb]))
                corr_set += [*desc_comb]

        corr_set = set(corr_set)

        corr_dict = {corr_desc: [] for corr_desc in corr_set}

        for corr_pair in corr_pairs:
            corr_dict[corr_pair[0]].append(corr_pair[1])

        for corr_key, corr_vals in corr_dict.items():
            if len(corr_vals) > 0:
                print("{} ({}): {}".format(corr_key, len(corr_vals), corr_vals))
        print("=" * 127)

    return return_dict


def plot_en_scan(result_dict, numfeat_thresh=100, rmse_thresh=0.3):
    en_alphas = result_dict["en_alphas"]
    en_l1_ratios = result_dict["en_l1_ratios"]
    numfeat_df = result_dict["numfeat_df"]
    rmse_df = result_dict["rmse_df"]

    heatmaps = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.2,
    )

    _ = heatmaps.add_trace(
        go.Heatmap(
            x=en_alphas,
            y=en_l1_ratios,
            z=numfeat_df[(rmse_df <= rmse_thresh) & (numfeat_df <= numfeat_thresh)],
            # z=en_heatmap_array,
            hoverinfo="x+y+z+text",
            hovertext=np.round(rmse_df.values, 3),
            zhoverformat="d",
            xgap=0,
            ygap=0,
            colorbar=dict(
                x=0.4,
                thickness=20,
                ticklen=3,
                tickfont=dict(size=18, color="black", family="Arial"),
            ),
        ),
        row=1,
        col=1,
    )

    _ = heatmaps.add_trace(
        go.Heatmap(
            x=en_alphas,
            y=en_l1_ratios,
            z=rmse_df[(rmse_df <= rmse_thresh) & (numfeat_df <= numfeat_thresh)],
            # z=en_rmse_array, #[rmse_df < .5],
            hoverinfo="x+y+z+text",
            zhoverformat=".3f",
            hovertext=numfeat_df.values,
            xgap=0,
            ygap=0,  # vertical_spacing=0.08, horizontal_spacing=0.08,
            colorbar=dict(
                x=1,
                thickness=20,
                ticklen=3,
                tickfont=dict(size=18, color="black", family="Arial"),
            ),
        ),
        row=1,
        col=2,
    )

    xaxis_dict = dict(
        title="&#x3b1;",
        title_font_size=20,
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
        ticks="outside",
        tickfont_size=18,
        hoverformat=".3f",
        tickwidth=3,
        ticklen=6,
    )
    yaxis_dict = dict(
        title=dict(
            text="l1-ratio",
            font_size=20,
            standoff=10,
        ),
        showline=True,
        linewidth=3,
        linecolor="black",
        mirror=True,
        showgrid=False,
        zeroline=False,
        gridcolor="rgba(0,0,0,0.3)",
        ticks="outside",
        tickfont_size=18,
        hoverformat=".3f",
        tickwidth=3,
        ticklen=6,
    )
    en_heatmap_layout = go.Layout(
        width=1200,
        height=500,
        font=dict(size=18, family="Arial", color="black"),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
        ),
        xaxis=xaxis_dict,
        xaxis2=xaxis_dict,
        yaxis=yaxis_dict,
        yaxis2=yaxis_dict,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    _ = heatmaps.update_layout(en_heatmap_layout)

    return heatmaps
