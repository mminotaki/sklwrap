import argparse
import itertools as it
import json
import math as m
import os
import pickle
import re
from subprocess import check_output

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from pandas.io.formats.style import Styler
from plotly.offline import download_plotlyjs, plot
from plotly.utils import PlotlyJSONEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import *


def apply_feat_mask(model, model_features, feat_thresh=0):

    try:
        coefs = model.coef_
    except AttributeError:
        coefs = model.feature_importances_

    if len(coefs.shape) == 2:
        coefs = np.sum(np.abs(model.coef_), axis=0)

    coef_mask = [True if _ > feat_thresh else False for _ in coefs]

    return list(it.compress(model_features, coef_mask))


def get_base_errors(df_in):
    avg_metal = [np.mean(df_in["E_rel_metal"].values)] * df_in.shape[0]
    avg_global = [np.mean(df_in["E_rel_global"].values)] * df_in.shape[0]
    avg_both = np.array([avg_metal, avg_global]).transpose()

    base_rmse_metal = mean_squared_error(avg_metal, df_in["E_rel_metal"], squared=False)
    base_rmse_global = mean_squared_error(
        avg_global, df_in["E_rel_global"], squared=False
    )
    base_rmse_both = mean_squared_error(
        avg_both, df_in[["E_rel_metal", "E_rel_global"]].values, squared=False
    )

    base_mae_metal = mean_absolute_error(avg_metal, df_in["E_rel_metal"])
    base_mae_global = mean_absolute_error(avg_global, df_in["E_rel_global"])
    base_mae_both = mean_absolute_error(
        avg_both, df_in[["E_rel_metal", "E_rel_global"]].values
    )

    result_string = """
    RMSEs:\n
    Metal: {:.2f}, Global: {:.2f}, Both: {:.2f}\n
    MAEs:\n
    Metal: {:.2f}, Global: {:.2f}, Both: {:.2f}
    """.format(
        base_rmse_metal,
        base_rmse_global,
        base_rmse_both,
        base_mae_metal,
        base_mae_global,
        base_mae_both,
    )

    return result_string


def get_prim_feat(
    ml_model,
    ml_features,
    prim_features,
    mask_thresh=0,
    measure="coeff",
    sort_dict=True,
    create_fig=True,
) -> dict:
    """"""
    """
    Function to decompose 2D features into primary ones and create histogram, either of counts or coefficients.
    :param ml_model:
    :param ml_features:
    :param prim_features:
    :param mask_thresh:
    :param measure:
    :param sort_dict:
    :return:
    """

    return_dict = {
        "selected_combi": None,
        "numfeats": None,
        "feat_dict": None,
        "feat_fig": None,
    }

    selected_features = apply_feat_mask(
        model=ml_model, model_features=ml_features, feat_thresh=mask_thresh
    )
    return_dict["selected_combi"] = selected_features

    prim_feat_dict = dict.fromkeys(prim_features, 0)

    return_dict["numfeats"] = len(selected_features)

    if measure == "coeff":
        # Negative coefficients indicate an inverse relationship but should not be a couse for a concern. Sum up
        try:
            coefs = np.abs(ml_model.coef_)
        except AttributeError:
            coefs = np.abs(ml_model.feature_importances_)

        if len(coefs.shape) == 2:
            coefs = np.sum(np.abs(coefs), axis=0)

        for iml_feature, ml_feature in enumerate(ml_features):

            if " * " in ml_feature:
                feat_split = [ml_feature.split(" * ")[0], ml_feature.split(" * ")[1]]
                prim_feat_dict[feat_split[0]] += coefs[iml_feature] / 2
                prim_feat_dict[feat_split[1]] += coefs[iml_feature] / 2

            elif " / " in ml_feature:
                feat_split = [ml_feature.split(" / ")[0], ml_feature.split(" / ")[1]]
                prim_feat_dict[feat_split[0]] += coefs[iml_feature] / 2
                prim_feat_dict[feat_split[1]] += coefs[iml_feature] / 2

            else:
                prim_feat_dict[ml_feature] += coefs[iml_feature]

    else:
        selected_feature_string = "-".join(selected_features)
        for prim_feature in prim_features:
            prim_feat_dict[prim_feature] = selected_feature_string.count(prim_feature)

    if sort_dict is True:
        prim_feat_dict = dict(
            sorted(prim_feat_dict.items(), key=lambda item: item[1], reverse=True)
        )

    return_dict["feat_dict"] = prim_feat_dict

    if create_fig is True:
        feat_fig = go.Figure()
        _ = feat_fig.add_trace(
            go.Bar(
                x=list(prim_feat_dict.keys()),
                y=list(prim_feat_dict.values()),
                text=["{:.3f}".format(val) for val in list(prim_feat_dict.values())],
                textposition="none",  # textposition='inside',
                # textfont=dict(size=28, family='Arial', color='black'),
                insidetextfont=dict(
                    size=20,
                    family="Arial",
                    color="black",
                ),
                # outsidetextfont=dict(size=20, family='Arial', color='black', ),
                showlegend=False,
            ),
        )

        # feat_layout = go.Layout(
        #     width=800, height=500,
        #     margin=dict(l=0, r=0, b=0, t=0, ),
        #     xaxis=dict(
        #         title='Primary Feature', title_font_size=20,
        #         showline=True, linewidth=3, linecolor='black', mirror=True,
        #         showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0.3)',
        #         ticks='outside', tickfont_size=14, tickwidth=3, ticklen=6),
        #     yaxis=dict(
        #         title=measure.title(), title_font_size=20,
        #         showline=True, linewidth=3, linecolor='black', mirror=True,
        #         showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0.3)',
        #         ticks='outside', tickfont_size=18, tickwidth=3, ticklen=6),
        #     paper_bgcolor='white', plot_bgcolor='white',
        # )

        feat_layout = go.Layout(
            width=809,
            height=500,
            font=dict(family="Arial", color="black"),
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
            xaxis=dict(
                title="Primary Feature",
                title_font_size=22,
                showline=True,
                linewidth=3,
                linecolor="black",
                mirror=True,
                showgrid=False,
                zeroline=False,
                gridcolor="rgba(0,0,0,0.3)",
                ticks="outside",
                tickfont_size=14,
                tickformat=".1f",
                tickwidth=3,
                ticklen=6,
            ),
            yaxis=dict(
                # title='E<sub>pred</sub> / eV', title_font_size=30, range=y_range_ext,
                # showline=True, linewidth=3, linecolor='black', mirror=True,
                # showgrid=False, zeroline=False, gridcolor='rgba(0,0,0,0.3)',
                # ticks='outside', tickfont_size=26, tickformat=".1f", tickwidth=3, ticklen=6),
                title=measure.title(),
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

        _ = feat_fig.update_layout(feat_layout)
        return_dict["feat_fig"] = feat_fig

    return return_dict


def per_atom_multiply(itable):
    return list(it.chain.from_iterable(it.repeat(x, 109) for x in itable))


# Previous `own_utils.py`
def set_figsize(width, fraction=1):
    """Set aesthetic figure dimensions to avoid scaling in latex.
    Obtained from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def get_one_line(filepath, line_number):
    """
    Simple helper function that single line from a file.
    :param filepath: Path to the file.
    :param line_number: Line number to be extracted
    :return:
    """
    return check_output(["sed", "-n", "%sp" % line_number, filepath])


def del_file(pfile):
    """
    Delete file if exists.
    :param pfile: File to be deleted
    """
    try:
        os.remove(pfile)
    except OSError:
        pass


def get_metal(poscar_file="POSCAR"):
    """
    Function to retrieve name of the metal from the POSCAR file.
    :param poscar_file: Name of POSCAR. Can be used to specify different path.
    :return: Returns metal adsorbed on surface.
    """
    try:
        metal = get_one_line(poscar_file, 6).decode().split()[-1].lower()
        return metal
    except FileNotFoundError:
        print("No POSCAR file found in current directory.")
        return None


def read_incar(incar_file="INCAR"):
    """
    Function to parse incar file into dictionary containing relevant options for MD-run. Keys are all in lowercase.
    :param incar_file: Name of INCAR. Can be used to specify different path.
    :return: Dict wih md-relevant parameters from INCAR file.
    """
    incar_dict = {
        "tebeg": None,
        "teend": None,
        "potim": None,
        "nsw": None,
        "nelm": None,
    }
    with open(incar_file, "r") as INCAR:
        incar = INCAR.readlines()[1:]
        for line in incar:
            for prop in incar_dict.keys():
                if prop in line.lower().split("#")[0]:
                    try:
                        incar_dict[prop] = int(line.strip("\n").split(" = ")[1])

                    # Exception handling for TEBEG = X; TEEND = Y. Separated by semicolon.
                    except ValueError:
                        if prop == "tebeg":
                            incar_dict[prop] = int(
                                line.strip("\n").split(";")[0].split(" = ")[1]
                            )
                        elif prop == "teend":
                            incar_dict[prop] = int(
                                line.strip("\n").split(";")[1].split(" = ")[1]
                            )

    if incar_dict["teend"] is None:
        incar_dict["teend"] = incar_dict["tebeg"]

    return incar_dict


def id_to_linetype(id_in):
    """
    Returns linetype for plotting depending on atom type and layer.
    '-': surface Ce
    '--': subsurface Ce
    '-.': subsurface Ce
    ':': other Ce or O
    :param id_in: ID of atom.
    :return: Linetype for Pyplot.
    """
    if id_in in [13, 14, 15, 16, 49]:  # Surface Ce
        line_type = "-"
    elif id_in in [9, 10, 11, 12]:  # subsurface Ce
        line_type = "--"
    elif id_in in [5, 6, 7, 8]:  # subsubsurface Ce
        line_type = "-."
    else:  # Other Ce
        line_type = ":"

    return line_type


def id_to_color(id_in):
    """
    Returns color for plotting depending on atom type and layer. Surface and subsurface Ce have special coloring. Other
    atoms red.
    :param id_in: ID of atom.
    :return: Linetype for Pyplot.
    """
    color_dict = {
        13: "b",
        14: "g",
        15: "r",
        16: "m",
        9: "tab:blue",
        10: "tab:green",
        11: "tab:red",
        12: "tab:pink",
        49: "k",
    }
    try:
        return color_dict[id_in]
    except KeyError:
        return "r"


def str2bool(str_in):
    """
    Function to convert various string inputs to boolean variables.
    :param str_in:
    :return: Boolean
    """
    if isinstance(str_in, bool):
        return str_in
    if str_in.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif str_in.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_atom_dict():
    """
    Function to read a POSCAR file and return dictionary with atomt types and numbers.
    :return: Dictionary with atom type as key and number of atom as value.
    """
    atom_dict = {
        key: get_one_line("POSCAR", 7).decode().split()[i]
        for i, key in enumerate(get_one_line("POSCAR", 6).decode().split())
    }

    return atom_dict


def create_datfile_dict(to_read, mdir=None):
    """
    Function to parse .dat-files which were previously created with the 'data_extr.sh' bash script. Depending on
    'per-calc' or 'per-atom' property, they are parsed differently.
    :param to_read: Which file to read. Options: 'ener', 'ewal', 'mag', 'chg'
    :param mdir: Specify metal directory
    :return: Returns dictionary with the calculation directories as keys and the properties as values (either single
    values or list.
    """
    if mdir == "co":
        mdir = "cob"

    with open(to_read + ".dat", "r") as dat_file:

        if to_read in ["ener", "ewal", "ext_pres"]:
            out_dict = {
                fline.rstrip("\n")
                .split(mdir)[-1]
                .split()[0]: fline.rstrip("\n")
                .split()[1]
                for fline in dat_file
                if len(fline.rstrip("\n").split()) == 2
                and abs(float(fline.rstrip("\n").split()[1])) > 0.001
            }

        elif to_read in ["mag", "chg"]:
            out_dict = {
                m_line.rstrip("\n").split(mdir)[-1]: []
                for m_line in dat_file
                if len(m_line) > 30
            }
            dat_file.seek(0)
            for m_line in dat_file:
                if len(m_line) > 30:
                    calc_dir = m_line.strip("\n").split(mdir)[-1]
                else:
                    try:
                        out_dict[calc_dir].append(float(m_line.strip("\n").split()[-1]))
                    except KeyError:
                        continue
    return out_dict


def header_select(atom_sel):
    """
    Function to select an appropriate classification header for (sub-)surface cerium/oxygen atoms.
    :param atom_sel: Atom to be considered. Must be a pyRDTP.Atom object.
    :return: Returns a header-string with the classification of the atom.
    """

    head_str = None
    if 6 < atom_sel.coords[-1] < 8 and atom_sel.element == "Ce":
        head_str = "Ce" + "_sub"
    elif 8 < atom_sel.coords[-1] and atom_sel.element == "Ce":
        head_str = "Ce" + "_s"
    elif 6 < atom_sel.coords[-1] < 9 and atom_sel.element == "O":
        head_str = "O" + "_sub"
    elif 9 < atom_sel.coords[-1] and atom_sel.element == "O":
        head_str = "O" + "_s"

    if atom_sel.element not in ["Ce", "O"]:
        head_str = atom_sel.element

    return head_str


def regex_remove_df(df_in, regex_remove=None):
    """
    Little helper function that allows to remove columns through matching of substrings.
    :param df_in: Input-df.
    :param regex_remove: List of substrings that should be removed.
    :return: Returns input-df with the specified columns removed.
    """
    if regex_remove is not None:
        return df_in[df_in.columns.drop(list(df_in.filter(regex=regex_remove)))].copy()
    else:
        return df_in


def get_nce3(df_in: pd.DataFrame, thresh: int = 0.8) -> int:
    """
    Returns number of Ce3+ from output.csv-dataframe.
    :param df_in: Dataframe from output.csv for one calculation.
    :param thresh: Magnetisation threshold.
    :return: Number of Ce3+.
    """
    nce3 = sum([1 for mgn in df_in.mgn[:-1] if mgn > thresh])
    print(nce3)

    return nce3


def list_consec(iter_in, threshold=2):
    """
    Finds subgrups of consecutively reoccuring elements in iterable.
    @param iter_in: Iterable as input.
    @param threshold: Minimum threshold of repetition for registration. (inclusive, default: 2)
    @return: Dictionary with elements of iterable as keys and each repetition lenght in input list in a list.
    """

    # ================================================================================#
    #                    USE NAMEDTUPLE INSTEAD OF DICT?                              #
    # ================================================================================#

    elements = set(iter_in)
    outdict = {element: [] for element in elements}

    _counter = 0
    temp = iter_in[0]
    for element in iter_in:
        if element == temp:
            _counter += 1
        else:
            if _counter >= threshold:
                outdict[temp].append(_counter)
            _counter = 1
            temp = element

    # Now last one missing
    if _counter > threshold:
        outdict[temp].append(_counter)

    return outdict


def md_to_df(direc=os.getcwd(), write_csv=False, fname="md_df.csv"):
    """
    Creates a pandas dataframe containing atomic coordinates and magnetisations for each step of an MD run in the
    respective directory.
    @param direc: Specify MD-directory. Default: present working directory.
    @param write_csv: Write dataframe to csv? Default: False.
    @param fname: Filename of csv. Default md_df.csv.
    @return: Returns dataframe.
    """

    if not direc[-1] == "/":
        direc += "/"

    # Read md_mag.out to list
    with open(direc + "md_mag.out") as md_mag:
        mag_lines = [line.rstrip("\n") for line in md_mag.readlines()]

    # Read XDATCAR to list
    with open(direc + "XDATCAR") as xdat:
        xdat_lines = [line.rstrip("\n") for line in xdat.readlines()]

    # Get number of steps and atom information from input
    nsteps = len(
        [xdat_line for xdat_line in xdat_lines if "configuration" in xdat_line]
    )
    atom_types = xdat_lines[
        xdat_lines.index("Direct configuration= 1", 0, 50) - 2
    ].split()
    atom_numbers = list(
        map(
            int,
            xdat_lines[xdat_lines.index("Direct configuration= 1", 0, 50) - 1].split(),
        )
    )
    natoms = sum(atom_numbers)
    atom_list = []
    for iatom_type, atom_type in enumerate(atom_types):
        atom_list.extend([atom_type] * atom_numbers[iatom_type])

    # Initialize np-array with correct shape
    df_data = np.zeros((nsteps, 4 * natoms))

    # ================================================================================#
    #   THE FOLLOWING CODE COULD PROBABLY BE COLLAPSED IF FILES ARE PARSED THE SAME   #
    # ================================================================================#

    # Go over XDATCAR and array with atomic coordinates
    md_step = -1
    atom_id = 0
    for line in xdat_lines[xdat_lines[:50].index("Direct configuration= 1") :]:
        if "configuration" not in line:
            for xyz in range(3):
                df_data[md_step, atom_id + xyz] = float(line.split()[xyz])
            atom_id += 4
        else:
            atom_id = 0
            md_step += 1

    # Go over md_mag and fill array with magnetizations
    md_step = -1
    for line in mag_lines:
        if "magnetization" not in line:
            df_data[md_step, 4 * int(line.split()[0]) - 1] = np.abs(
                float(line.split()[-1])
            )
        else:
            md_step += 1

    # Create column names
    columns = []
    for i, atom in enumerate(atom_list):
        temp = "_".join([atom, str(i)])
        for j in "xyzm":
            columns.append("-".join([j, temp]))

    # Create df with all data
    df_out = pd.DataFrame(columns=columns, data=df_data)
    df_out.index.name = "step"

    # Return
    if write_csv:
        df_out.to_csv(direc + fname, sep=";")

    return df_out


def md_to_maglist(md_df, mag_thresh=0.8, round_mag=True):

    # Reduce dataframe down to relevant information
    ce_df = md_df.filter(regex="m-Ce.*")
    ce_cols = [ce for ce in ce_df.columns if ce_df[ce].max() > mag_thresh]
    ce_df = ce_df[ce_df.columns.intersection(ce_cols)]

    # Get series and list of magnetizations for further analysis
    mag_series = ce_df.sum(axis=1)
    if round_mag:
        mag_series = mag_series.round()  # round series here?
    # mag_df = mag_series.to_frame(name='tot_mag')
    array_out = mag_series.to_numpy()

    return array_out


def extract_md_lifetimes(data):
    summed_lifetimes = {k: None for k in data.keys()}

    for key, value in data.items():
        summed_lifetimes[key] = sum(value)

    all_os = list(range(5))
    final_data = []

    for mos in all_os:
        try:
            final_data.append(summed_lifetimes[mos])
        except KeyError:
            final_data.append(0)

    return final_data


def plotlyfig2json(fig, fpath=None):
    """
    Serialize a plotly figure object to JSON so it can be persisted to disk.
    Figure's persisted as JSON can be rebuilt using the plotly JSON chart API:

    http://help.plot.ly/json-chart-schema/

    If `fpath` is provided, JSON is written to file.

    Modified from https://github.com/nteract/nteract/issues/1229
    """

    redata = json.loads(json.dumps(fig.data, cls=PlotlyJSONEncoder))
    relayout = json.loads(json.dumps(fig.layout, cls=PlotlyJSONEncoder))

    fig_json = json.dumps({"data": redata, "layout": relayout})

    if fpath:
        with open(fpath, "w") as f:
            f.write(fig_json)
    else:
        return fig_json


def plotlyfromjson(fpath):
    """Render a plotly figure from a json file"""
    with open(fpath, "r") as f:
        v = json.loads(f.read())

    fig = go.Figure(data=v["data"], layout=v["layout"])

    return fig


def l1(predict, reference):
    try:
        return np.mean([abs(x - y) for x, y in zip(predict, reference)])
    except TypeError:
        print("Error: both entries must be 1-D lists of the same length")


def l2(predict, reference):
    try:
        return m.sqrt(np.mean([(x - y) ** 2 for x, y in zip(predict, reference)]))
    except TypeError:
        print("Error: both entries must be 1-D lists of the same length")


def write_to_html_file(df, title="", filename="out.html"):
    """
    Write an entire dataframe to an HTML file with nice formatting.
    https://stackoverflow.com/questions/47704441/applying-styling-to-pandas-dataframe-saved-to-html-file
    """

    result = """
<html>
<head>
<style>

    h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
    }
    table {
        margin-left: 3px;
        margin-right: 3px;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
    }
    table tbody tr:hover {
        background-color: #dddddd;
    }
    .wide {
        width: 90%;
    }

</style>
</head>
<body>
    """
    result += "<h2> %s </h2>\n" % title
    if type(df) == Styler:
        result += df.render()
    else:
        result += df.to_html(classes="wide", escape=False)
    result += """
</body>
</html>
"""
    with open(filename, "w") as f:
        f.write(result)
