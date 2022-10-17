from .data import *
from .misc import *
import re
import numpy as np
import pandas as pd
import math
from pandas.api.types import is_numeric_dtype
from pprint import pprint
import copy


def thresh_helper(x):
    if x < RADIUS_THRESH:
        return x
    else:
        return np.nan


def ce3onn_helper(x):
    try:
        if x < RADIUS_THRESH:
            return x
        else:
            return np.nan
    except TypeError:
        return x


def sampling_df_remove(df_in_atom):

    df_out_atom = df_in_atom.copy(deep=True)

    # Split removal of problematic calcs (e.g. Ce3 in fixed part of slab) and actual subsurface calcs.
    # Use threshold of 0.6 for subsurface and subsubsurface Ce3 removal.
    calc_problem_list = list(
        df_out_atom.loc[
            (df_out_atom["elem"] == "Ce")
            &  # Only consider Ce
            # Ce3 localized in fixed part of slab
            (
                ((df_out_atom["mgn"] > 0.6) & (df_out_atom["id"] <= 18))
                |
                # Ce with mag ~0.5. Cannot use CE3_THRESH for selection here (could for tresh = 0.6, but for thresh = 0.8 different and too many calcs get removed)
                (
                    ((df_out_atom["mgn"] > 0.4) & (df_out_atom["mgn"] < 0.6))
                    & (df_out_atom["id"] != 109)
                )
            )
        ]
        .groupby("dir", sort=False)
        .indices.keys()
    )

    df_out_atom = df_out_atom[~df_out_atom["dir"].isin(calc_problem_list)]

    calc_subsurf_list = list(
        df_out_atom.loc[
            (df_out_atom["elem"] == "Ce")
            &
            # Ce3 localized in subsurface layer
            (
                (df_out_atom["mgn"] > 0.6)
                & (df_out_atom["id"] > 18)
                & (df_out_atom["id"] <= 27)
            )
            # ((df_out_atom['mgn'] > 0.6) & (18 < df_out_atom['id'] >= 27))
        ]
        .groupby("dir", sort=False)
        .indices.keys()
    )

    # Remove 3o1 calculations for now, as these not included in Nathan's dataset
    df_out_atom = df_out_atom.loc[~df_out_atom["dir"].str.contains("3o1_coord")]

    return {
        "df_out_atom": df_out_atom,
        "calc_problem_list": calc_problem_list,
        "calc_subsurf_list": calc_subsurf_list,
    }


def sampling_df_basics(df_in_atom, df_in_calc, ce_groupby):

    df_out_raw = df_in_atom.copy(deep=True)
    df_out_calc = df_in_calc.copy(deep=True)

    # Add column with metal species for easier access (without str.contains on dir-column later)
    metal_list = re.findall(
        "|".join(["/" + _met + "/" for _met in METALS]),
        "".join(df_out_calc["dir"].to_list()),
    )
    metal_list = [met[1:-1] for met in metal_list]
    df_out_calc.insert(loc=1, column="metal", value=metal_list)

    # Metal-oxygen distances
    mo_dists = (
        df_out_raw.loc[df_out_raw["id"] == 109, Osurf_dists]
        .apply(pd.to_numeric)
        .reset_index(drop=True)
    )

    # Number of O ligands
    ncoord = (
        mo_dists.applymap(lambda x: x < RADIUS_THRESH)
        .apply(sum, axis=1)
        .rename("ncoord")
        .reset_index(drop=True)
    )

    # Oxidation state
    oxstate = pd.Series(
        data=[len(x[1][x[1]["mgn"] > CE3_THRESH]) for x in ce_groupby], dtype=int
    )

    # Metal magnetizations
    metal_mgns = df_out_raw.loc[df_out_raw["id"] == 109]["mgn"].to_list()

    # Hash identifiers
    hash_ids = [
        hash(_) for _ in df_out_raw.loc[df_out_raw["id"] == 109]["dir"].to_list()
    ]

    # Ce3 magnetizations continous
    # Magnetizations already as absolute numbers, can do selection based on > 0.1 or so
    mgn_cont_df = (
        df_in_atom.loc[(df_in_atom["id"] != 109) & (df_in_atom["mgn"] > 0.1)][
            [
                "dir",
                "mgn",
            ]
        ]
        .groupby("dir", sort=False)
        .sum()
        .reset_index()
    )
    mgn_cont_df = mgn_cont_df.rename(columns={"mgn": "mgn_cont"})
    # mgn_cont_var = np.round(df_in_atom.loc[(df_in_atom['id'] != 109) & (df_in_atom['mgn'] > 0.1)][['dir', 'mgn', ]].groupby('dir', sort=False).sum().reset_index()['mgn'].values, 2)
    # mgn_cont_var = np.round(df_in_atom.loc[df_in_atom['id'] != 109][['dir', 'mgn', ]].groupby('dir', sort=False).sum().reset_index()['mgn'].values, 2)

    # Append data to initial dfs
    df_out_calc.insert(loc=3, column="ncoord", value=ncoord)
    df_out_raw.insert(loc=3, column="ncoord", value=per_atom_multiply(itable=ncoord))

    df_out_calc.insert(loc=4, column="mos", value=oxstate)
    df_out_raw.insert(loc=4, column="mos", value=per_atom_multiply(itable=oxstate))

    # The ones where previously no atoms had an absolute mgn > 0.1 will be nans now in mgn_cont. Replace with 0, so
    # that there is no artificial deviation between mos=0 and mgn_cont. That is also the whole point why it is done in
    # this manner: applying threshold, merging, and nan-replacement, instead of allowing for minor float values without
    # the threshold.
    df_out_calc = pd.merge(df_out_calc, mgn_cont_df, on="dir", how="outer")
    df_out_calc["mgn_cont"] = df_out_calc["mgn_cont"].fillna(0)

    # df_out_calc.insert(loc=4, column='mgn_cont', value=mgn_cont_var)
    # df_out_raw.insert(loc=4, column='mgn_cont', value=per_atom_multiply(itable=mgn_cont_var))

    df_out_raw.insert(
        loc=4, column="metal_mgn", value=per_atom_multiply(itable=metal_mgns)
    )
    df_out_calc.insert(loc=4, column="metal_mgn", value=metal_mgns)

    df_out_calc.insert(loc=5, column="hash_id", value=hash_ids)
    df_out_raw.insert(loc=5, column="hash_id", value=per_atom_multiply(itable=hash_ids))

    # Yet another sanity check - Check deviation of continuous mgn and assigned mos to identify calcs with multiple Ce
    # with mgn=0.3, such as Ag-2O, Au-2O, etc.
    diff_list = []
    for mgn, mos in zip(df_out_calc["mgn_cont"], oxstate):
        if mos == 0:
            diff_list.append(mgn)
        else:
            diff_list.append(np.abs(mgn - mos) / mos)

    diff_list = np.array(diff_list)

    df_out_calc["mgn_diff"] = diff_list

    return {"df_out_raw": df_out_raw, "df_out_calc": df_out_calc}


def sampling_df_ien(df_in_calc):

    df_out_calc = df_in_calc.copy(deep=True)

    ie_headers = ["IE_N", "IE_N-1", "cum_IE_N", "cum_IE_N-1"]
    m_ies = np.zeros(shape=(df_out_calc.shape[0], 4))

    for irow_tuple, row_tuple in enumerate(df_out_calc.itertuples()):
        if getattr(row_tuple, "mos") != 0:
            for iie_header, ie_header in enumerate(ie_headers):
                if "N-1" not in ie_header:
                    m_ies[irow_tuple, iie_header] = getattr(
                        row_tuple,
                        ie_header.replace("N", str(int(getattr(row_tuple, "mos")))),
                    )
                else:
                    if getattr(row_tuple, "mos") != 1:
                        m_ies[irow_tuple, iie_header] = getattr(
                            row_tuple,
                            ie_header.replace(
                                "N-1", str(int(getattr(row_tuple, "mos") - 1))
                            ),
                        )

    df_out_calc[ie_headers] = m_ies

    return {"df_out_calc": df_out_calc}


def sampling_df_mos(df_in_calc, metal_rows):

    df_out_calc = df_in_calc.copy(deep=True)

    # Default value of 1.94145A is 1/2 of the oxygen separation in the pristine 2O structure.
    # Mean/min/max are calculated on the actual values, without nan.
    # Values are sorted first, and then the default value is inserted.

    # mo_array = np.zeros(shape=(len(metal_rows), 4))

    mo_array = np.empty(shape=(len(metal_rows), 4))
    mo_array[:] = np.nan
    mo_headers = ["d-MO_{}".format(i) for i in range(1, 5)]
    inv_mo_headers = ["1/d-MO_{}".format(i) for i in range(1, 5)]

    for imo_row in list(range(metal_rows.shape[0])):
        mo_array[imo_row, :] = np.sort(
            metal_rows.iloc[[imo_row]][Osurf_dists].applymap(thresh_helper).values
        )[0, :4]

    mo_array = mo_array.round(1)
    inv_mo_array = np.reciprocal(mo_array)

    mo_data = np.concatenate([mo_array, inv_mo_array], axis=1)

    mo_df = pd.DataFrame(data=mo_data, columns=mo_headers + inv_mo_headers)

    mo_df = desc_stats(
        df_in=mo_df, desc_name="d-MO", desc_headers=mo_headers, default=11.65
    )  # 1.94145
    mo_df = desc_stats(
        df_in=mo_df, desc_name="1/d-MO", desc_headers=inv_mo_headers, default=1 / 11.65
    )  # 1.94145

    mo_df["dir"] = metal_rows["dir"].values

    df_out_calc = pd.merge(df_out_calc, mo_df, on="dir", how="outer")

    return {"df_out_calc": df_out_calc}
    # devel_df = pd.merge(ml_df, mo_df, on='dir', how='outer')


def sampling_df_mce3(df_in_calc, ce_groupby):

    df_out_calc = df_in_calc.copy(deep=True)

    mce3_array = np.empty(shape=(len(ce_groupby), 4))
    mce3_array[:] = np.nan

    mce3_headers = ["d-MCe3_" + str(i) for i in range(1, 5)]
    mce3_headers_l = ["d-MCe3_{}_l".format(i) for i in range(1, 5)]
    inv_mce3_headers = ["1/d-MCe3_" + str(i) for i in range(1, 5)]
    inv_mce3_headers_l = ["1/d-MCe3_{}_l".format(i) for i in range(1, 5)]
    counter = 0

    for i, ce_group in ce_groupby:
        ce_df = pd.DataFrame(ce_group)
        sort_dists = sorted(ce_df.loc[(ce_df["mgn"] > CE3_THRESH)]["d-M_109"].values)
        mce3_array[counter, :] = sort_dists + [np.nan] * (4 - len(sort_dists))
        counter += 1

    mce3_array = np.array(mce3_array).astype(float).round(1)
    mce3_df = pd.DataFrame(mce3_array, columns=mce3_headers)
    mce3_df[inv_mce3_headers] = np.reciprocal(mce3_array)

    # Default value of 11.6487A is the length of the lateral lattice vectors in the slab.
    # Mean/min/max are calculated on the actual values, without nan.
    # Values are sorted first, and then the default value is inserted.
    mce3_df = desc_stats(
        df_in=mce3_df, desc_name="d-MCe3", desc_headers=mce3_headers, default=11.65
    )  # 11.6487
    mce3_df = desc_stats(
        df_in=mce3_df,
        desc_name="1/d-MCe3",
        desc_headers=inv_mce3_headers,
        default=1 / 11.65,
    )

    # mce3_array = np.concatenate([mce3_array, inv_mce3_array], axis=1)

    mce3_df_l = pd.DataFrame(mce3_array, columns=mce3_headers_l)
    # print('After copy:')
    # print(mce3_df_l.tail())
    # Need to replace NaNs here, otherwise they get assigned to 1 through the applymap that references values to the lattice vector length.
    mce3_df_l = mce3_df_l.replace(np.nan, 3882.9)
    # print('Replace nan with dummy value:')
    # print(mce3_df_l.tail())
    # # print(mce3_df_l[-5:])
    mce3_df_l = mce3_df_l.applymap(
        lambda x: min(
            [1 / np.sqrt(2), np.sqrt(10) / 2, 1.5 * np.sqrt(2), 1000],
            key=lambda y: abs(y - x / 3.8829),
        )
    )
    # print('Carry out lattice vector referencing:')
    # print(mce3_df_l.tail())
    # # The 3882.9/1000 is just a dummy value so that the NaNs don't get lost during lattice vector referencing, and can be recovered for `desc_stats`.
    mce3_df_l = mce3_df_l.replace(1000, np.nan)
    # print('Replaced dummy value again with nan for statistical evaluation:')
    # print(mce3_df_l.tail())

    mce3_df_l[inv_mce3_headers_l] = np.reciprocal(mce3_df_l.to_numpy())
    mce3_df_l = desc_stats(
        df_in=mce3_df_l, desc_name="d-MCe3_l", desc_headers=mce3_headers_l, default=3
    )
    mce3_df_l = desc_stats(
        df_in=mce3_df_l,
        desc_name="1/d-MCe3_l",
        desc_headers=inv_mce3_headers_l,
        default=1 / 3,
    )

    # Replace nan by default only after desc_stats
    df_out_calc = pd.concat([df_out_calc, mce3_df, mce3_df_l], axis=1)

    return {"df_out_calc": df_out_calc}


def sampling_df_moce(df_in_atom, df_in_calc, metal_rows):

    df_out_calc = df_in_calc.copy(deep=True)

    # 1) Get list of oxygen ligand IDs (or in this case the headers of the corresponding distance terms)
    mo_ce_df = metal_rows[Osurf_dists].applymap(thresh_helper)
    mo_ce_df = pd.concat([metal_rows["dir"], mo_ce_df], axis=1)
    o_headers = []
    for name, iterrow in mo_ce_df.iterrows():
        o_headers.append(
            [
                iterrow[iterrow == mo].index[0]
                for mo in sorted(iterrow[Osurf_dists].values)[:8]
                if mo < RADIUS_THRESH
            ]
        )

    # 2) Check surface cerium atoms and collect the ones with a distance smaller than threshold (in bulk d=2.4, next-nearest d=4.5)
    # omce_array = np.zeros(shape=(metal_rows['dir'].size, 8))
    omce_array = np.empty(shape=(metal_rows["dir"].size, 8))
    omce_array[:] = np.nan

    omce_headers = ["d-O[M]Ce3_" + str(i) for i in range(1, 9)]
    inv_omce_headers = ["1/d-O[M]Ce3_" + str(i) for i in range(1, 9)]

    for icalculation, calculation in enumerate(metal_rows["dir"].to_list()):
        temp_df = df_in_atom.loc[
            (df_in_atom["dir"] == calculation)
            & (df_in_atom["elem"] == "Ce")
            & (df_in_atom["id"] > 26)
        ]
        o_header = o_headers[icalculation]
        dist_df = temp_df[o_header].applymap(thresh_helper)

        # Calculate respective distance lists
        dist_list = sorted(
            [dist for dist in dist_df.values.flatten().tolist() if not math.isnan(dist)]
        )
        omce_array[icalculation, :] = dist_list + [np.nan] * (8 - len(dist_list))

    omce_array = omce_array.round(1)
    inv_omce_array = np.reciprocal(omce_array)
    omce_data = np.concatenate([omce_array, inv_omce_array], axis=1)

    omce_df = pd.DataFrame(omce_data, columns=omce_headers + inv_omce_headers)

    omce_df.insert(loc=0, column="dir", value=metal_rows["dir"].to_list())

    # Default value of 2.37784A is the Ce-O distance in the bulk (botton of the slab).
    # Mean/min/max are calculated on the actual values, without nan.
    # Values are sorted first, and then the default value is inserted.
    omce_df = desc_stats(
        df_in=omce_df, desc_name="d-O[M]Ce3", desc_headers=omce_headers, default=11.65
    )  # 2.37784
    omce_df = desc_stats(
        df_in=omce_df,
        desc_name="1/d-O[M]Ce3",
        desc_headers=inv_omce_headers,
        default=1 / 11.65,
    )

    df_out_calc = pd.merge(df_out_calc, omce_df, on="dir", how="outer")

    return {"df_out_calc": df_out_calc}


def sampling_df_ce3ce3(df_in_calc, ce_groupby):

    df_out_calc = df_in_calc.copy(deep=True)

    # Default value of 11.6487A is the length of the lateral lattice vectors in the slab.
    # Mean/min/max are calculated on the actual values, without nan.
    # Values are sorted first, and then the default value is inserted.

    # Maximum of 6 distinct combinations between 4 Ce3 (in the case of Rh) possible

    ce3_ce3_headers = ["d-Ce3Ce3_" + str(i) for i in range(1, 7)]
    ce3_ce3_headers_l = ["d-Ce3Ce3_{}_l".format(i) for i in range(1, 7)]
    inv_ce3_ce3_headers = ["1/d-Ce3Ce3_" + str(i) for i in range(1, 7)]
    inv_ce3_ce3_headers_l = ["1/d-Ce3Ce3_{}_l".format(i) for i in range(1, 7)]

    # ce3_ce3_array = np.zeros(shape=(len(ce_groupby), 6))
    ce3_ce3_array = np.empty(shape=(len(ce_groupby), 6))
    ce3_ce3_array[:] = np.nan

    counter = 0
    for dir_name, ce_group in ce_groupby:
        ce_df = pd.DataFrame(ce_group)
        ce3_df = ce_df.loc[(ce_df["mgn"] > CE3_THRESH)]
        if ce3_df.shape[0] < 2:
            ce3_ce3_array[
                counter, :
            ] = np.nan  # 11.6487 # Set lattice distance here instead of NaN
        else:
            calc_dist_list = []
            for ce3_id_comb in it.combinations(ce3_df["id"].values, 2):
                calc_dist_list.append(
                    ce3_df.loc[ce3_df["id"] == ce3_id_comb[0]][
                        "d-Ce_" + str(int(ce3_id_comb[1]))
                    ].values[0]
                )
            ce3_ce3_array[counter, :] = sorted(calc_dist_list) + [np.nan] * (
                6 - len(calc_dist_list)
            )
        #             print(ce3_ce3_array[counter,:])

        counter += 1

    ce3_ce3_array = ce3_ce3_array.round(1)

    ce3_ce3_df = pd.DataFrame(ce3_ce3_array, columns=ce3_ce3_headers)
    ce3_ce3_df[inv_ce3_ce3_headers] = np.reciprocal(ce3_ce3_array)

    ce3_ce3_df = desc_stats(
        df_in=ce3_ce3_df,
        desc_name="d-Ce3Ce3",
        desc_headers=ce3_ce3_headers,
        default=11.65,
    )  # 11.6487
    ce3_ce3_df = desc_stats(
        df_in=ce3_ce3_df,
        desc_name="1/d-Ce3Ce3",
        desc_headers=inv_ce3_ce3_headers,
        default=1 / 11.65,
    )

    ce3_ce3_df_l = pd.DataFrame(ce3_ce3_array, columns=ce3_ce3_headers_l)
    #     print('After copy:')
    #     print(ce3_ce3_df_l.tail())
    # Need to replace NaNs here, otherwise they get assigned to 1 through the applymap that references values to the lattice vector length.
    ce3_ce3_df_l = ce3_ce3_df_l.replace(np.nan, 3882.9)
    #     print('Replace nan with dummy value:')
    #     print(ce3_ce3_df_l.tail())
    # print(ce3_ce3_df_l[-5:])
    ce3_ce3_df_l = ce3_ce3_df_l.applymap(
        lambda x: min([1, np.sqrt(2), 3, 1000], key=lambda y: abs(y - x / 3.8829))
    )
    #     print('Carry out lattice vector referencing:')
    #     print(ce3_ce3_df_l.tail())
    # The 3882.9/1000 is just a dummy value so that the NaNs don't get lost during lattice vector referencing, and can be recovered for `desc_stats`.
    ce3_ce3_df_l = ce3_ce3_df_l.replace(1000, np.nan)
    #     print('Replaced dummy value again with nan for statistical evaluation:')
    #     print(ce3_ce3_df_l.tail())
    #     ce3_ce3_df_l = ce3_ce3_df_l.rename(columns=dict(zip(ce3_ce3_df.columns, ce3_ce3_headers_l)))
    #     print('After rename:')
    #     print(ce3_ce3_df_l.tail())

    #     print(ce3_ce3_df_l.tail())

    ce3_ce3_df_l[inv_ce3_ce3_headers_l] = np.reciprocal(ce3_ce3_df_l.to_numpy())

    ce3_ce3_df_l = desc_stats(
        df_in=ce3_ce3_df_l,
        desc_name="d-Ce3Ce3_l",
        desc_headers=ce3_ce3_headers_l,
        default=3,
    )
    ce3_ce3_df_l = desc_stats(
        df_in=ce3_ce3_df_l,
        desc_name="1/d-Ce3Ce3_l",
        desc_headers=inv_ce3_ce3_headers_l,
        default=1 / 3,
    )

    # full_df.loc[full_df['min(d-MCe3)'] < 5, 'min(d-MCe3)_l'] = 1/np.sqrt(2)
    # full_df.loc[(full_df['min(d-MCe3)'] >= 5) & (full_df['min(d-MCe3)'] < 7.5), 'min(d-MCe3)_l'] = np.sqrt(10)/2
    # full_df.loc[(full_df['min(d-MCe3)'] >= 7.5) & (full_df['min(d-MCe3)'] < 10), 'min(d-MCe3)_l'] = 3/2*np.sqrt(2)
    # full_df.loc[10 <= full_df['min(d-MCe3)'], 'min(d-MCe3)_l'] = 3

    df_out_calc = pd.concat([df_out_calc, ce3_ce3_df, ce3_ce3_df_l], axis=1)

    return {"df_out_calc": df_out_calc}


def sampling_df_ce3onn(df_in_atom, df_in_calc, ce_groupby):

    df_out_calc = df_in_calc.copy(deep=True)

    ce3_ONN_sepd, ce3_ONN_shared = (
        [],
        [],
    )  # O atoms that are bound to separate Ce3, or that are shared by two Ce3

    for prep_calc_tuple in df_out_calc.itertuples(index=False):
        ceo_dict = {}
        row_df = df_in_atom.loc[df_in_atom["dir"] == prep_calc_tuple[0]]
        surf_ce = row_df.loc[
            (row_df["elem"] == "Ce")
            & (row_df["id"] >= 19)
            & (row_df["mgn"] > CE3_THRESH)
        ]
        surf_ce_ids = surf_ce["id"].values

        surf_ce_Os = (
            surf_ce[Osurf_dists]
            .applymap(ce3onn_helper)
            .dropna(axis=1, how="all")
            .reset_index(drop=True)
        )

        # The values of the ceo_dict are 0-based bc these are extracted from the headers of the csv-file, where they are also
        # 0-based. Leave for now as we are here only concerned with their number, otherwise change that in the bash script to
        # extract the data.
        surf_ce_Os.columns = surf_ce_Os.columns.str.replace("d-O_", "")
        for ice_id, ce_id in enumerate(surf_ce_ids):
            ceo_dict[ce_id] = list(surf_ce_Os.iloc[ice_id].dropna().index.astype("int"))

        # pprint(prep_calc_tuple._asdict(), sort_dicts=False)
        # print(getattr(prep_calc_tuple, 'Ce3_ids'))
        try:
            ce3_ids = getattr(prep_calc_tuple, "Ce3_ids").split("_")
            if ce3_ids[0] == "0":
                ce3_ONN_sepd.append(0)
                ce3_ONN_shared.append(0)
            else:
                if len(ce3_ids) == 1:
                    try:
                        ce3_ONN_sepd.append(len(ceo_dict[int(ce3_ids[0])]))
                    except ValueError:
                        ce3_ONN_sepd.append(0)
                    ce3_ONN_shared.append(0)

                elif len(ce3_ids) >= 2:
                    # print(ceo_dict)
                    # print([ce3_id for ce3_id in ce3_ids])
                    o_setlist = [set(ceo_dict[int(ce3_id)]) for ce3_id in ce3_ids]
                    o_flatlist = [item for sublist in o_setlist for item in sublist]

                    shared = []
                    for o_set_comb in it.combinations(o_setlist, 2):
                        single_comb_set = set.intersection(*o_set_comb)
                        if len(single_comb_set) > 0:
                            shared.append(*set.intersection(*o_set_comb))

                    separated = set([sepd for sepd in o_flatlist if sepd not in shared])

                    ce3_ONN_sepd.append(len(separated))
                    ce3_ONN_shared.append(len(shared))

        except AttributeError:
            # AttributeError exception not needed anymore, as no empty string for OS=0, causing nan on split.
            # => Somehow still needed. For whatever reason: `ag/2o_coord/1ce3/pos_28/sec-gamma/` has nan.
            ce3_ONN_sepd.append(0)
            ce3_ONN_shared.append(0)

    df_out_calc["Ce3-ONN_sepd"] = ce3_ONN_sepd
    df_out_calc["Ce3-ONN_shared"] = ce3_ONN_shared

    # Reference these features to number of Ce31

    # General oxygen per Ce3-descriptor, taking both separated and shared into account
    df_out_calc["o_per_ce3"] = (
        df_out_calc["Ce3-ONN_sepd"] + 2 * df_out_calc["Ce3-ONN_shared"]
    ) / df_out_calc["mos"]

    # Shared and separated oxygen per cerium atom
    df_out_calc["o_per_ce3_shared"] = df_out_calc["Ce3-ONN_shared"] / df_out_calc["mos"]
    df_out_calc["o_per_ce3_sepd"] = df_out_calc["Ce3-ONN_sepd"] / df_out_calc["mos"]

    # Number of oxygen in isolated layer... again per Ce3.
    ce_layer_frac = []
    count = 0
    for ce_dist in df_out_calc["Ce3_ids"]:
        ce_layer_frac.append(
            len([_ for _ in ce_dist.split("_") if _ in ["30", "33", "36"]])
        )
        count += 1

    df_out_calc["ce_layer_frac"] = ce_layer_frac
    df_out_calc["ce_layer_frac"] = df_out_calc["ce_layer_frac"] / df_out_calc["mos"]

    fill_cols = ["o_per_ce3", "o_per_ce3_shared", "o_per_ce3_sepd", "ce_layer_frac"]

    for col in fill_cols:
        df_out_calc[col] = df_out_calc[col].fillna(0)

    return {"df_out_calc": df_out_calc}


def sampling_df_energies(df_in_calc):

    df_out_calc = df_in_calc.copy(deep=True)

    metal_df_list = []

    for metal in METALS:
        metal_df = df_out_calc.loc[df_out_calc["metal"] == metal]
        metal_ncoord_mins = []

        # Add relative energies referenced to per-metal energy minimum.
        metal_df["E_rel_metal"] = metal_df["ener"] - metal_df["ener"].min()
        metal_df["E_rel_avg"] = metal_df["ener"] - metal_df["ener"].mean()

        for ncoord in [2, 3, 4]:
            metal_ncoord_df = metal_df.loc[metal_df["ncoord"] == ncoord]
            metal_ncoord_mins.extend(
                [metal_ncoord_df["ener"].min()] * metal_ncoord_df.shape[0]
            )

        # Add relative energies referenced against 2O/respective surface and the bulk metal
        ener_2o = []
        ener_surf = []

        for df_tuple in metal_df.itertuples():
            ener_surf.append(
                getattr(df_tuple, "ener")
                - bulk_mes[getattr(df_tuple, "metal")]
                - ceo_pris["3x3"][str(getattr(df_tuple, "ncoord")) + "o"]
            )
            ener_2o.append(
                getattr(df_tuple, "ener")
                - bulk_mes[getattr(df_tuple, "metal")]
                - ceo_pris["3x3"]["2o"]
            )

        metal_df["E_rel_global"], metal_df["E_rel_surf"] = ener_2o, ener_surf
        metal_df["E_rel_metal_ncoord"] = metal_df["ener"].to_numpy() - np.array(
            metal_ncoord_mins
        )

        # Keep only lowest one for each Ce3-ID
        metal_df = (
            metal_df.sort_values("ener", ascending=True)
            .drop_duplicates(subset=["Ce3_ids", "ncoord", "mos"])
            .sort_index()
        )

        metal_df_list.append(metal_df)

    df_out_calc = pd.concat(metal_df_list)

    return {"df_out_calc": df_out_calc}


def sampling_df_imputation(df_in_calc):

    df_out_calc = df_in_calc.copy(deep=True)

    clean_columns = df_out_calc.columns[df_out_calc.isna().any()].tolist()

    metal_df_list = []
    for metal in METALS:
        metal_df = df_out_calc.loc[df_out_calc["metal"] == metal]

        # Add relative energies referenced to per-metal energy minimum.
        metal_min = metal_df["ener"].min()
        metal_df["E_rel_metal"] = metal_df["ener"] - metal_min

        # Add relative energies referenced against 2O/respective surface and the bulk metal
        ener_2o = []
        ener_surf = []

        for df_tuple in metal_df.itertuples():
            ener_surf.append(
                getattr(df_tuple, "ener")
                - bulk_mes[getattr(df_tuple, "metal")]
                - ceo_pris["3x3"][str(int(getattr(df_tuple, "ncoord"))) + "o"]
            )
            ener_2o.append(
                getattr(df_tuple, "ener")
                - bulk_mes[getattr(df_tuple, "metal")]
                - ceo_pris["3x3"]["2o"]
            )

        metal_df["E_rel_global"], metal_df["E_rel_surf"] = ener_2o, ener_surf

        for clean_column in clean_columns:
            if clean_column != "Ce3_ids":
                metal_df[clean_column] = metal_df[clean_column].replace(
                    np.nan, metal_df[clean_column].mean()
                )
            else:
                metal_df[clean_column] = metal_df[clean_column].replace(np.nan, "")

        # Keep only lowest one for one Ce3-ID
        metal_df = (
            metal_df.sort_values("ener", ascending=True)
            .drop_duplicates(subset=["Ce3_ids", "ncoord", "mos"])
            .sort_index()
        )

        metal_df_list.append(metal_df)

    df_out_calc = pd.concat(metal_df_list)

    return {"df_out_calc": df_out_calc}


def sampling_df_labels(df_in_calc):

    df_out_calc = df_in_calc.copy(deep=True)
    plot_labels = []

    for i in range(df_out_calc.shape[0]):
        row_df = df_in_calc.iloc[[i]]

        label_str = "M: {}; Coord: {}; OS: {}; IDs: {}; mean(d-Ce3Ce3): {}; Ce3-ONN: ({},{})".format(
            row_df["metal"].values[0].title(),
            row_df["ncoord"].values[0],
            row_df["mos"].values[0],
            row_df["Ce3_ids"].values[0],
            round(row_df["mean(d-Ce3Ce3)"].values[0], 3),
            row_df["Ce3-ONN_sepd"].values[0],
            row_df["Ce3-ONN_shared"].values[0],
        )

        label_str += "<br>E (tot): {}; E (met): {}; E (glob): {}".format(
            round(row_df["ener"].values[0], 3),
            round(row_df["E_rel_metal"].values[0], 3),
            round(row_df["E_rel_global"].values[0], 3),
        )

        label_str += "<br>Dir: {}".format(
            row_df["dir"].values[0][row_df["dir"].values[0].find("/calcs/") :]
        )

        plot_labels.append(label_str)

    df_out_calc["plot_label"] = plot_labels

    return {"df_out_calc": df_out_calc}


def easy_combinatorics(df_in, features_in, features_only=False):
    combinatorics_df = df_in[features_in]

    # Get immutable tuple for iteration in the for loops. Otherwise gets changed constantly and they run forever.
    column_list = copy.deepcopy(sorted(list(combinatorics_df.columns)))
    counter = 0

    if features_only is False:
        for icol1, col1 in enumerate(column_list):
            for icol2, col2 in enumerate(column_list):
                if (
                    col1 != col2
                    and is_numeric_dtype(combinatorics_df[col1])
                    and is_numeric_dtype(combinatorics_df[col2])
                ):

                    # Switching order of factors in product leads to same result, therefore next if with greater statement.
                    if icol2 > icol1:
                        combinatorics_df["{} * {}".format(col1, col2)] = (
                            combinatorics_df[col1] * combinatorics_df[col2]
                        )

                    # Order matters for fraction, therefore no greater than clause, but only checking if denominator is zero
                    if not (combinatorics_df[col2] == 0).any():
                        combinatorics_df["{} / {}".format(col1, col2)] = (
                            combinatorics_df[col1] / combinatorics_df[col2]
                        )

        # Include primary features in combinatorics df?
        features_out = [
            feature_out
            for feature_out in list(tuple(combinatorics_df.columns))
            if feature_out not in features_in
        ]

        return {"df": combinatorics_df, "features": features_out}

    else:
        features_out = []
        for icol1, col1 in enumerate(column_list):
            for icol2, col2 in enumerate(column_list):
                if (
                    col1 != col2
                    and is_numeric_dtype(combinatorics_df[col1])
                    and is_numeric_dtype(combinatorics_df[col2])
                ):

                    # Switching order of factors in product leads to same result, therefore next if with greater statement.
                    if icol2 > icol1:
                        features_out.append("{} * {}".format(col1, col2))

                    # Order matters for fraction, therefore no greater than clause, but only checking if denominator is zero
                    if not (combinatorics_df[col2] == 0).any():
                        features_out.append("{} / {}".format(col1, col2))

        return {"df": None, "features": features_out}


def desc_stats(df_in, desc_name, desc_headers, default=None):

    if "1/" in desc_headers[0]:
        round_to = 3
    else:
        round_to = 2

    df_out = df_in.copy(deep=True)

    # Each of the following pandas operations returns nan if all the input values are nan.

    # d-MO is never nan
    # d-MCe3 is nan for OS=0
    # d-O[M]Ce3 is nan for OS=0
    # d-Ce3_Ce3 is nan for OS < 2

    # These three are evaluated before the NaNs are imputed.
    # NaNs for these three were not imputed but kept instead. Now replace by default here
    df_out["min({})".format(desc_name)] = (
        df_in.loc[:, desc_headers].min(axis=1).round(round_to).replace(np.nan, default)
    )
    df_out["mean({})".format(desc_name)] = (
        df_in.loc[:, desc_headers].mean(axis=1).round(round_to).replace(np.nan, default)
    )
    df_out["max({})".format(desc_name)] = (
        df_in.loc[:, desc_headers].max(axis=1).round(round_to).replace(np.nan, default)
    )
    # Here,
    df_out["std({})".format(desc_name)] = (
        df_in.loc[:, desc_headers].std(axis=1).replace(np.nan, 0).round(3)
    )
    # Here, NaNs are imputed by default values and then the sum is calculated
    df_out["sum({})".format(desc_name)] = (
        df_in.loc[:, desc_headers].replace(np.nan, default).sum(axis=1).round(round_to)
    )

    # Replace the initial nan values that were ignored in the previous statistical evaluation here by nan, so that
    # I don't have to do this afterwards separately for the direct and inverse distances.
    df_out[desc_headers] = df_out[desc_headers].replace(np.nan, default)

    return df_out
