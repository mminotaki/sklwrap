import copy

import numpy as np
from pandas.api.types import is_numeric_dtype

from .data import *
from .misc import *


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
