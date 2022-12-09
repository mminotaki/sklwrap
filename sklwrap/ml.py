import copy
from operator import itemgetter

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import trange

# from .data import *
# from .misc import apply_feat_mask
from .vis import plot_errors, plot_regr

NEED_TO_STANDARDIZE = (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    SVR,
)


def run_regr(
    df_in,
    ml_model,
    ml_features,
    ml_target,
    cv_setup={"cv_type": "kfold", "cv_spec": 5},
):
    # ! Don't evaluate the errors here??
    df_func = df_in.copy(deep=True)
    num_rows = df_func.shape[0]
    y_full = df_func[ml_target].values

    # Initialize all the empty lists that hold all the data for the return dictionary.
    ml_models = []
    scalers = []

    rmse_trains, rmse_tests, rmse_fulls = [], [], []
    mae_trains, mae_tests, mae_fulls = [], [], []
    rsquared_trains, rsquared_tests, rsquared_fulls = [], [], []

    # Add splitting indices to Dataframe based on K-Fold or LOGOCV. Could externalize this as function.
    # This is done so that they can be treated with the same footing later on when the data is standardized and the models are trained.
    # TODO: Move this to a separate function.
    if cv_setup["cv_type"].lower() == "kfold":
        split_column_number = cv_setup["cv_spec"]
        split_array = np.full((num_rows, split_column_number), False)

        kf = KFold(n_splits=cv_setup["cv_spec"], shuffle=False)
        for isplit, split in enumerate(kf.split(range(num_rows))):
            train_indices, test_indices = split
            split_array[[train_indices], isplit] = True

    elif cv_setup["cv_type"].lower() == "logocv":
        logocv_column = cv_setup["cv_spec"]
        split_column_number = len(set(df_func[logocv_column]))
        split_array = np.full((num_rows, split_column_number), False)
        for ilogocv_value, logocv_value in enumerate(
            set(df_func[logocv_column].values)
        ):
            split_array[:, ilogocv_value] = [
                _ == logocv_value for _ in df_func[logocv_column].values
            ]

    elif cv_setup["cv_type"].lower() == "loocv":
        raise NotImplementedError("LOOCV not implemented yet.")

    split_column_names = ["train_{:05d}".format(i) for i in range(split_column_number)]
    pred_column_names = ["pred_{:05d}".format(i) for i in range(split_column_number)]

    df_bool = pd.DataFrame(
        data=split_array, columns=split_column_names, index=df_func.index
    )
    df_func = pd.concat([df_func, df_bool], axis=1)
    y_pred_arrays = []

    for split_column_name, pred_column_name in list(
        zip(split_column_names, pred_column_names)
    ):

        split_column_bool = df_func[split_column_name].values
        # print('split_column_name', split_column_name)

        df_train = df_func[split_column_bool].copy(deep=True)
        df_test = df_func[np.logical_not(split_column_bool)].copy(deep=True)

        x_train = df_train[ml_features].values
        x_test = df_test[ml_features].values
        y_train = df_train[ml_target].values
        y_test = df_test[ml_target].values

        # Standardization within data splitting to avoid data leakage from testing data.
        if isinstance(ml_model, NEED_TO_STANDARDIZE):
            train_scaler = StandardScaler().fit(x_train)
            x_train = train_scaler.transform(x_train)
            x_test = train_scaler.transform(x_test)
            scalers.append(train_scaler)

        # TODO: Maybe it'll work the same way, by just feeding a pipeline? Insert data standardization into pipeline???
        # if isinstance(ml_model, Pipeline):
        #     print("Your ML model is a pipeline. I expect data standardization to be in the pipeline, if it is needed.")

        # Fit and predict
        _ = ml_model.fit(x_train, y_train)
        y_train_pred, y_test_pred = ml_model.predict(x_train), ml_model.predict(x_test)

        # ! Using np.concatenate will always put the test values behind the train values.
        # y_full_pred = np.concatenate([y_train_pred, y_test_pred])

        df_train[pred_column_name] = y_train_pred
        df_test[pred_column_name] = y_test_pred

        df_pred_full = pd.concat([df_train, df_test]).sort_index()

        # ! Use pd.DataFrame to merge on index and keep ordering, however, in the end just need the pred column.
        y_pred_ordered = df_pred_full[pred_column_name].values
        y_pred_arrays.append(y_pred_ordered)

        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        rmse_full = mean_squared_error(y_full, y_pred_ordered, squared=False)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mae_full = mean_absolute_error(y_full, y_pred_ordered)
        rsquared_train = r2_score(y_train, y_train_pred)
        rsquared_test = r2_score(y_test, y_test_pred)
        rsquared_full = r2_score(y_full, y_pred_ordered)

        ml_models.append(ml_model)
        rmse_trains.append(rmse_train)
        rmse_tests.append(rmse_test)
        rmse_fulls.append(rmse_full)
        mae_trains.append(mae_train)
        mae_tests.append(mae_test)
        mae_fulls.append(mae_full)
        rsquared_trains.append(rsquared_train)
        rsquared_tests.append(rsquared_test)
        rsquared_fulls.append(rsquared_full)

    # Concatenate the sorted predictions into df_func.
    df_func = pd.concat(
        [
            df_func,
            pd.DataFrame(data=np.array(y_pred_arrays).T, columns=pred_column_names),
        ],
        axis=1,
    )

    df_func = df_func.rename(columns={ml_target: "y"})

    # ! Visualisation for development.
    # for split_column_name, pred_column_name in list(zip(split_column_names, pred_column_names)):
    #     print(split_column_name, pred_column_name)
    #     fig = px.scatter(df_func, x="mV", y=pred_column_name, color=split_column_name)
    #     _ = fig.update_layout(regr_layout)
    #     x_range = df_func["mV"].min(), df_func["mV"].max()
    #     x_range_ext = (x_range[0] - 0.075*np.ptp(x_range), x_range[1] + 0.075*np.ptp(x_range))
    #     _ = fig.update_xaxes(range=x_range_ext)
    #     fig.show()

    error_dict = {
        "rmse_trains": rmse_trains,
        "rmse_tests": rmse_tests,
        "rmse_fulls": rmse_fulls,
        "mae_trains": mae_trains,
        "mae_tests": mae_tests,
        "mae_fulls": mae_fulls,
        "rsquared_trains": rsquared_trains,
        "rsquared_tests": rsquared_tests,
        "rsquared_fulls": rsquared_fulls,
    }

    best_id = "{0:05d}".format(np.argmin(rmse_tests))

    return {
        "df_in": df_func,
        "cv_setup": cv_setup,
        "ml_models": ml_models,
        "scalers": scalers,
        "error_dict": error_dict,
        "best_id": best_id,
    }


def vary_ml_param(
    df_in,
    ml_base_model,
    ml_features,
    ml_target,
    ml_param_dict,
    cv_setup=None,
    color_setup=None,  # ! Copy-paste passing to plot_regr from outside. Must be improved.
    *args,
    **kwargs,
):

    # Missing: Descriptor figure and number of features list/figure.
    return_dict = {}

    # coefs, numfeatures, models, rmses, y_preds, rsquareds = [], [], [], [], [], []

    counter = 0
    ml_models, test_rmses, regr_runs = [], [], []
    ml_param_values = list(ml_param_dict.values())[0]
    errors_array = np.zeros(shape=(len(ml_param_values), 9))

    for ml_param_value in ml_param_values:

        ml_mod_model = copy.deepcopy(
            ml_base_model.set_params(**{list(ml_param_dict.keys())[0]: ml_param_value})
        )

        ml_param_run = run_regr(
            df_in=df_in,
            ml_model=ml_mod_model,
            ml_features=ml_features,
            ml_target=ml_target,
            cv_setup=cv_setup,
        )

        regr_runs.append(ml_param_run)
        ml_param_model = ml_param_run["ml_models"][int(ml_param_run["best_id"])]
        error_dict = ml_param_run["error_dict"]
        ml_models.append(ml_param_model)

        # If before full CV done, average the errors.
        for error_dict_key in list(error_dict.keys()):
            error_dict[error_dict_key] = [np.mean(error_dict[error_dict_key])]

        test_rmses.append(error_dict["rmse_tests"][0])

        errors_array[counter, :] = [
            error_dict_value[0] for error_dict_value in list(error_dict.values())
        ]

        counter += 1

    best_id = np.argmin(test_rmses)

    return_dict["ml_models"] = ml_models
    return_dict["best_id"] = best_id
    return_dict["best_param"] = ml_param_values[best_id]
    return_dict["test_rmses"] = test_rmses
    return_dict["regr_runs"] = regr_runs

    best_model = ml_models[best_id]

    if len(ml_param_values) == 1:
        best_model_run = ml_param_run
    else:
        best_model_run = run_regr(
            df_in=df_in,
            ml_model=best_model,
            ml_features=ml_features,
            ml_target=ml_target,
            cv_setup=cv_setup,
        )

    # Create energy plots
    # This could now also be done outside. Return just best model and plot the result of best model fit.
    # However, for convenience reasons do it here.

    regr_figs = plot_regr(
        regr_dict=best_model_run,
        color_setup=color_setup,
        regr_layout=kwargs.get("regr_layout", None),
    )

    error_headers = [error_key.replace("_", "s_") for error_key in error_dict.keys()]
    errors_dict = {
        key: value for key, value in zip(error_headers, errors_array.transpose())
    }

    error_fig = plot_errors(
        error_dict=errors_dict,
        x_values=ml_param_values,
        plot_measures=[
            "rmses_trains",
            "rmses_tests",
            "rsquareds_trains",
            "rsquareds_tests",
            "maes_trains",
            "maes_tests",
        ],
        annot_text=[""] * len(ml_param_values),
        x_title=list(ml_param_dict.keys())[0],
    )

    return_dict["regr_figs"] = regr_figs
    return_dict["error_fig"] = error_fig
    return_dict["error_dict"] = errors_dict

    return return_dict


# @timecall
# def run_en_scan(
#     en_df, en_features, en_target, en_alphas, en_l1_ratios, cv_type="kfold", n_splits=5
# ):

#     # If csv exists, read i, otherwise do scan.
#     # l1_ratios lower than 0.1 significantly increase calculation time. Idk why. Try out pure ridge run for timing test.

#     en_numfeat_array = np.zeros(shape=(len(en_l1_ratios), len(en_alphas)))
#     en_rmse_array = np.zeros(shape=(len(en_l1_ratios), len(en_alphas)))

#     for ien_alpha, en_alpha in enumerate(en_alphas):
#         for ien_l1_ratio, en_l1_ratio in enumerate(en_l1_ratios):

#             en_run = run_regr(
#                 df_in=en_df,
#                 ml_features=en_features,
#                 ml_target=en_target,
#                 ml_model=ElasticNet(
#                     alpha=en_alpha,
#                     l1_ratio=en_l1_ratio,
#                     max_iter=int(float("1e5")),
#                     tol=float("1e-3"),
#                     random_state=0,
#                 ),
#                 cv_type=cv_type,
#                 n_splits=N_SPLITS,
#             )

#             selected_features = apply_feat_mask(
#                 model=en_run["ml_models"][
#                     np.argmin(en_run["error_dict"]["rmse_tests"])
#                 ],
#                 model_features=en_features,
#                 feat_thresh=0,
#             )

#             en_cv_best_rmse = np.min(en_run["error_dict"]["rmse_tests"])

#             en_rmse_array[ien_l1_ratio, ien_alpha] = en_cv_best_rmse

#             en_numfeat_array[ien_l1_ratio, ien_alpha] = len(selected_features)

#     numfeat_df = pd.DataFrame(data=en_numfeat_array).apply(
#         pd.to_numeric, downcast="integer"
#     )
#     rmse_df = pd.DataFrame(data=en_rmse_array)

#     return {
#         "en_alphas": en_alphas,
#         "en_l1_ratios": en_l1_ratios,
#         "numfeat_df": numfeat_df,
#         "rmse_df": rmse_df,
#     }


# def loocv_all_points(df_in, ml_model, ml_features, ml_target, column, plot_range):
#     # TODO: Could adapt the other function such that it does the same. Keep somewhat duplicated code here.

#     rsquareds, rmses, maes, regr_figs = [], [], [], []
#     column_unique_vals = df_in[column].unique()

#     for column_value in column_unique_vals:

#         train_df = df_in.loc[
#             df_in[column].isin([_ for _ in column_unique_vals if _ != column_value])
#         ]
#         test_df = df_in.loc[df_in[column] == column_value]

#         X_train, y_train = (
#             train_df[ml_features].to_numpy(),
#             train_df[ml_target].values.ravel(),
#         )
#         X_test, y_test = (
#             test_df[ml_features].to_numpy(),
#             test_df[ml_target].values.ravel(),
#         )

#         if isinstance(ml_model, NEED_TO_STANDARDIZE) is True:
#             train_scaler = StandardScaler().fit(X_train)
#             X_train = train_scaler.transform(X_train)
#             X_test = train_scaler.transform(X_test)

#         _ = ml_model.fit(X_train, y_train)

#         y_pred = ml_model.predict(X_test)

#         # All errors again only on testing data
#         rmse = mean_squared_error(y_test, y_pred, squared=False)
#         mae = mean_absolute_error(y_test, y_pred)
#         rsquared = r2_score(y_test, y_pred)

#         rsquareds.append(rsquared)
#         rmses.append(rmse)
#         maes.append(mae)

#         # print("column_value: {} | RMSE: {:.3f} | MAE: {:.3f}".format(column_value, rmse, mae))

#         # Plot energies
#         ener_fig = go.Figure()

#         # Plot energy data points
#         _ = ener_fig.add_trace(
#             go.Scatter(
#                 x=y_test,
#                 y=y_pred,
#                 text=test_df["plot_label"].tolist(),
#                 mode="markers",
#                 marker=dict(
#                     size=8,
#                     symbol=0,
#                     color=color_dict.get(column_value, "blue"),
#                     opacity=1,
#                 ),
#                 hoverinfo="x+y+text",
#                 showlegend=True,
#                 name=column_value.title(),
#             ),
#         )

#         # Add ideal fit line to plot
#         _ = ener_fig.add_trace(
#             go.Scatter(
#                 x=plot_range,
#                 y=plot_range,
#                 mode="lines",
#                 line=dict(color="rgb(0, 0, 0, 0.1)", width=2, dash="dash"),
#                 hoverinfo="skip",
#                 showlegend=False,
#             ),
#         )

#         _ = ener_fig.add_annotation(
#             xanchor="left",
#             yanchor="top",
#             xref="paper",
#             yref="paper",
#             x=0,
#             y=1,
#             align="left",
#             text="R<sup>2</sup> = {:.3f}<br>RMSE = {:.3f}<br>MAE = {:.3f}".format(
#                 rsquared, rmse, mae
#             ),
#             font_size=26,
#             font_family="Arial",
#             showarrow=False,
#             bgcolor="rgba(0,0,0,0.1)",
#         )

#         _ = ener_fig.update_layout(energy_layout)
#         range_layout = go.Layout(xaxis_range=plot_range, yaxis_range=plot_range)
#         _ = ener_fig.update_layout(range_layout)
#         regr_figs.append(ener_fig)

#     return {
#         "regr_figs": regr_figs,
#         "rsquareds": rsquareds,
#         "rmses": rmses,
#         "maes": maes,
#     }
