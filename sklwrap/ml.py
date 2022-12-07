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
# from .vis import plot_energies, plot_errors

NEED_TO_STANDARDIZE = (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    SVR,
)


# def add_split_columns(

# ):
#     pass


# def vary_ml_param(
#     df_in,
#     ml_base_model,
#     ml_features,
#     ml_target,
#     ml_param_dict,
#     cv_type="kfold",
#     n_splits=5,
#     verbose=False,
# ):

#     # Missing: Descriptor figure and number of features list/figure.
#     return_dict = {}

#     # coefs, numfeatures, models, rmses, y_preds, rsquareds = [], [], [], [], [], []

#     counter = 0
#     ml_models, test_rmses, feature_coefs = [], [], []
#     ml_param_values = list(ml_param_dict.values())[0]
#     errors_array = np.zeros(shape=(len(ml_param_values), 9))

#     for ml_param_value in ml_param_values:

#         ml_mod_model = copy.deepcopy(
#             ml_base_model.set_params(**{list(ml_param_dict.keys())[0]: ml_param_value})
#         )

#         ml_param_run = run_regr(
#             df_in=df_in,
#             ml_model=ml_mod_model,
#             ml_features=ml_features,
#             ml_target=ml_target,
#             cv_type=cv_type,
#             n_splits=n_splits,
#         )

#         ml_param_model = ml_param_run["ml_models"][ml_param_run["best_id"]]
#         error_dict = ml_param_run["error_dict"]
#         result_dict = ml_param_run["result_dict"]

#         ml_models.append(ml_param_model)

#         # If before full CV done, average the errors.
#         for error_dict_key in list(error_dict.keys()):
#             error_dict[error_dict_key] = [np.mean(error_dict[error_dict_key])]

#         test_rmses.append(error_dict["rmse_tests"][0])

#         errors_array[counter, :] = [
#             error_dict_value[0] for error_dict_value in list(error_dict.values())
#         ]

#         counter += 1

#     best_id = np.argmin(test_rmses)

#     return_dict["ml_models"] = ml_models
#     return_dict["best_id"] = best_id
#     return_dict["best_param"] = ml_param_values[best_id]
#     return_dict["test_rmses"] = test_rmses

#     best_model = ml_models[best_id]

#     if len(ml_param_values) == 1:
#         best_model_run = ml_param_run
#     else:
#         best_model_run = run_regr(
#             df_in=df_in,
#             ml_model=best_model,
#             ml_features=ml_features,
#             ml_target=ml_target,
#             cv_type=cv_type,
#             n_splits=n_splits,
#         )

#     # Create energy plots
#     # This could now also be done outside. Return just best model and plot the result of best model fit.
#     # However, for convenience reasons do it here.
#     # TODO: Generalize this to any column name that should be plotted...

#     ener_fig = plot_energies(
#         result_dict=best_model_run["result_dict"],
#         error_dict=best_model_run["error_dict"],
#     )

#     return_dict["ener_fig"] = ener_fig

#     error_headers = [error_key.replace("_", "s_") for error_key in error_dict.keys()]
#     errors_dict = {
#         key: value for key, value in zip(error_headers, errors_array.transpose())
#     }

#     error_fig = plot_errors(
#         error_dict=errors_dict,
#         x_values=ml_param_values,
#         plot_measures=[
#             "rmses_trains",
#             "rmses_tests",
#             "rsquareds_trains",
#             "rsquareds_tests",
#             "maes_trains",
#             "maes_tests",
#         ],
#         annot_text=[""] * len(ml_param_values),
#         x_title=list(ml_param_dict.keys())[0],
#     )

#     return_dict["error_fig"] = error_fig
#     return_dict["errors_dict"] = errors_dict

#     # return_dict['numfeat_fig'] = numfeat_fig

#     return return_dict


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

#     rsquareds, rmses, maes, ener_figs = [], [], [], []
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
#         ener_figs.append(ener_fig)

#     return {
#         "ener_figs": ener_figs,
#         "rsquareds": rsquareds,
#         "rmses": rmses,
#         "maes": maes,
#     }
