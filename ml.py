from .data import *
from .misc import apply_feat_mask
from .vis import plot_errors, plot_energies
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
import pandas as pd
import copy
from tqdm import trange
from plotly.subplots import make_subplots
import plotly.graph_objs as go




def create_split(X, y, n_splits=5, split_select=0):

    split_count = 0

    kf_split = KFold(n_splits=n_splits, shuffle=False).split(X, y)

    for train_index, test_index in kf_split:
        if split_count == split_select:
            break
        split_count += 1

    return X[train_index], X[test_index], y[train_index], y[test_index]


def run_regr(df_in, ml_model, ml_features, ml_target, cv_type='kfold', n_splits=5):

    # TODO: Remove and/or rename, as linear model not applicaple anymore. Generally, estimators that require data standardization.

    ml_models = []
    X_trains, X_tests, X_fulls = [], [], []
    y_trains, y_tests, y_fulls, y_train_preds, y_test_preds, y_full_preds = [], [], [], [], [], []
    m_trains, m_tests, m_fulls, l_trains, l_tests, l_fulls = [], [], [], [], [], [],
    rmse_trains, rmse_tests, rmse_fulls = [], [], []
    mae_trains, mae_tests, mae_fulls = [], [], []
    rsquared_trains, rsquared_tests, rsquared_fulls = [], [], []


    # if isinstance(ml_model, Pipeline): # -> Maybe it'll work the same way, by just feeding a pipeline...
    #     print("Your ML model is a pipeline. I expect data standardization to be in the pipeline, if it is needed.")

    # Don't return DF here, as depending on KFold-split settings and LOOCV a different number of data distributions
    # will get evaluated.

    if cv_type == 'kfold':

        # TODO: Verify why normal selection with square brackets doesn't work.
        # TODO: Also, this is not good code, as distinction between list and np.array is very shady to distinguish between a list of feature names, and actual values passed.
        X = df_in[ml_features].values
        if len(ml_target) == 1:
            y = df_in[ml_target].values.ravel()
        elif len(ml_target) > 1:
            # TODO: This is not relevant if I'm not doing multivariate regression
            y = df_in[ml_target].values
        else:
            print(ml_features, type(ml_features))
            print(ml_target)
            raise TypeError

        for n_split in range(n_splits):

            X_train, X_test, y_train, y_test = create_split(X=X, y=y, n_splits=n_splits, split_select=n_split)

            # TODO: Generalize references to 'metal' columns.
            m_train, m_test, l_train, l_test = create_split(X=np.array(df_in['metal'].to_list()),
                                                            y=np.array(df_in['plot_label'].to_list()),
                                                            n_splits=n_splits, split_select=n_split)

            # Need to standardize data for linear models. Not necessary for RF.
            # TODO: Implement usage of a pipeline without the code breaking, for example for PCR. Scaler in pipeline? I'd say so, but haw can we retrieve the scaled data???
            # TODO: Code fragment duplicated once for k-fold, once for loocv -> This issue will be resolved with the general package, when CV is evaluated lazily by adding a column to the df.
            if isinstance(ml_model, NEED_TO_STANDARDIZE) is True:
                # This code trains and scales. Note, I don't rename it, so that I can still use the same variables, irrespective of if it is scaled or not.
                train_scaler = StandardScaler().fit(X_train)
                X_train = train_scaler.transform(X_train)
                X_test = train_scaler.transform(X_test)

            # Fit and predict
            _ = ml_model.fit(X_train, y_train)
            y_train_pred, y_test_pred = ml_model.predict(X_train), ml_model.predict(X_test)

            # Re-concatenation instead of taking initial y again for the case of shuffling in train-test-split.
            X_full = np.concatenate([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            y_full_pred = np.concatenate([y_train_pred, y_test_pred])
            m_full = np.concatenate([m_train, m_test])
            l_full = np.concatenate([l_train, l_test])

            rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
            rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
            rmse_full = mean_squared_error(y_full, y_full_pred, squared=False)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mae_full = mean_absolute_error(y_full, y_full_pred)
            rsquared_train = r2_score(y_train, y_train_pred)
            rsquared_test = r2_score(y_test, y_test_pred)
            rsquared_full = r2_score(y_full, y_full_pred)

            ml_models.append(ml_model)
            X_trains.append(X_train)
            X_tests.append(X_test)
            X_fulls.append(X_full)
            y_trains.append(y_train)
            y_tests.append(y_test)
            y_fulls.append(y_full)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)
            y_full_preds.append(y_full_pred)
            m_trains.append(m_train)
            m_tests.append(m_test)
            m_fulls.append(m_full)
            l_trains.append(l_train)
            l_tests.append(l_test)
            l_fulls.append(l_full)
            rmse_trains.append(rmse_train)
            rmse_tests.append(rmse_test)
            rmse_fulls.append(rmse_full)
            mae_trains.append(mae_train)
            mae_tests.append(mae_test)
            mae_fulls.append(mae_full)
            rsquared_trains.append(rsquared_train)
            rsquared_tests.append(rsquared_test)
            rsquared_fulls.append(rsquared_full)

    # Generalize loocv to column name and unique values here.
    else:

        # Do similar as above, but not over KFold-split and different data sets, but different metals/or sthg general.
        # Here, I need to do the splitting with the df and not with X, and y, bc I select the data points based on the
        # "metal" column. However, could also first evaluate the indices along this column and then use that...

        for group_val in df_in[cv_type].unique():

            nom_df = df_in.loc[df_in[cv_type].isin([_ for _ in df_in[cv_type].unique() if _ != group_val])]
            m_df = df_in.loc[df_in[cv_type] == group_val]

            X_train, X_test = nom_df[ml_features].to_numpy(), m_df[ml_features].to_numpy()
            # y_train , y_test = nom_df[ml_target].to_numpy(), m_df[ml_target]

            if len(ml_target) == 1:
                y_train, y_test = nom_df[ml_target].values.ravel(), m_df[ml_target].values.ravel()
            elif len(ml_target) > 1:
                y_train, y_test = nom_df[ml_target].values, m_df[ml_target].values
            else:
                print(ml_features, type(ml_features))
                print(ml_target)
                raise TypeError

            X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
            y_train, y_test = y_train.astype('float32'), y_test.astype('float32')

            m_train, m_test = nom_df[cv_type].to_list(), m_df[cv_type].to_list()
            l_train, l_test = nom_df['plot_label'].to_list(), m_df['plot_label'].to_list()

            # Need to standardize data for linear models. Not necessary for RF.
            if isinstance(ml_model, NEED_TO_STANDARDIZE) is True:
                # This code trains and scales. Note, I don't rename it, so that I can still use the same variables, irrespective of if it is scaled or not.
                train_scaler = StandardScaler().fit(X_train)
                X_train = train_scaler.transform(X_train)
                X_test = train_scaler.transform(X_test)

            # Fit and predict
            _ = ml_model.fit(X_train, y_train)
            y_train_pred, y_test_pred = ml_model.predict(X_train), ml_model.predict(X_test)

            # Re-concatenation instead of taking initial y again for the case of shuffling in train-test-split.
            X_full = np.concatenate([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            y_full_pred = np.concatenate([y_train_pred, y_test_pred])
            m_full = np.concatenate([m_train, m_test])
            l_full = np.concatenate([l_train, l_test])

            rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
            rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
            rmse_full = mean_squared_error(y_full, y_full_pred, squared=False)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mae_full = mean_absolute_error(y_full, y_full_pred)
            rsquared_train = r2_score(y_train, y_train_pred)
            rsquared_test = r2_score(y_test, y_test_pred)
            rsquared_full = r2_score(y_full, y_full_pred)

            ml_models.append(ml_model)
            X_trains.append(X_train)
            X_tests.append(X_test)
            X_fulls.append(X_full)
            y_trains.append(y_train)
            y_tests.append(y_test)
            y_fulls.append(y_full)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)
            y_full_preds.append(y_full_pred)
            m_trains.append(m_train)
            m_tests.append(m_test)
            m_fulls.append(m_full)
            l_trains.append(l_train)
            l_tests.append(l_test)
            l_fulls.append(l_full)
            rmse_trains.append(rmse_train)
            rmse_tests.append(rmse_test)
            rmse_fulls.append(rmse_full)
            mae_trains.append(mae_train)
            mae_tests.append(mae_test)
            mae_fulls.append(mae_full)
            rsquared_trains.append(rsquared_train)
            rsquared_tests.append(rsquared_test)
            rsquared_fulls.append(rsquared_full)

    # Here done with the cross-validation, just sort all data in the return-dictionary.
    result_dict = {
        'X_trains': X_trains,
        'X_tests': X_tests,
        'X_fulls': X_fulls,
        'y_trains': y_trains,
        'y_tests': y_tests,
        'y_fulls': y_fulls,
        'y_train_preds': y_train_preds,
        'y_test_preds': y_test_preds,
        'y_full_preds': y_full_preds,
        'm_trains': m_trains,
        'm_tests': m_tests,
        'm_fulls': m_fulls,
        'l_trains': l_trains,
        'l_tests': l_tests,
        'l_fulls': l_fulls,
    }

    error_dict = {
        'rmse_trains': rmse_trains,
        'rmse_tests': rmse_tests,
        'rmse_fulls': rmse_fulls,
        'mae_trains': mae_trains,
        'mae_tests': mae_tests,
        'mae_fulls': mae_fulls,
        'rsquared_trains': rsquared_trains,
        'rsquared_tests': rsquared_tests,
        'rsquared_fulls': rsquared_fulls,
    }
    best_id = np.argmin(rmse_tests)

    return {'ml_models': ml_models, 'best_id': best_id, 'result_dict': result_dict, 'error_dict': error_dict}


# @timecall
def vary_ml_param(df_in, ml_base_model, ml_features, ml_target, ml_param_dict, cv_type='kfold', n_splits=5, verbose=False,):

    # Missing: Descriptor figure and number of features list/figure.
    return_dict = {}

    # coefs, numfeatures, models, rmses, y_preds, rsquareds = [], [], [], [], [], []

    counter = 0
    ml_models, test_rmses, feature_coefs = [], [], []
    param_values = list(ml_param_dict.values())[0]
    # Lazy style, using try-except instead of proper programming. Could do also with isinstance. However, as convention,
    # always pass param_values as list, with length one for single value.
    # try:
    errors_array = np.zeros(shape=(len(param_values), 9))
    # except TypeError:
    #     errors_array = np.zeros(shape=(1, 6))

    # If verbosity turned on, print timing information with trange.
    if verbose is True:
        ml_param_range = trange(param_values)
    else:
        ml_param_range = param_values

    for ml_param_value in ml_param_range:

        # print("Param: {:.6f}".format(ml_param_value))

        ml_mod_model = copy.deepcopy(ml_base_model.set_params(**{list(ml_param_dict.keys())[0]: ml_param_value}))
        # print("ml_mod_model: {}".format(ml_mod_model))

        ml_param_run = run_regr(df_in=df_in, ml_model=ml_mod_model, ml_features=ml_features, ml_target=ml_target,
                                cv_type=cv_type, n_splits=n_splits)

        ml_param_model = ml_param_run['ml_models'][ml_param_run['best_id']]
        error_dict = ml_param_run['error_dict']
        result_dict = ml_param_run['result_dict']

        ml_models.append(ml_param_model)

        # If before full CV done, average the errors.
        for error_dict_key in list(error_dict.keys()):
            error_dict[error_dict_key] = [np.mean(error_dict[error_dict_key])]

        test_rmses.append(error_dict['rmse_tests'][0])

        errors_array[counter, :] = [error_dict_value[0] for error_dict_value in list(error_dict.values())]

        counter += 1

    best_id = np.argmin(test_rmses)

    return_dict['ml_models'] = ml_models
    return_dict['best_id'] = best_id
    return_dict['best_param'] = ml_param_range[best_id]
    return_dict['test_rmses'] = test_rmses

    best_model = ml_models[best_id]

    if len(ml_param_range) == 1:
        best_model_run = ml_param_run
    else:
        best_model_run = run_regr(df_in=df_in, ml_model=best_model, ml_features=ml_features, ml_target=ml_target,
                                  cv_type=cv_type, n_splits=n_splits)

    # Create energy plots
    # This could now also be done outside. Return just best model and plot the result of best model fit.
    # However, for convenience reasons do it here.
    # TODO: Generalize this to any column name that should be plotted...

    ener_fig = plot_energies(result_dict=best_model_run['result_dict'],
                                    error_dict=best_model_run['error_dict'],
                                    )

    return_dict['ener_fig'] = ener_fig

    error_headers = [error_key.replace('_', 's_') for error_key in error_dict.keys()]
    errors_dict = {key: value for key, value in zip(error_headers, errors_array.transpose())}

    error_fig = plot_errors(error_dict=errors_dict, x_values=param_values,
                            plot_measures=['rmses_trains', 'rmses_tests', 'rsquareds_trains', 'rsquareds_tests', 'maes_trains', 'maes_tests'],
                            annot_text=['']*len(param_values), x_title=list(ml_param_dict.keys())[0])

    return_dict['error_fig'] = error_fig
    return_dict['errors_dict'] = errors_dict

    # return_dict['numfeat_fig'] = numfeat_fig

    return return_dict


# @timecall
def run_en_scan(en_df, en_features, en_target, en_alphas, en_l1_ratios, cv_type='kfold', n_splits=5):

    # If csv exists, read i, otherwise do scan.
    # l1_ratios lower than 0.1 significantly increase calculation time. Idk why. Try out pure ridge run for timing test.

    en_numfeat_array = np.zeros(shape=(len(en_l1_ratios), len(en_alphas)))
    en_rmse_array = np.zeros(shape=(len(en_l1_ratios), len(en_alphas)))

    for ien_alpha, en_alpha in enumerate(en_alphas):
        for ien_l1_ratio, en_l1_ratio in enumerate(en_l1_ratios):

            en_run = run_regr(df_in=en_df, ml_features=en_features, ml_target=en_target,
                            ml_model=ElasticNet(
                                alpha=en_alpha, l1_ratio=en_l1_ratio,
                                max_iter=int(float('1e5')), tol=float('1e-3'),
                                random_state=0,
                            ),
                            cv_type=cv_type, n_splits=N_SPLITS,
                            )

            selected_features = apply_feat_mask(model=en_run['ml_models'][np.argmin(en_run['error_dict']['rmse_tests'])],
                                                model_features=en_features, feat_thresh=0)

            en_cv_best_rmse = np.min(en_run['error_dict']['rmse_tests'])

            en_rmse_array[ien_l1_ratio, ien_alpha] = en_cv_best_rmse

            en_numfeat_array[ien_l1_ratio, ien_alpha] = len(selected_features)

    numfeat_df = pd.DataFrame(data=en_numfeat_array).apply(pd.to_numeric, downcast='integer')
    rmse_df = pd.DataFrame(data=en_rmse_array)

    return {'en_alphas': en_alphas, 'en_l1_ratios': en_l1_ratios, 'numfeat_df': numfeat_df, 'rmse_df': rmse_df}


def loocv_all_points(df_in, ml_model, ml_features, ml_target, column, plot_range):
    # TODO: Could adapt the other function such that it does the same. Keep somewhat duplicated code here.

    rsquareds, rmses, maes, ener_figs = [], [], [], []
    column_unique_vals = df_in[column].unique()

    for column_value in column_unique_vals:

        train_df = df_in.loc[df_in[column].isin([_ for _ in column_unique_vals if _ != column_value])]
        test_df = df_in.loc[df_in[column] == column_value]

        X_train, y_train = train_df[ml_features].to_numpy(), train_df[ml_target].values.ravel()
        X_test, y_test = test_df[ml_features].to_numpy(), test_df[ml_target].values.ravel()

        if isinstance(ml_model, NEED_TO_STANDARDIZE) is True:
            train_scaler = StandardScaler().fit(X_train)
            X_train = train_scaler.transform(X_train)
            X_test = train_scaler.transform(X_test)

        _ = ml_model.fit(X_train, y_train)

        y_pred = ml_model.predict(X_test)

        # All errors again only on testing data
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        rsquared = r2_score(y_test, y_pred)

        rsquareds.append(rsquared)
        rmses.append(rmse)
        maes.append(mae)

        # print("column_value: {} | RMSE: {:.3f} | MAE: {:.3f}".format(column_value, rmse, mae))

        # Plot energies
        ener_fig = go.Figure()

        # Plot energy data points
        _ = ener_fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                text=test_df['plot_label'].tolist(),
                mode='markers', marker=dict(size=8, symbol=0, color=color_dict.get(column_value, 'blue'), opacity=1),
                hoverinfo='x+y+text', showlegend=True, name=column_value.title()
            ),
        )

        # Add ideal fit line to plot
        _ = ener_fig.add_trace(
            go.Scatter(
                x=plot_range,
                y=plot_range,
                mode='lines', line=dict(color='rgb(0, 0, 0, 0.1)', width=2, dash='dash'), hoverinfo='skip', showlegend=False,
            ),
        )

        _ = ener_fig.add_annotation(
            xanchor='left', yanchor='top',
            xref='paper', yref='paper',
            x=0, y=1,
            align="left",
            text="R<sup>2</sup> = {:.3f}<br>RMSE = {:.3f}<br>MAE = {:.3f}".format(rsquared, rmse, mae),
            font_size=26, font_family="Arial", showarrow=False,
            bgcolor='rgba(0,0,0,0.1)'
        )

        _ = ener_fig.update_layout(energy_layout)
        range_layout = go.Layout(xaxis_range=plot_range, yaxis_range=plot_range)
        _ = ener_fig.update_layout(range_layout)
        ener_figs.append(ener_fig)

    return {'ener_figs': ener_figs, 'rsquareds': rsquareds, 'rmses': rmses, 'maes': maes}