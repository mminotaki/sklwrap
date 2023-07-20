import copy
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm

from .vis import plot_errors, plot_regr

NEED_TO_STANDARDIZE = (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    SVR,
)


def add_cv_columns(
    df_in,
    cv_setup={"cv_type": "kfold", "cv_spec": 5},
    overwrite=False,
    retain_setup=None,
):
    """Add train and test indices as boolean columnns with train_x headers to dataframe.

    Args:
        df_in (pd.DataFrame): DataFrame to which the data splitting should be applied.
        cv_setup (dict, optional): Cross-validation setup. Either kfold or logocv. Defaults to {"cv_type": "kfold", "cv_spec": 5}.
        overwrite (bool, optional): If train columns that already exist in the DataFrame should be overwritten. Defaults to False.
        retain_setup (dict, optional): If n highest/lowest data points of one column should be retained in the training data. Setup for example: {"column": target, "ascending": False, "retain_where": train, "number": 2}. Defaults to None.

    Raises:
        NotImplementedError: If cv_setup not implemented.

    Returns:
        pd.DataFrame: DataFrame with train_x columns added.

    TODO:
        Replace number in retain setup with slice?
    """
    df_func = df_in.copy(deep=True)

    if retain_setup is not None:
        df_func = df_func.sort_values(
            by=retain_setup["column"], ascending=retain_setup["ascending"]
        )
        df_retain = df_func.head(retain_setup["number"])
        df_func = df_func.iloc[retain_setup["number"] :]
        df_func = df_func.sample(frac=1, random_state=42)

    num_rows = df_func.shape[0]

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

        split_array = np.logical_not(split_array)

    elif cv_setup["cv_type"].lower() == "custom":
        split_column_number = 2
        custom_column = list(cv_setup["cv_spec"].keys())[0]
        custom_test = list(cv_setup["cv_spec"].values())[0]
        print(custom_test)
        boolean_list = df_func[custom_column].isin(custom_test).values
        split_array = np.array([np.logical_not(boolean_list), boolean_list]).transpose()

    elif cv_setup["cv_type"].lower() == "loocv":
        raise NotImplementedError("LOOCV not implemented yet.")

    split_column_names = ["train_{:05d}".format(i) for i in range(split_column_number)]

    # ! Currently overwriting not yet implemented.
    if set(split_column_names) == set([_ for _ in df_func.columns]):
        print(
            "Same number of training columns already in dataframe. Not changing anything."
        )
    else:
        if split_column_names[0] in df_func.columns:
            print(
                "Some training columns already in dataframe -> Overwrite: {}.".format(
                    overwrite
                )
            )
            if overwrite is True:
                df_func = df_func[
                    df_func.columns.drop(list(df_func.filter(regex="train_")))
                ]
            else:
                return df_func

        df_bool = pd.DataFrame(
            data=split_array, columns=split_column_names, index=df_func.index
        )
        df_func = pd.concat([df_func, df_bool], axis=1)

    if retain_setup is not None:
        if retain_setup["retain_where"].lower() == "train":
            fill_value = True
        elif retain_setup["retain_where"].lower() == "test":
            fill_value = False
        retain_bool_array = np.full(
            shape=(retain_setup["number"], split_column_number), fill_value=fill_value
        )
        df_bool_retain = pd.DataFrame(
            data=retain_bool_array, columns=split_column_names, index=df_retain.index
        )
        df_retain = pd.concat([df_retain, df_bool_retain], axis=1)

        df_func = pd.concat([df_func, df_retain], ignore_index=True)

    return df_func


def run_regr(
    df_in: pd.DataFrame,
    ml_model,
    ml_features,
    ml_target,
):
    """
    Runs a regression analysis using the specified machine learning model and features.

    This function performs a regression analysis on the provided DataFrame using the given machine learning model and features.
    It splits the data into training and testing sets based on the provided split column names and calculates various evaluation metrics.
    The function returns a dictionary containing the results and intermediate data.

    Parameters:
        df_in (pd.DataFrame): The input DataFrame.
        ml_model: The machine learning model to use for regression analysis.
        ml_features (list): The list of column names representing the features for regression.
        ml_target (str): The column name representing the target variable for regression.

    Returns:
        dict: A dictionary containing the following items:
            - 'df_in' (pd.DataFrame): The modified input DataFrame with additional prediction columns.
            - 'ml_models' (list): A list of trained machine learning models.
            - 'scalers' (list): A list of scalers used for feature standardization (or None if not needed).
            - 'error_dict' (dict): A dictionary containing evaluation metrics such as RMSE, MAE, and R-squared.
            - 'best_id' (int): The index of the best performing model based on RMSE on the testing set.
    """
    # ! Don't filter down df columns here, as they might be needed for plotting later.
    # df_func = df_in[ml_features + [ml_target] + train_column_names].copy(deep=True)
    df_func = df_in.copy(deep=True)
    y_full = df_func[ml_target].values

    # Initialize all the empty lists that hold all the data for the return dictionary.
    ml_models = []
    scalers = []

    rmse_trains, rmse_tests, rmse_fulls = [], [], []
    mae_trains, mae_tests, mae_fulls = [], [], []
    rsquared_trains, rsquared_tests, rsquared_fulls = [], [], []

    train_column_names = [col for col in df_func.columns if col.startswith("train_")]
    # test_column_names = [col.replace("train_", "test_") for col in train_column_names]
    # ? How can I actually use this???
    if len(train_column_names) == 1:
        no_cv = True
    else:
        no_cv = False

    # print(test_column_names)

    pred_column_names = [
        "pred_{:05d}".format(i) for i in range(len(train_column_names))
    ]
    y_pred_arrays = []

    for split_column_name, pred_column_name in list(
        zip(train_column_names, pred_column_names)
    ):
        # print(split_column_name, pred_column_name)

        split_column_bool = df_func[split_column_name].values
        # print('split_column_name', split_column_name)

        df_train = df_func[split_column_bool].copy(deep=True)
        # if df_train.shape[0] == df_func.shape[0]:
        if no_cv is True:
            df_test = df_train.copy(deep=True)
        else:
            df_test = df_func[np.logical_not(split_column_bool)].copy(deep=True)

        x_train = df_train[ml_features].values
        x_test = df_test[ml_features].values
        y_train = df_train[ml_target].values
        y_test = df_test[ml_target].values
        logging.debug(
            x_train.shape, x_test.shape, y_train.shape, y_test.shape, y_full.shape
        )

        # Standardization within data splitting to avoid data leakage from testing data.
        if isinstance(ml_model, NEED_TO_STANDARDIZE):
            train_scaler = StandardScaler().fit(x_train)
            x_train = train_scaler.transform(x_train)
            x_test = train_scaler.transform(x_test)
            scalers.append(train_scaler)
        else:
            scalers.append(None)

        # TODO: Maybe it'll work the same way, by just feeding a pipeline? Insert data standardization into pipeline???
        # if isinstance(ml_model, Pipeline):
        #     print("Your ML model is a pipeline. I expect data standardization to be in the pipeline, if it is needed.")

        # Fit and predict
        _ = ml_model.fit(x_train, y_train)
        y_train_pred, y_test_pred = ml_model.predict(x_train), ml_model.predict(x_test)

        # ! Using np.concatenate will always put the test values behind the train values -> Have to sort on index.
        # y_full_pred = np.concatenate([y_train_pred, y_test_pred])

        df_train[pred_column_name] = y_train_pred
        df_test[pred_column_name] = y_test_pred

        # print(list(df_train.columns))
        # print(list(df_test.columns))

        # ! I don't want an additional "index" column, but I want to retain the indices...
        # ! Don't sort on index, though, retain as is.
        if no_cv is False:
            df_pred_full = pd.concat([df_train, df_test]).sort_index()
        else:
            # df_pred_full = df_train.copy(deep=True)
            df_pred_full = df_test.copy(deep=True)

        # print(df_train.head())
        # print(df_test.head())

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
        # print("=" * 100)

    # Concatenate the sorted predictions into df_func.
    df_pred = pd.DataFrame(data=np.array(y_pred_arrays).T, columns=pred_column_names)
    df_func = pd.concat([df_func, df_pred], axis=1)

    df_func = df_func.rename(columns={ml_target: "y"})

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
        "ml_models": ml_models,
        "scalers": scalers,
        "error_dict": error_dict,
        "best_id": int(best_id),
    }


def vary_ml_param(
    df_in,
    ml_base_model,
    ml_features,
    ml_targets,
    ml_param_dict,
    color_setup=None,  # ! Copy-paste passing to plot_regr from outside. Must be improved.
    *args,
    **kwargs,
):
    """
    Varies a machine learning parameter and performs regression analysis for each parameter value.

    This function performs regression analysis using the specified machine learning model and features.
    It varies a machine learning parameter by iterating over the provided parameter values and runs regression analysis for each value.
    The function returns a dictionary containing the results and intermediate data.

    Parameters:
        df_in (pd.DataFrame): The input DataFrame.
        ml_base_model: The base machine learning model to use for regression analysis.
        ml_features (list): The list of column names representing the features for regression.
        ml_target (str): The column name representing the target variable for regression.
        ml_param_dict (dict): A dictionary containing the machine learning parameter name as the key and a list of parameter values as the value.
        color_setup (dict, optional): Color setup dictionary for plotting (default: None).
        *args: Additional positional arguments to pass to the 'run_regr' function.
        **kwargs: Additional keyword arguments to pass to the 'run_regr' function and plotting functions.

    Returns:
        dict: A dictionary containing the following items:
            - 'ml_models' (list): A list of trained machine learning models for each parameter value.
            - 'best_id' (int): The index of the best performing model based on RMSE on the testing set.
            - 'best_param': The best parameter value.
            - 'test_rmses' (list): A list of RMSE values on the testing set for each parameter value.
            - 'regr_runs' (list): A list of regression analysis results for each parameter value.
            - 'regr_figs' (dict): A dictionary containing regression analysis figures.
            - 'error_fig' (matplotlib.figure.Figure): The figure showing the errors for different parameter values.
            - 'error_dict' (dict): A dictionary containing the errors for different parameter values.
    """
    # Missing: Descriptor figure and number of features list/figure.
    return_dict = {}

    # coefs, numfeatures, models, rmses, y_preds, rsquareds = [], [], [], [], [], []

    counter = 0
    ml_models, test_rmses, regr_runs = [], [], []
    ml_param_values = list(ml_param_dict.values())[0]
    errors_array = np.zeros(shape=(len(ml_param_values), 9))

    # ! Hack to make it work with the same syntax now.
    if isinstance(ml_targets, list):
        ml_target = ml_targets[0]
    else:
        ml_target = ml_targets

    for ml_param_value in tqdm(ml_param_values):
        ml_mod_model = copy.deepcopy(
            ml_base_model.set_params(**{list(ml_param_dict.keys())[0]: ml_param_value})
        )

        ml_param_run = run_regr(
            df_in=df_in,
            ml_model=ml_mod_model,
            ml_features=ml_features,
            ml_target=ml_target,
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
        )

    # Create energy plots
    # This could now also be done outside. Return just best model and plot the result of best model fit.
    # However, for convenience reasons do it here.

    regr_figs = plot_regr(
        color_column="metal",  # ! Add color_column
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
