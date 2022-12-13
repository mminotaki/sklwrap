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

# def prim_feat_hist(dict_in):
#     hist_fig = go.Figure()
#     hist_layout = go.Layout(
#         width=1000,
#         height=600,
#         font=dict(size=20, family="Arial", color="black"),
#         margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#         ),
#         xaxis=dict(
#             title="Primary Feature",
#             title_font_size=20,
#             showline=True,
#             linewidth=3,
#             linecolor="black",
#             mirror=True,
#             showgrid=False,
#             zeroline=False,
#             gridcolor="rgba(0,0,0,0.3)",
#             ticks="outside",
#             tickfont_size=12,
#             tickwidth=3,
#             ticklen=6,
#             categoryorder="mean descending",
#         ),
#         yaxis=dict(
#             title="Coeff",
#             title_font_size=20,
#             showline=True,
#             linewidth=3,
#             linecolor="black",
#             mirror=True,
#             showgrid=False,
#             zeroline=False,
#             gridcolor="rgba(0,0,0,0.3)",
#             ticks="outside",
#             tickfont_size=18,
#             tickwidth=3,
#             ticklen=6,
#         ),
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#         barmode="stack",
#         legend=dict(
#             yanchor="top", y=1, xanchor="right", x=1, bgcolor="rgba(0,0,0,0.1)"
#         ),
#     )

#     _ = hist_fig.add_trace(
#         go.Bar(
#             name="test",
#             x=list(dict_in.keys()),
#             y=list(dict_in.values()),
#             text=["{:.3f}".format(val) for val in list(dict_in.values())],
#             textposition="inside",
#             textfont=dict(size=20, family="Arial", color="black"),
#         ),
#     )

#     _ = hist_fig.update_layout(hist_layout)
#     return hist_fig


# def corr_analysis(df_in, features, thresh=0.95, verbose=False, create_image=True):
#     # Visualize variable correlations

#     corr_df = df_in[features].corr().abs()

#     if thresh is None:
#         thresh_corr_df = corr_df[(corr_df > 1 - float("1e-9"))]
#     else:
#         thresh_corr_df = corr_df[(corr_df > thresh) & (corr_df != 1)]

#     # plot_corr_df = thresh_corr_df.where(np.tril(np.ones(thresh_corr_df.shape)).astype(bool))

#     return_dict = {"corr_df": thresh_corr_df}

#     if create_image is False:
#         return return_dict

#     # Define hovertext for x unified hovermode that for any descriptor all correlations are shown.
#     if len(features) < 200:
#         hovertext = []
#         for illist, llist in enumerate(thresh_corr_df.values.tolist()):
#             temp = ""
#             for ielement, element in enumerate(llist):
#                 if not math.isnan(element):
#                     temp += "{}: {}<br>".format(features[ielement], round(element, 3))
#             hovertext.append([temp[:-4]] * ielement)
#         hovertext = np.array(hovertext)  # .transpose()

#     # Plot correlation heatmap
#     corr_fig = go.Figure()

#     _ = corr_fig.add_trace(
#         go.Heatmap(
#             x=corr_df.columns,
#             y=corr_df.columns,
#             z=thresh_corr_df.values,
#             text=np.around(thresh_corr_df.values, decimals=2),
#             hoverinfo="x+y+z",
#             xgap=0.2,
#             ygap=0.2,
#             colorbar=dict(thickness=20, ticklen=3, y=0.501, len=1.026),
#         )
#     )

#     full_corr_layout = go.Layout(
#         width=1000,
#         height=1000,
#         font=dict(size=16, family="Arial", color="black"),
#         margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#         ),
#         # title=dict(text='Correlation > {}'.format(thresh), x=0.5),
#         # hovermode='x unified',
#         xaxis=dict(
#             showgrid=False,
#             autorange="reversed",
#             tickfont_size=14,
#             linecolor="black",
#             showline=True,
#             mirror=True,
#         ),
#         yaxis=dict(
#             showgrid=False,
#             autorange="reversed",
#             tickfont_size=14,
#             linecolor="black",
#             showline=True,
#             mirror=True,
#         ),
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#     )

#     _ = corr_fig.update_layout(full_corr_layout)
#     return_dict["corr_fig"] = corr_fig

#     corr_pairs, corr_set = [], []

#     # Print out correlated features for each feature if verbosity is set.
#     if verbose is True:
#         print("Correlated descriptors:\n")
#         for desc_comb in it.combinations(corr_df.columns, 2):
#             corr = corr_df[desc_comb[0]][desc_comb[1]]
#             if corr > thresh:
#                 corr_pairs.append(sorted([*desc_comb]))
#                 corr_set += [*desc_comb]

#         corr_set = set(corr_set)

#         corr_dict = {corr_desc: [] for corr_desc in corr_set}

#         for corr_pair in corr_pairs:
#             corr_dict[corr_pair[0]].append(corr_pair[1])

#         for corr_key, corr_vals in corr_dict.items():
#             if len(corr_vals) > 0:
#                 print("{} ({}): {}".format(corr_key, len(corr_vals), corr_vals))
#         print("=" * 127)

#     return return_dict


# def plot_en_scan(result_dict, numfeat_thresh=100, rmse_thresh=0.3):
#     en_alphas = result_dict["en_alphas"]
#     en_l1_ratios = result_dict["en_l1_ratios"]
#     numfeat_df = result_dict["numfeat_df"]
#     rmse_df = result_dict["rmse_df"]

#     heatmaps = make_subplots(
#         rows=1,
#         cols=2,
#         horizontal_spacing=0.2,
#     )

#     _ = heatmaps.add_trace(
#         go.Heatmap(
#             x=en_alphas,
#             y=en_l1_ratios,
#             z=numfeat_df[(rmse_df <= rmse_thresh) & (numfeat_df <= numfeat_thresh)],
#             # z=en_heatmap_array,
#             hoverinfo="x+y+z+text",
#             hovertext=np.round(rmse_df.values, 3),
#             zhoverformat="d",
#             xgap=0,
#             ygap=0,
#             colorbar=dict(
#                 x=0.4,
#                 thickness=20,
#                 ticklen=3,
#                 tickfont=dict(size=18, color="black", family="Arial"),
#             ),
#         ),
#         row=1,
#         col=1,
#     )

#     _ = heatmaps.add_trace(
#         go.Heatmap(
#             x=en_alphas,
#             y=en_l1_ratios,
#             z=rmse_df[(rmse_df <= rmse_thresh) & (numfeat_df <= numfeat_thresh)],
#             # z=en_rmse_array, #[rmse_df < .5],
#             hoverinfo="x+y+z+text",
#             zhoverformat=".3f",
#             hovertext=numfeat_df.values,
#             xgap=0,
#             ygap=0,  # vertical_spacing=0.08, horizontal_spacing=0.08,
#             colorbar=dict(
#                 x=1,
#                 thickness=20,
#                 ticklen=3,
#                 tickfont=dict(size=18, color="black", family="Arial"),
#             ),
#         ),
#         row=1,
#         col=2,
#     )

#     xaxis_dict = dict(
#         title="&#x3b1;",
#         title_font_size=20,
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         gridcolor="rgba(0,0,0,0.3)",
#         ticks="outside",
#         tickfont_size=18,
#         hoverformat=".3f",
#         tickwidth=3,
#         ticklen=6,
#     )
#     yaxis_dict = dict(
#         title=dict(
#             text="l1-ratio",
#             font_size=20,
#             standoff=10,
#         ),
#         showline=True,
#         linewidth=3,
#         linecolor="black",
#         mirror=True,
#         showgrid=False,
#         zeroline=False,
#         gridcolor="rgba(0,0,0,0.3)",
#         ticks="outside",
#         tickfont_size=18,
#         hoverformat=".3f",
#         tickwidth=3,
#         ticklen=6,
#     )
#     en_heatmap_layout = go.Layout(
#         width=1200,
#         height=500,
#         font=dict(size=18, family="Arial", color="black"),
#         margin=dict(
#             l=0,
#             r=0,
#             b=0,
#             t=0,
#         ),
#         xaxis=xaxis_dict,
#         xaxis2=xaxis_dict,
#         yaxis=yaxis_dict,
#         yaxis2=yaxis_dict,
#         paper_bgcolor="white",
#         plot_bgcolor="white",
#     )

#     _ = heatmaps.update_layout(en_heatmap_layout)

#     return heatmaps