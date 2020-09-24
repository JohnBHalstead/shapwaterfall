# shapwaterfall
# version 0.0.4 (September 24, 2020)
# Principle author: John Halstead, jhalstead@vmware.com

def ShapWaterFall(Model, X_tng, X_sc, ref1, ref2, num_feature):
    import pandas as pd
    import numpy as np
    import shap
    import matplotlib.pyplot as plt
    import waterfall_chart

    # label names until we figure out how sql alchemy can fully work on Linux
    clients_to_show = [ref1, ref2]

    # Data Frame management
    if isinstance(X_sc, pd.DataFrame):
        X_v = X_sc
    else:
        X_v = pd.DataFrame(X_sc)
    if isinstance(X_tng, pd.DataFrame):
        X_t = X_tng
    else:
        X_t = pd.DataFrame(X_tng)

    # SHAP Values
    explainer = shap.TreeExplainer(Model, shap.sample(X_t, 100))

    # Data
    data_for_prediction1 = X_v[(X_v.Reference == clients_to_show[0])]
    data_for_prediction1 = data_for_prediction1.drop('Reference', 1)
    data_for_prediction2 = X_v[(X_v.Reference == clients_to_show[1])]
    data_for_prediction2 = data_for_prediction2.drop('Reference', 1)

    # Insert a binary option to ensure order goes from lower to higher propensity
    if Model.predict_proba(data_for_prediction1)[:, 1] <= Model.predict_proba(data_for_prediction2)[:, 1]:
        frames = [data_for_prediction1, data_for_prediction2]
    else:
        frames = [data_for_prediction2, data_for_prediction1]
        clients_to_show = [ref2, ref1]

    # Computations for Waterfall Chart
    data_for_prediction = pd.concat(frames)
    data_for_prediction = pd.DataFrame(data_for_prediction)
    feature_names = data_for_prediction.columns.values
    shap_values = explainer.shap_values(data_for_prediction)
    Feat_contrib = pd.DataFrame(list(map(np.ravel, shap_values[1])), columns=feature_names)
    counter1 = len(Feat_contrib.columns)
    Feat_contrib['base_line_diff'] = Feat_contrib.sum(axis=1)
    Feat_contrib['prediction'] = Model.predict_proba(data_for_prediction)[:, 1]
    Feat_contrib['baseline'] = Feat_contrib.prediction - Feat_contrib.base_line_diff
    diff_df = pd.DataFrame(
        {'features': Feat_contrib.diff().iloc[1, :].index, 'contrib': Feat_contrib.diff().iloc[1, :].values})[
              :counter1].sort_values(by='contrib', ascending=False).reset_index(drop=True)

    # Waterfall Chart
    plt.rcParams.update({'figure.figsize': (16, 12), 'figure.dpi': 100})
    xlist = [[clients_to_show[0], 'Other {a} Features'.format(a=counter1-num_feature)], diff_df.features.tolist()[:num_feature]]
    xlist = [item for sublist in xlist for item in sublist]
    ylist = [[np.round(Feat_contrib.prediction[0], 6), np.round(diff_df.contrib[num_feature:].sum(), 6)],
             np.round(diff_df.contrib.tolist(), 6)[:num_feature]]
    ylist = [item for sublist in ylist for item in sublist]
    waterfall_df = pd.DataFrame({"x_values": xlist, 'y_values': ylist})
    plt.rcParams.update({'figure.figsize': (16, 12), 'figure.dpi': 100})
    plot = waterfall_chart.plot(xlist, ylist, net_label=str(clients_to_show[1]), rotation_value=90, formatting='{:,.3f}')
    plot.show()
