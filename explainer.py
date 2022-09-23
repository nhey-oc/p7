import pandas as pd
import pickle

import shap
from lime import lime_tabular


def main():

    X = pd.read_pickle('dataframes/X_filled_random_undersampled.pkl')
    y = pd.read_pickle('dataframes/y_filled_random_undersampled.pkl')

    with open('models/LGBM_model.pkl', 'rb') as f:
        LGBM_model = pickle.load(f)

    #explainer = lime_tabular.LimeTabularExplainer(X_train, mode="regression", feature_names= boston.feature_names)


    explainer = shap.KernelExplainer(LGBM_model.predict, shap.sample(X, 2500))
    shap_values = explainer.shap_values(X.iloc[0])

    shap.waterfall_plot(explainer.expected_value,shap_values,X.iloc[0])


    partition_explainer = shap.PartitionExplainer(LGBM_model, X)

    print(partition_explainer)
    shap.bar_plot(partition_explainer.shap_values(X_filled_undersampled[0]),
                  feature_names=df.columns,
                  max_display=12)

if __name__ == "__main__":
    main()