import pandas as pd
import pickle


def main():

    X = pd.read_pickle('dataframes/X.pkl')
    y = pd.read_pickle('dataframes/y.pkl')
    LGBM_model = pickle.load(f)

    partition_explainer = shap.PartitionExplainer(LGBM_model, X)

    shap.bar_plot(partition_explainer.shap_values(X_filled_undersampled[0]),
                  feature_names=df.columns,
                  max_display=12)

if __name__ == "__main__":
    main()