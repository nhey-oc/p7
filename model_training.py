# basical imports
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import os
import re

# Kaggle kernel import
from contextlib import contextmanager

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Metrics
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ParameterGrid
from collections import Counter
from numpy import where
from matplotlib import pyplot

# Pretreatment
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import imblearn
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import ClusterCentroids

# Models imports
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.metrics import fbeta_score, make_scorer


@contextmanager
def timer(title):
    t0 = time.time()
    print("---------------------------\n", title)
    yield
    print(" - done in {:.0f}s".format(time.time() - t0))
    print("End of", title)
    print("---------------------------")


def SVC_model(C, X_train, y_train, X_test, y_test):
    # C: SVC hyper parameter to optimize for.
    model = SVC(C = C)
    model.fit(X_train, y_train)
    y_score = model.decision_function(X_test)
    f = roc_auc_score(y_test, y_score)
    return f


def fillna_fun(df, target_name='TARGET', threshold=10):
    """
    fill the dataframe,
    don't  touch to the TARGET column,
    remove col with more than 1/threshold NaN values.
    """
    df_v2 = df.copy()
    target = df_v2[target_name]
    df_v2 = df_v2.drop(target_name, axis=1)

    print('Start with ', (df_v2.isna().sum() > 1 ).sum(), ' columns with NaN and ', df_v2.isna().sum().sum(), ' cells.' , sep="")
    for col in df_v2:
        if (df_v2[col].isna().sum() > (len(df_v2)/ threshold) ):
            df_v2 = df_v2.drop(col, axis=1)
    print('Now there is ', (df_v2.isna().sum() > 1 ).sum(), ' columns with NaN and ', df_v2.isna().sum().sum(), ' cells.', sep="")

    #Imputing with default parameters
    imputer = KNNImputer()

    #Reshaping to meet the dimensional requirement
    imputer.fit(df_v2)

    # KNNImputer fited is saved to reused the same in prod.
    with open('models/KNN_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

    df_filled =  pd.DataFrame(imputer.transform(df_v2), columns=df_v2.columns)

    return pd.concat([df_filled, target], axis=1)


def extract_x_y(df):
    # Divide in training/test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    # Set X and y
    X = train_df.drop('TARGET', axis=1)
    y = train_df['TARGET']

    ## nhey debug
    X_new_col = [re.sub('[^A-Za-z0-9_]+', '', col) for col in X.columns]
    X = X.rename(columns = {X.columns[i]: X_new_col[i] for i in range(len(X_new_col))})
    ##

    return X, y


def LGBMCla_train(X, y, stratified = False, num_folds=2, need_fillna=True):
    # Preset the hyper-params.
    param_grid = ParameterGrid(
                        {
                            'learning_rate': [0.1, 0.01, 0.001],
                            # rf return an error with these parameters
                            # Afters some try, gbdt is the best. To reuse in full demonstration.
                            'boosting_type': ['gbdt'], #, 'dart', 'goss'],
                            # For params = {'boosting_type': 'goss', 'learning_rate': 0.1, 'max_depth': -1, 'metric': 'roc_auc', 'n_estimators': 100, 'n_jobs': 4, 'num_leaves': 31, 'objective': 'binary'} -> ROC AUC score : [0.77987063 0.7800991  0.77470005 0.77978778 0.77859153]
                            # Goss best score

                            # For params = {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': 10, 'metric': 'roc_auc', 'n_estimators': 200, 'n_jobs': 4, 'num_leaves': 31, 'objective': 'binary'} -> ROC AUC score : [0.77940085 0.7791308  0.77155305 0.77884131 0.77750047]
                            # dart best

                            # For params = {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 10, 'metric': 'roc_auc', 'n_estimators': 100, 'n_jobs': 4, 'num_leaves': 31, 'objective': 'binary'} -> ROC AUC score : [0.78435043 0.78482766 0.77777107 0.78458169 0.78171164]
                            # gbdt best
                            'metric': ['roc_auc'],
                            'num_leaves': [10, 20, 31],
                            'max_depth': [-1, 5, 10],
                            # After some try, we see that n_iterations = 20 is not enough and 500 seems to overfit.
                            # Set it to initial value to gain some time.
                            'n_estimators': [50, 100, 200],
                            'objective': ['binary'],
                            'n_jobs': [4]
                        }
                    )

    LGBMC_dic = dict()

    best_score = 0
    nb_comb = len(list(param_grid))

    i=0
    for params in param_grid:
        i+=1
        print(i, "/", nb_comb, sep="")
        # Declare the model
        clf =
        (**params)

        score = np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc'))

        LGBMC_dic[str(params)] = score
        if score > best_score:
            best_score = score
            best_params = params

    with open('models/LGBMC_dic.pkl', 'wb') as f:
        pickle.dump(LGBMC_dic, f)

    return {"mehtod": "LGBM", "best_score": best_score, "params": params}


def SVCCla_train(X, y):
    # Preset the hyper-params.
    param_grid = ParameterGrid(
                        {
                            "C": [0.1, 1, 10],
                            "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"]
                        }
                    )

    best_score = 0
    nb_comb = len(list(param_grid))

    SVC_dic = dict()

    i=0
    for params in param_grid:
        i+=1
        print(i, "/", nb_comb, sep="")
        # Declare the model
        clf = SVC(**params)

        score = np.mean(cross_val_score(clf, X, y, cv=5, scoring='roc_auc'))
        SVC_dic[str(params)] = score

        if score > best_score:
            best_score = score
            best_params = params

    with open('SVC_dic.pkl', 'wb') as f:
        pickle.dump(SVC_dic, f)

    return {"mehtod": "LGBM", "best_score": best_score, "params": best_params}


def XGBClassifier_train(X, y):
    # Inspired by https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390
    # Preset the hyper-params.
    param_grid = ParameterGrid(
        {
            "gamma": [0.5, 0.25, 0],
            "reg_alpha": [0, 1],
            "max_depth": [1, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
        }
    )

    best_score = 0
    nb_comb = len(list(param_grid))

    XGB_dic = dict()

    fb_score = make_scorer(fbeta_score, beta=3)

    i=0
    for params in param_grid:
        i+=1
        print(i, "/", nb_comb, sep="")
        # Declare the model
        xgb_cl = xgb.XGBClassifier(**params)

        score = np.mean(cross_val_score(xgb_cl, X, y, cv=5, scoring=fb_score))
        XGB_dic[str(params)] = score

        if score > best_score:
            print(score, params)
            best_score = score
            best_params = params

    with open('XGB_dic.pkl', 'wb') as f:
        pickle.dump(XGB_dic, f)

    return {"mehtod": "XGB", "best_score": best_score, "params": best_params}


def GBCla_train(X, y):
    # Preset the hyper-params.
    param_grid = ParameterGrid(
                        {
                            "n_estimators": [25, 100, 250],
                            "learning_rate": [0.1, 0.01, 0.001],
                            "max_depth": [1, 5, 10]
                        }
                    )
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=42)


def TSNE_visu_fct(X_tsne, y_cat_num, labels, score=None, method=None):
    """
    *params :
        X_tsne : DataFrame
        y_cat_num : Number of categories
        labels : name of categories
        ARI_score : Put the ARI_score in the title
        method : Put the method in the title.
    """

    ttl = ""
    if method != None:
        ttl += 'Approche utilisée : '+method
    if score != None:
        if len(ttl) > 5:
            ttl += ' - '
        ttl += 'ARI score = '+str(score)

    fig = plt.figure(figsize=(15,6))
    fig.suptitle(ttl)

    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Catégorie")

    title_1 = 'Réprésentation des articles par catégories réelles'

    plt.title(title_1)

    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")

    title_2 = 'Réprésentation des articles par clusters'

    plt.title(title_2)

    plt.show()


def random_undersampling(X, y):
    # https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
    undersample = NearMiss(version=3, n_neighbors_ver3=3)
    # transform the dataset
    X_under, y_under = undersample.fit_resample(X, y)

    return X_under, y_under

def undersampling_with_KNN(X, y):
    # https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/
    # define the undersampling method
    undersample = imblearn.under_sampling.CondensedNearestNeighbour(n_neighbors=1)

    X, y = undersample.fit_resample(X, y)

    counter = Counter(y)


    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.show()

    print("X.shape :", X.shape, "y.shape :", y.shape)

    return X, y


def undersampling_with_centroids(X, y):
    # https://imbalanced-learn.org/stable/under_sampling.html
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)

    return X_resampled, y_resampled


def main(need_fillna=True):
    print("Start of the script.")

    # Read the dataframe
    df = pd.read_pickle('dataframes/df.pkl')
    print("The pretreatment is done with the Kaggle kernel : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script")
    print("The size of the dataframe is :", df.shape)

    # Manage NaNs
    print("For some Classifier models I need to manage NaNs.")
    if need_fillna:
        with timer("Fillna"):
            df_filled = fillna_fun(df, threshold=25)
        df_filled.to_pickle('dataframes/df_fillna.pkl')
    else:
        df_filled = pd.read_pickle('dataframes/df_fillna.pkl')
    print("Size of the dataframe after NaNs remove is", df_filled.shape)

    #############################
    # Start of training models. #
    #############################

    X, y = extract_x_y(df)
    X_filled, y_filled = extract_x_y(df_filled)

    #with timer("undersampling_with_centroids"):
    #    X_filled_under, y_filled_under = undersampling_with_centroids(X_filled, y_filled)
    #with timer("random_undersampling"):
    #    X_filled_under, y_filled_under = random_undersampling(X_filled, y_filled)


    #with open('dataframes/X_filled_under.pkl', 'wb') as f:
    #    pickle.dump(X_filled_under, f)
    #with open('dataframes/y_filled_under.pkl', 'wb') as f:
    #    pickle.dump(y_filled_under, f)

    with open('dataframes/X_filled_under.pkl', 'rb') as f:
        X_filled_under = pickle.load(f)
    with open('dataframes/y_filled_under.pkl', 'rb') as f:
        y_filled_under = pickle.load(f)

    results = list()


    with timer("XGB Classifier"):
        results.append(XGBClassifier_train(X_filled_under, y_filled_under))
        XGB_model = SVC(**results[-1]["params"])
        XGB_model.fit(X_filled_under, y_filled_under)
        with open('XGB_model.pkl', 'wb') as f:
            pickle.dump(XGB_model, f)

    #with timer("SVC Classifier"):
    #    results.append(SVCCla_train(X_filled, y_filled))
    #    SVC_model = SVC(**results[-1]["params"])
    #    SVC_model.fit(X_filled, y_filled)
    #    with open('SVC_model.pkl', 'wb') as f:
    #        pickle.dump(SVC_model, f)

    #with timer("LGBMClassifier"):
    #    results.append(LGBMCla_train(X, y))
    #    LGBM_model= LGBMClassifier(**results[-1]["params"])
    #    LGBM_model.fit(X, y)
    #    with open('LGBM_model.pkl', 'wb') as f:
    #        pickle.dump(LGBM_model, f)

    # Retrain best method and save the fitted classifer.


if __name__ == "__main__":
    main(need_fillna=False)