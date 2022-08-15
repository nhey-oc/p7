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

import shap

@contextmanager
def timer(title):
    t0 = time.time()
    print("---------------------------\n", title)
    yield
    print(" - done in {:.0f}s".format(time.time() - t0))
    print("End of", title)
    print("---------------------------")


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
    #test_df = df[df['TARGET'].isnull()]

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
        clf = LGBMClassifier(**params)

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


def proceed_to_undersample(X, y, undersampling_method = "random"):

    if undersampling_method == "centroids":
        # Deal with centroids undersampling
        with timer("undersampling_with_centroids"):
            X_undersampled, y_undersampled = undersampling_with_centroids(X, y)

    elif undersampling_method == "random":
        # Deal with random undersampling
        with timer("random_undersampling"):
            X_undersampled, y_undersampled = random_undersampling(X, y)

    return X_undersampled, y_undersampled


def main(need_fillna=True):
    print("Start of the script.")

    #######################################
    # Read the full pretreated dataframe. #
    #######################################

    df = pd.read_pickle('dataframes/df.pkl')
    print("The pretreatment is done with the Kaggle kernel : https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script")
    print("The size of the dataframe is :", df.shape)

    ################
    # Manage NaNs. #
    ################

    print("For some Classifier models I need to manage NaNs.")

    if need_fillna:
        with timer("Fillna"):
            df_filled = fillna_fun(df, threshold=25)
            df_filled.to_pickle('dataframes/df_fillna.pkl')
    else:
        df_filled = pd.read_pickle('dataframes/df_fillna.pkl')
    print("Size of the dataframe after NaNs remove is", df_filled.shape)

    #########################################
    # Separate X from y for each dataframe. #
    #########################################

    X, y = extract_x_y(df)
    X_filled, y_filled = extract_x_y(df_filled)

    #################################
    # Start of deal with unbalance. #
    #################################

    undersampling_needed = True

    if undersampling_needed:
            X_filled_random_undersampled,\
            y_filled_random_undersampled = proceed_to_undersample(X_filled,
                                                                  y_filled,
                                                                  undersampling_method="random")
            with open('dataframes/X_filled_random_undersampled.pkl', 'wb') as f:
                pickle.dump(X_filled_random_undersampled, f)
            with open('dataframes/y_filled_random_undersampled.pkl', 'wb') as f:
                pickle.dump(y_filled_random_undersampled, f)

            X_filled_centroids_undersampled, \
            y_filled_centroids_undersampled = proceed_to_undersample(X_filled,
                                                                  y_filled,
                                                                  undersampling_method="centroids")
            with open('dataframes/X_filled_centroids_undersampled.pkl', 'wb') as f:
                pickle.dump(X_filled_centroids_undersampled, f)
            with open('dataframes/y_filled_centroids_undersampled.pkl', 'wb') as f:
                pickle.dump(y_filled_centroids_undersampled, f)

    else:
        with open('dataframes/X_filled_random_undersampled.pkl', 'rb') as f:
            X_filled_random_undersampled = pickle.load(f)
        with open('dataframes/y_filled_random_undersampled.pkl', 'rb') as f:
            y_filled_random_undersampled = pickle.load(f)

        with open('dataframes/X_filled_centroids_undersampled.pkl', 'rb') as f:
            X_filled_centroids_undersampled = pickle.load(f)
        with open('dataframes/y_filled_centroids_undersampled.pkl', 'rb') as f:
            y_filled_centroids_undersampled = pickle.load(f)

    print("New shape with undersampling method is : X : ",
          X_filled_random_undersampled.shape,
          "  y : ",
          y_filled_random_undersampled.shape,
          sep="")

    #############################
    # Start of training models. #
    #############################

    results = dict()

    # Trying the 3 models with default params

    SVC_clf = SVC()
    SVC_score = np.mean(cross_val_score(SVC_clf,
                                        X_filled_undersampled,
                                        y_filled_undersampled,
                                        cv=5,
                                        scoring='roc_auc'))
    results['SVC'] = SVC_score
    print("SVC default score is : ", SVC_score)

    XGBC_clf = xgb.XGBClassifier()
    XGBC_score = np.mean(cross_val_score(XGBC_clf,
                                         X_filled_undersampled,
                                         y_filled_undersampled,
                                         cv=5,
                                         scoring='roc_auc'))
    results['XGB'] = XGBC_score
    print("XGB default score is : ", XGBC_score)

    #LGBMC_clf = LGBMClassifier()
    #LGBMC_all_datas_score = np.mean(cross_val_score(LGBMC_clf,
    #                                                X,
    #                                                y,
    #                                                cv=5,
    #                                                scoring='roc_auc'))
    #results['LGBMC all datas'] = LGBMC_all_datas_score
    #print("Light GBMC default score is : ", LGBMC_score)

    LGBMC_clf = LGBMClassifier()
    LGBMC_all_datas_score = np.mean(cross_val_score(LGBMC_clf,
                                                    X_filled_undersampled,
                                                    y_filled_undersampled,
                                                    cv=5,
                                                    scoring='roc_auc'))
    results['LGBM'] = LGBMC_all_datas_score
    print("Light GBM default score is : ", LGBMC_score)

    best_method = max(results, key=results.get)



    if best_method == "SVC":
        with timer("SVC Classifier"):
            score_and_params = SVCCla_train(X_filled_undersampled, y_filled_undersampled)
            SVC_model = SVC(**score_and_params["params"])
            SVC_model.fit(X_filled_undersampled, y_filled_undersampled)
            with open('SVC_model.pkl', 'wb') as f:
                pickle.dump(SVC_model, f)
            model = SVC_model

    elif best_method == "XGB":
        with timer("XGB Classifier"):
            score_and_params = XGBClassifier_train(X_filled_undersampled, y_filled_undersampled)
            XGB_model = SVC(**score_and_params["params"])
            XGB_model.fit(X_filled_undersampled, y_filled_undersampled)
            with open('XGB_model.pkl', 'wb') as f:
                pickle.dump(XGB_model, f)
            model = XGB_model

    elif best_method == "LGBM":
        with timer("LGBMClassifier"):
            score_and_params = LGBMCla_train(X_filled_undersampled, y_filled_undersampled)
            LGBM_model= LGBMClassifier(**score_and_params["params"])
            LGBM_model.fit(X_filled_undersampled, y_filled_undersampled)
            with open('LGBM_model.pkl', 'wb') as f:
                pickle.dump(LGBM_model, f)
            model = LGBM_model

    partition_explainer = shap.PartitionExplainer(model, X_filled_undersampled)

    shap.bar_plot(partition_explainer.shap_values(X_filled_undersampled[0]),
              feature_names=df.columns,
              max_display=12)

if __name__ == "__main__":
    main(need_fillna=False)