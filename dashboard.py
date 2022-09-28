import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pickle
import shap

import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def print_client_vs_distribution(X, client, col, bins=None):

    heigh_ratio = 3/5
    width_ratio = 1/100
    scatter_radius = 75
    if bins == None:
        nb_bins = min(X[col].nunique(), 20)
    else:
        nb_bins = bins

    fig, ax = plt.subplots()
    ax.hist(X[col], bins=nb_bins)
    low, top = plt.ylim()
    left, right = plt.xlim()

    heigh = top*heigh_ratio

    plt.bar(client[col], heigh, color='red', width=(right-left)*width_ratio)
    plt.scatter(client[col], heigh, color='red', s=scatter_radius)

    title = col+ " distribution"
    plt.title(title)

    st.pyplot(fig)

def choose_your_client(X, y):

    #Add sidebar to the app
    st.sidebar.markdown("### Please choose your client")

    # Let the user choose one of the 10 clients.
    id_list=X["index"].unique().tolist()
    id_list.sort(reverse=True)
    client_id = st.sidebar.selectbox("Client ID", id_list, index=0)
    client = X[X["index"] == client_id].iloc[0]

    return client, y.loc[client['index']]

def main():
    # Load full data
    X_train = pd.read_pickle('dataframes/X_undersampled.pkl')
    X_test = pd.read_pickle('dataframes/X_test_undersampled.pkl')
    y_test = pd.read_pickle('dataframes/y_test_undersampled.pkl')
    with open("dataframes/shap_dict.pkl", "rb") as f:
        shap_dict = pickle.load(f)

    print(X_test['index'])

    client, failed_to_reimburse = choose_your_client(X_test, y_test)

    client_dict = client.to_dict()
    response = requests.post('http://127.0.0.1:8000/predict', json=client_dict)
    response_json = int(json.loads(response.content.decode('utf-8'))['prediction'])

    #Add title and subtitle to the main interface of the app
    estimation = "We think that this client will reimburse." if response_json == 0 else "We think that this client will not reimburse."
    st.title(estimation)
    reality = "And the client will reimburse. :)" if failed_to_reimburse == 0 else "And the client will'nt reimburse."
    st.markdown(reality)

    col1, col2 = st.columns(2)

    with col1:
        print_client_vs_distribution(X_train, client, "AMT_GOODS_PRICE")
        print_client_vs_distribution(X_train, client, "AMT_CREDIT")
        print_client_vs_distribution(X_train, client, "CNT_FAM_MEMBERS", bins= 22)

    with col2:
        fig = plt.figure(num=1, clear=True)
        shap.bar_plot(shap_dict[client["index"]],
                      feature_names=X_test.columns,
                      max_display=8, show=False)
        plt.savefig('temp.png', bbox_inches='tight')
        st.image("temp.png")

if __name__ == "__main__":
    main()