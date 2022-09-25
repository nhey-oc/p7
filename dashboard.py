import streamlit as st
import pandas as pd
import numpy as np
import requests
import json


import plotly.figure_factory as ff
import matplotlib.pyplot as plt

def print_client_vs_distribution(X, client, col):

    heigh_ratio = 3/5
    width_ratio = 1/100
    scatter_radius = 75
    nb_bins = min(X[col].nunique(), 20)

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

def choose_your_client(X):

    #Add sidebar to the app
    st.sidebar.markdown("### Please choose your client")

    # Let the user choose one of the 10 clients.
    id_list=X["SK_ID_CURR"].unique().tolist()
    id_list.sort(reverse=True)
    client_id = st.sidebar.selectbox("Client ID", id_list, index=0)
    client = X[X["SK_ID_CURR"] == client_id].iloc[0]

    return client

# Load full data
X_init = pd.read_pickle('dataframes/X_filled_random_undersampled.pkl')
# Take 10 random customers
X_sample = X_init.sample(10)

client = choose_your_client(X_sample)

client_dict = client.to_dict()
response = requests.post('http://127.0.0.1:8000/predict', json=client_dict)
response_json = int(json.loads(response.content.decode('utf-8'))['prediction'])

#Add title and subtitle to the main interface of the app
estimation = "Client will reimburse." if response_json == 0 else "Client will not reimburse."
st.title(estimation)
st.markdown(response_json)

col1, col2 = st.columns(2)

with col1:
    print_client_vs_distribution(X_init, client, "AMT_GOODS_PRICE")
    print_client_vs_distribution(X_init, client, "AMT_CREDIT")
    print_client_vs_distribution(X_init, client, "CNT_FAM_MEMBERS")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")
