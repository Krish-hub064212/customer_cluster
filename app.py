import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
import streamlit as st

#loadthe saved model
kmeans = joblib.load('Model.pkl')
df = pd.read_csv('Mall_Customers.csv')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_array = X.values

#streamlitpplicationpage
st.set_page_config(page_title='Customer Segmentation', layout='centered')
st.title('Customer cluster')
st.write('Customer clustering using K-Means Clustering')

#inputs
annual_income = st.number_input("annual income of the customer",min_value=0,max_value=400,value = 50)
spending_score = st.slider("spending score of the customer btn 1-100",1,100,20)

#predit the cluster
if st.button("predict"):
    input_data = np.array([[annual_income,spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f'The customer belongs to cluster {cluster}')
