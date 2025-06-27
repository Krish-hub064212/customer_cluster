# Prepare cluster of customers to predict their purchase power based on their income and spending score.
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
df =pd.read_csv('Mall_Customers.csv')

print(df.info())

x=df[["Annual Income (k$)","Spending Score (1-100)"]]

wcss_list = []
for i in range(1, 11):
    kmeans=KMeans(n_clusters=i, init='k-means++',random_state=1)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
print(wcss_list)
plt.plot(range(1, 11), wcss_list)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


model = KMeans(n_clusters=6, init='k-means++', random_state=1)
y_predict= model.fit_predict(x)

print(y_predict)


X_array = x.values

plt.scatter(X_array[y_predict == 0, 0], X_array[y_predict == 0, 1], s=100, color='red')
plt.scatter(X_array[y_predict == 1, 0], X_array[y_predict == 1, 1], s=100, color='blue')
plt.scatter(X_array[y_predict == 2, 0], X_array[y_predict == 2, 1], s=100, color='green')
plt.scatter(X_array[y_predict == 3, 0], X_array[y_predict == 3, 1], s=100, color='cyan')
plt.scatter(X_array[y_predict == 4, 0], X_array[y_predict == 4, 1], s=100, color='magenta')
plt.scatter(X_array[y_predict == 5, 0], X_array[y_predict == 5, 1], s=100, color='yellow')

plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


joblib.dump(model,'Model.pkl')
print("Model saved as Model.pkl")

