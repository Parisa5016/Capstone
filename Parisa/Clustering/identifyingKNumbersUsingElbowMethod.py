import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_PCA = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\4.Dimensionality_Reduction_Techniques\PCA_transformed_online_shoppers_intention_1.csv")

# Use the Elbow Method to find the optimal K
inertia = []                                # Represents the sum of squared distances between data points and their nearest cluster center
for k in range(1, 10):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(df_PCA)
    inertia.append(kmeans.inertia_)         # Storing the inertia values for different numbers of clusters.

# Plot the Elbow Method result
plt.plot(range(1, 10), inertia)
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of k")
plt.ylabel("Inertia")
plt.show
plt.pause(40)

# Fit K-Means with the optimal number of clusters (e.g., K = 4)
kmeans = KMeans(n_clusters=4, random_state=42)
df_PCA['Cluster'] = kmeans.fit_predict(df_PCA) 

# Analyze the clusters
print(df_PCA['Cluster'].value_counts())

