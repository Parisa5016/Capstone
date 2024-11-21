import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the 3-component PCA-transformed dataset
df_scaled = pd.read_csv("PCA_transformed_online_shoppers_intention_1.csv")

# Step 2: Apply K-Means clustering with k=4
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(df_scaled)

# Add the cluster labels to the DataFrame
df_scaled['Cluster'] = labels

# Step 3: Save the dataframe with cluster labels as a CSV file
output_path = r"C:\Users\alire\Desktop\Capstone Project\5.Clustering Algorithms\KMeansClusteredUsingPCA_1.csv"
df_scaled.to_csv(output_path, index=False)

# Step 4: Plot the clusters
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis') # Use .iloc[] to slice the DataFrame
plt.title('K-Means Clustering with PCA (k=4)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
