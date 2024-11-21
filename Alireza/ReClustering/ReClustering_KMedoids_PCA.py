import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the new 'Churn' feature
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\9.Model Training & Evaluating\RandomForest_trained_model.csv")

# Select features for clustering (including the new Churn feature)
X = df[['Length', 'Recency', 'Frequency', 'StayingRate', 'Churn']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Churn'] = df['Churn']

# Determine optimal k for clustering using the Elbow Method on PCA-reduced data
sse = []
k_range = range(1, 9)
for k in k_range:
    kmedoids = KMedoids(n_clusters=k, random_state=42, metric='euclidean')
    kmedoids.fit(X_pca)
    # Calculate the total cost as sum of distances to medoids
    sse.append(kmedoids.inertia_)  # inertia_ in sklearn_extra’s KMedoids represents sum of distances to medoids

# Plot the elbow graph
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Distances (Cost)')
plt.title('Elbow Method for Optimal k (t-PCA reduced data)')
plt.show()

# Apply K-Medoids clustering with PCA-reduced data
kmedoids = KMedoids(n_clusters=6, random_state=42)
clusters = kmedoids.fit_predict(X_pca)
df_pca['Cluster'] = clusters

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', style='Churn', markers=['o', 'X'])
plt.title('Customer Segmentation with K-Medoids (PCA Reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Save the PCA-reduced dataset with cluster labels
df_pca.to_csv(r"C:\Users\alire\Desktop\Capstone Project\10.Reclustering After Churn Labelling\KMedoids_ReClustered_PCA_with_churn.csv", index=False)
