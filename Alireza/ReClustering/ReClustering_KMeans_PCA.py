import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the new 'Churn' feature
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\9.Model Training & Evaluating\LogisticRegression_trained_model.csv")

# Select features for clustering (including the new Churn feature)
X = df[['Length', 'Recency', 'Frequency', 'StayingRate', 'Churn']]

# Step 1: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA for dimensionality reduction (2 components for 2D visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Churn'] = df['Churn']

# Step 3: Determine optimal k for clustering using the Elbow Method on PCA-reduced data
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    sse.append(kmeans.inertia_)

plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k (PCA-reduced data)')
plt.show()

# Step 4: Apply K-Means with the chosen k (e.g., k=5 based on Elbow plot)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df_pca['Cluster'] = clusters

# Step 5: Visualize the clusters in the PCA-reduced space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='viridis', style='Churn', markers=['o', 'X'])
plt.title(f'Customer Segmentation with {optimal_k} Clusters (PCA-reduced data, Including Churn)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Optional: Save the PCA-reduced dataset with cluster labels
df_pca.to_csv(r"C:\Users\alire\Desktop\Capstone Project\10.Reclustering After Churn Labelling\KMeans_ReClustered_PCA_with_churn.csv", index=False)
