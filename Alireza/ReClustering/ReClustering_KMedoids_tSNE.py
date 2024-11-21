import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the new 'Churn' feature
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\8.Churn Labeling\churnLabled_preprocessed_online_shoppers_intention.csv")

# Select features for clustering (including the new Churn feature)
X = df[['Length', 'Recency', 'Frequency', 'StayingRate', 'Churn']]

# Step 1: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply t-SNE for dimensionality reduction (2 components for 2D visualization)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)
df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
df_tsne['Churn'] = df['Churn']

# Step 3: Determine the optimal number of clusters using the Elbow Method on t-SNE reduced data
sse = []
k_range = range(1, 9)
for k in k_range:
    kmedoids = KMedoids(n_clusters=k, random_state=42, metric='euclidean')
    kmedoids.fit(X_tsne)
    # Calculate the total cost as sum of distances to medoids
    sse.append(kmedoids.inertia_)  # inertia_ in sklearn_extra’s KMedoids represents sum of distances to medoids

# Plot the elbow graph
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Distances (Cost)')
plt.title('Elbow Method for Optimal k (t-SNE reduced data)')
plt.show()

# Step 4: Apply K-Medoids with the chosen k (e.g., k=4 based on Elbow plot)
optimal_k = 4
kmedoids = KMedoids(n_clusters=optimal_k, random_state=42)
clusters = kmedoids.fit_predict(X_tsne)
df_tsne['Cluster'] = clusters

# Step 5: Visualize the clusters in the t-SNE reduced space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='viridis', style='Churn', markers=['o', 'X'])
plt.title(f'Customer Segmentation with {optimal_k} Clusters (t-SNE Reduced Data, Including Churn)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster')
plt.show()

# Optional: Save the t-SNE reduced dataset with cluster labels
df_tsne.to_csv(r"C:\Users\alire\Desktop\Capstone Project\10.Reclustering After Churn Labelling\KMedoids_ReClustered_tSNE_with_churn.csv", index=False)
