import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset with the new 'Churn' feature
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\8.Churn Labeling\churnLabled_preprocessed_online_shoppers_intention.csv")

# Select features for clustering (including the new Churn feature)
X = df[['Length', 'Recency', 'Frequency', 'StayingRate', 'Churn']]

# Step 1: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Define the Autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 2  # Compress to 2 dimensions for visualization

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Compile the Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Train the Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

# Extract the Encoder part for dimensionality reduction
encoder = Model(input_layer, encoded)
X_autoencoded = encoder.predict(X_scaled)
df_autoencoded = pd.DataFrame(X_autoencoded, columns=['Autoenc1', 'Autoenc2'])
df_autoencoded['Churn'] = df['Churn']

# Step 3: Determine the optimal number of clusters using the Elbow Method on Autoencoder-reduced data
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_autoencoded)
    sse.append(kmeans.inertia_)

plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k (Autoencoder-reduced data)')
plt.show()

# Step 4: Apply K-Means with the chosen k (e.g., k=4 based on Elbow plot)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_autoencoded)
df_autoencoded['Cluster'] = clusters

# Step 5: Visualize the clusters in the Autoencoder-reduced space
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_autoencoded, x='Autoenc1', y='Autoenc2', hue='Cluster', palette='viridis', style='Churn', markers=['o', 'X'])
plt.title(f'Customer Segmentation with {optimal_k} Clusters (Autoencoder Reduced Data, Including Churn)')
plt.xlabel('Autoencoder Component 1')
plt.ylabel('Autoencoder Component 2')
plt.legend(title='Cluster')
plt.show()

# Optional: Save the Autoencoder-reduced dataset with cluster labels
df_autoencoded.to_csv(r"C:\Users\alire\Desktop\Capstone Project\10.Reclustering After Churn Labelling\ReClustered_Autoencoder_with_churn.csv", index=False)
