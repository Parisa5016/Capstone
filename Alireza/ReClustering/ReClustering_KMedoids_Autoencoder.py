import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with the new 'Churn' feature
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\8.Churn Labeling\churnLabled_preprocessed_online_shoppers_intention.csv")

# Select features for clustering (including the new Churn feature)
X = df[['Length', 'Recency', 'Frequency', 'StayingRate', 'Churn']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define and train the Autoencoder for 2-dimensional encoding
input_dim = X_scaled.shape[1]
encoding_dim = 2  # Reducing to 2 components

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)

# Extract the encoder model to reduce the data
encoder = Model(input_layer, encoded)
X_autoencoded = encoder.predict(X_scaled)
df_autoencoded = pd.DataFrame(X_autoencoded, columns=['Autoenc1', 'Autoenc2'])
df_autoencoded['Churn'] = df['Churn']

sse = []
k_range = range(1, 9)
for k in k_range:
    kmedoids = KMedoids(n_clusters=k, random_state=42, metric='euclidean')
    kmedoids.fit(X_autoencoded)
    # Calculate the total cost as sum of distances to medoids
    sse.append(kmedoids.inertia_)  # inertia_ in sklearn_extra’s KMedoids represents sum of distances to medoids

# Plot the elbow graph
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Sum of Distances (Cost)')
plt.title('Elbow Method for Optimal k (Autoencoder reduced data)')
plt.show()

# Apply K-Medoids clustering with Autoencoder reduced data
kmedoids = KMedoids(n_clusters=4, random_state=42)
clusters = kmedoids.fit_predict(X_autoencoded)
df_autoencoded['Cluster'] = clusters

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_autoencoded, x='Autoenc1', y='Autoenc2', hue='Cluster', palette='viridis', style='Churn', markers=['o', 'X'])
plt.title('Customer Segmentation with K-Medoids (Autoencoder Reduced Data)')
plt.xlabel('Autoencoder Component 1')
plt.ylabel('Autoencoder Component 2')
plt.legend(title='Cluster')
plt.show()

# Save the Autoencoder-reduced dataset with cluster labels
df_autoencoded.to_csv(r"C:\Users\alire\Desktop\Capstone Project\9.Model Training & Evaluating\KMedoids_ReClustered_Autoencoder_with_churn.csv", index=False)
