import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check current directory
print("Working Directory:", os.getcwd())

# Load original features data
df_original = pd.read_csv("Alireza/data/RandomForest_trained_model.csv")
print("Loaded df_original successfully.")

# Load KMeans-applied dataset
df_kmeans = pd.read_csv("Alireza/Data/KMeans_ReClustered_PCA_with_churn.csv")
print("Loaded df_kmeans successfully.")

# Add cluster labels
df_original['Cluster'] = df_kmeans['Cluster']

# Save updated DataFrame
output_csv_path = "data/features_with_KMeans_ReClusters_PCA.csv"
df_original.to_csv(output_csv_path, index=False)
print(f"Saved updated file to {output_csv_path}")

# Plot features
features = ['Length', 'Recency', 'Frequency', 'StayingRate', 'Revenue', 'Churn']
fig, axs = plt.subplots(1, len(features), figsize=(20, 5))

for i, feature in enumerate(features):
    sns.barplot(x='Cluster', y=feature, data=df_original, ax=axs[i], ci=None, palette='Set1')
    axs[i].set_title(f'{feature}')
    axs[i].set_xlabel('Cluster')
    axs[i].set_ylabel(feature)

plt.suptitle("K-Means Clustering Analysis on PCA-Reduced Dataset", fontsize=16)
plt.tight_layout()
plt.show()
