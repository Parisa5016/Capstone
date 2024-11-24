import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the original features data
df_original = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\1.Data Preprocessing\preprocessed_online_shoppers_intention.csv")

# Step 2: Load the KMeans-applied dataset (with PC1, PC2, and Cluster)
df_kmedoids = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\5.Clustering Algorithms\KMeansClusteredUsingT-SNE_1.csv")

# Step 3: Add the cluster labels from the KMeans-applied file to the original dataset
df_original['Cluster'] = df_kmedoids['Cluster']

# Step 4: Save the updated DataFrame with features and cluster labels as a CSV file
output_csv_path = r"C:\Users\alire\Desktop\Capstone Project\6.VisualizationAfterClustering\features_with_KMedoidsClusters_&_t-SNE_1.csv"
df_original.to_csv(output_csv_path, index=False)

# Step 4: List of features to plot
features = ['Length', 'Recency', 'Frequency', 'StayingRate', 'Revenue']

# Step 5: Create subplots for each feature
fig, axs = plt.subplots(1, len(features), figsize=(18, 5))

for i, feature in enumerate(features):
    sns.barplot(x='Cluster', y=feature, data=df_original, ax=axs[i], ci=None, 
                palette='Set1')  # Use 'Set1' color palette for distinct colors
    axs[i].set_title(f'{feature}')
    axs[i].set_xlabel('Cluster')
    axs[i].set_ylabel(feature)

# Add a super title for the entire figure
plt.suptitle("K-Medoids Clustering Analysis on t_SNE-Reduced Dataset", fontsize=16)

plt.tight_layout()
plt.show()
