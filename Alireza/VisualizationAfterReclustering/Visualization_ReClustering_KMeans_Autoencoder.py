import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns  # for creating a bar plot.

# Step 1: Load the original features data
df_original = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\8.Churn Labeling\churnLabled_preprocessed_online_shoppers_intention.csv")

# Step 2: Load the KMeans-applied dataset (with PC1, PC2, and Cluster)
df_kmeans = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\10.Reclustering After Churn Labelling\KMeans_ReClustered_Autoencoder_with_churn.csv")

# Step 3: Add the cluster labels from the KMeans-applied file to the original dataset
df_original['Cluster'] = df_kmeans['Cluster']

# Step 4: Save the updated DataFrame with features and cluster labels as a CSV file
output_csv_path = r"C:\Users\alire\Desktop\Capstone Project\11.Visualization After ReClustering\features_with_KMeans_ReClusters_Autoencoder.csv"
df_original.to_csv(output_csv_path, index=False)

# Step 5: List of features to plot, including Churn
features = ['Length', 'Recency', 'Frequency', 'StayingRate', 'Revenue', 'Churn']

# Step 6: Create subplots for each feature
fig, axs = plt.subplots(1, len(features), figsize=(20, 5))

for i, feature in enumerate(features):
    sns.barplot(x='Cluster', y=feature, data=df_original, ax=axs[i], ci=None, 
                palette='Set1')  # Use 'Set1' color palette for distinct colors
    axs[i].set_title(f'{feature}')
    axs[i].set_xlabel('Cluster')
    axs[i].set_ylabel(feature)

# Add a super title for the entire figure
plt.suptitle("K-Means Clustering Analysis on Autoencoder-Reduced Dataset", fontsize=16)

plt.tight_layout()

# Show the plot
plt.show()
