import pandas as pd
import matplotlib.pyplot as plt

df_scaled = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\2.Data Transformation (Scaling)\scaled_preprocessed_online_shoppers_intention_not_MinMax.csv")

# Drop the original non-numeric columns
df_scaled = df_scaled.drop(columns=['Month', 'VisitorType'])

# Now apply PCA on the corrected dataset
from sklearn.decomposition import PCA

# Assuming other features are scaled, include 'Month_Numeric' and the transformed 'VisitorType'
pca = PCA(n_components = 2)
df_pca = pca.fit_transform(df_scaled)

# Check explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")

# Convert PCA-transformed data back to a DataFrame
df_pca_df = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])

# Save the PCA-transformed data
df_pca_df.to_csv(r"C:\Users\alire\Desktop\Capstone Project\4.Dimensionality_Reduction_Techniques\PCA_transformed_online_shoppers_intention_1.csv", index=False)

# Create a scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], color='blue', edgecolor='k', alpha=0.6)
plt.title('PCA - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
