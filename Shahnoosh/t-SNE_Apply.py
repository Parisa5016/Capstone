import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df_scaled = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\2.Data Transformation (Scaling)\scaled_preprocessed_online_shoppers_intention_not_MinMax.csv")

# Drop the original non-numeric columns
df_scaled = df_scaled.drop(columns=['Month', 'VisitorType'])

# Apply t-SNE to reduce the data to 2 dimensions
tsne = TSNE(n_components = 2, random_state = 42)
df_tsne = tsne.fit_transform(df_scaled)

# Convert the t-SNE result (NumPy array) to a DataFrame
df_tsne_df = pd.DataFrame(df_tsne, columns=['Dimension 1', 'Dimension 2'])

# Save the t-SNE result to a CSV file
df_tsne_df.to_csv(r"C:\Users\alire\Desktop\Capstone Project\4.Dimensionality_Reduction_Techniques\t-SNE_transformed_online_shoppers_intention_1.csv", index=False)

# Plot the t-SNE result
plt.scatter(df_tsne[:, 0], df_tsne[:, 1])
plt.title("t-SNE visualization of customers")
plt.show()
