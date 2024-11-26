# Let's perform visual and statistical inspections on the provided dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the scaled dataset
df_scaled = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\2.Data Transformation (Scaling)\scaled_preprocessed_online_shoppers_intention_not_MinMax.csv")

# List the features of interest for scaling check
features = ['Length', 'Recency', 'Frequency', 'StayingRate']

# Plot histograms for visual inspection of distribution
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Distribution of Scaled Features")

for i, feature in enumerate(features):
    row, col = divmod(i, 2)    
    sns.histplot(df_scaled[feature], ax=axs[row, col], kde=True, color='b', stat='density', linewidth=1.5)
    axs[row, col].set_title(f'{feature} Distribution')

plt.tight_layout()
plt.show()

# Display basic statistics for the scaled features
df_scaled[features].describe()

