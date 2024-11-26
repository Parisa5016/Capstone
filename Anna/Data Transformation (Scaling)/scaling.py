import pandas as pd

df_scaled = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\1.Data Preprocessing\preprocessed_online_shoppers_intention.csv")

# # Check the range of each feature
# print(df_scaled[['Length', 'Recency', 'Frequency', 'StayingRate']].describe())

from sklearn.preprocessing import StandardScaler

# Select the four features to scale
features_to_scale = ['Length', 'Recency', 'Frequency', 'StayingRate']

# Apply standard scaling (mean = 0, std = 1)
scaler = StandardScaler()
df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

# Min-Max Scaling (Optional)
# minmax_scaler = MinMaxScaler()
# df_normalized = minmax_scaler.fit_transform(df_scaled[features_to_scale])

# Check the range again after scaling
print(df_scaled[features_to_scale].describe())

# Save the corrected scaled dataset
df_scaled.to_csv(r"C:\Users\alire\Desktop\Capstone Project\2.Data Transformation (Scaling)\scaled_preprocessed_online_shoppers_intention_not_MinMax", index=False)
