import pandas as pd

# Load the clustered dataset
df_clustered = pd.read_csv(r"preprocessed_online_shoppers_intention.csv")

# Define thresholds for Recency, Frequency, and Staying Rate
recency_threshold = 1                # Customers who haven’t interacted in over one month are likely at risk of churn.
frequency_threshold = 1              # Very low interaction frequency, indicating minimal engagement.
staying_rate_threshold = 2           # Low engagement per session, suggesting the customer isn’t highly interested when they do interact.

# Create the Churn column: 1 for churned, 0 for active
df_clustered['Churn'] = df_clustered.apply(lambda row: 
    1 if (row['Recency'] > recency_threshold and 
          row['Frequency'] <= frequency_threshold and 
          row['StayingRate'] <= staying_rate_threshold) 
    else 0, axis=1)

# Save the KMeans-PCA clustered dataset containing "Churn" column
df_clustered.to_csv('churnLabled_preprocessed_online_shoppers_intention.csv', index=False)

print(df_clustered.Churn)
