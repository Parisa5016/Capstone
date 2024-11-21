import pandas as pd

df = pd.read_csv(r"oroginal_online_shoppers_intention.csv") 

# df.head()                                 # Display the first few rows of the dataset
# df.info()                                 # Get general information about the dataset
# df.describe()                             # Get basic statistics of numerical columns

df = df.isnull().sum()                      # Check for missing values

df.drop_duplicates(inplace = True)          # Remove duplicate rows

# Create 'total_page_view' by summing page views
df['total_page_view'] = df['Administrative'] + df['Informational'] + df['ProductRelated']

# Create 'total_page_duration' by summing duration columns
df['total_page_duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']

# Drop the six original columns related to page views and duration
df = df.drop(columns=['Administrative', 'Informational', 'ProductRelated', 'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration'])

# Drop unnecessary features
df = df.drop(columns=['Browser', 'Region', 'OperatingSystems', 'TrafficType', 'SpecialDay', 'Weekend'])

# Map 'Month' to numerical values
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# Convert 'Revenue' to binary (1 for True, 0 for False)
df['Revenue'] = df['Revenue'].astype(int)

# Convert 'VisitorType' to numerical values (1 for Returning, 0 for New and Others)
df['Month_Numeric'] = df['Month'].map(month_mapping)

def calculate_length(visitor_type, month_numeric):
    if visitor_type == 'Returning_Visitor':
        return month_numeric  # Difference from January (month 1)
    else:
        return 1  # Default 1 month for New_Visitor
    
# Calculate Length    
df['Length'] = df.apply(lambda row: calculate_length(row['VisitorType'], row['Month_Numeric']), axis=1)

# Calculate Recency
df['Recency'] = 12 - df['Month_Numeric'] + 1

# Calculate Frequency
df['Frequency'] = df['total_page_view']

# Calculate Staying Rate
df['StayingRate'] = df['PageValues'] * (1 - df['ExitRates'])

# Save the preprocessed dataset
df.to_csv(r"preprocessed_online_shoppers_intention.csv", index=False)

