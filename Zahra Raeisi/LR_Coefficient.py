import numpy as np
import matplotlib.pyplot as plt

# Get the coefficients from the logistic regression model
coefficients = lr_model_weighted.coef_[0]

# Create a DataFrame to hold the feature names and their corresponding coefficients
feature_names = ['Length', 'Recency', 'Frequency', 'StayingRate']
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort the coefficients by absolute value (importance)
coeff_df['Abs_Coefficient'] = np.abs(coeff_df['Coefficient'])
coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot the coefficients
plt.figure(figsize=(8, 6))
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.show()

# Display the sorted DataFrame
print(coeff_df)