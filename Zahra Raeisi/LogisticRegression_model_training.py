import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Load the preprocessed and labeled dataset
df = pd.read_csv(r"churnLabled_preprocessed_online_shoppers_intention.csv")

# Features: Use the relevant columns for prediction
X = df[['Length', 'Recency', 'Frequency', 'StayingRate']]

# Target: Define 'Churn' column as the target
y = df['Churn']                       # Assumes there's a 'Churn' column indicating churn status (1 = Churn, 0 = No churn)

# Split the data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Check the proportion of churned vs. non-churned in each set
print("Train set class distribution:\n", y_train.value_counts(normalize=True))
print("Test set class distribution:\n", y_test.value_counts(normalize=True))

# Check the shapes of the train/test datasets
print(X_train.shape, X_test.shape)

# Logistic Regression with class_weight to handle imbalance
lr_model_weighted = LogisticRegression(class_weight='balanced')         # 'balanced' handles class imbalance
lr_model_weighted.fit(X_train, y_train)

# Make predictions on the test set
y_pred_lr_weighted = lr_model_weighted.predict(X_test)

# Evaluate the Logistic Regression performance
print("Logistic Regression (Class Weight) Accuracy:", accuracy_score(y_test, y_pred_lr_weighted))
print("Logistic Regression (Class Weight) Classification Report:\n", classification_report(y_test, y_pred_lr_weighted))
print("Logistic Regression (Class Weight) AUC:", roc_auc_score(y_test, y_pred_lr_weighted))

# Logistic Regression Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr_weighted)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp_lr.plot(cmap='coolwarm')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Save the model trined by Logistic Regression
df.to_csv(r"LogisticRegression_trained_model.csv", index=False)

# Perform cross-validation on the LRFS-based Logistic Regression model
cv_scores = cross_val_score(lr_model_weighted, X, y, cv=5, scoring='roc_auc')
print("Cross-validation AUC scores:", cv_scores)
print("Mean AUC score:", cv_scores.mean())

# Example for Logistic Regression tuning
param_grid = {'C': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print("Best parameters for Logistic Regression:", grid_search.best_params_)
