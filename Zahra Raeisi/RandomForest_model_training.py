import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the preprocessed and labeled dataset
df = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\8.Churn Labeling\churnLabled_preprocessed_online_shoppers_intention.csv")

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

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate Random Foret performance
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Random Forest AUC:", roc_auc_score(y_test, y_pred_rf))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Save the model trined by Logistic Regression
df.to_csv(r"C:\Users\alire\Desktop\Capstone Project\9.Model Training & Evaluating\RandomForest_trained_model.csv", index=False)

