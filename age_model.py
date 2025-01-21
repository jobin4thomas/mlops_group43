import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load data from Kaggle 
# (Note: You'll need to download the dataset and provide the local path)
data = pd.read_csv("age_prediction_dataset.csv") 

# Separate features and target variable
features = data.drop("Age", axis=1)
target = data["Age"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Create a Random Forest Regressor model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# For classification, you'd need to convert continuous 'Age' to categorical 
# (e.g., age groups) and use a classification model. 
# Here's a simplified example with assumptions:

# 1. Discretize 'Age' into age groups (for illustration)
bins = [0, 18, 30, 45, 60, 100] 
labels = ["<18", "18-29", "30-44", "45-59", "60+"]
y_train_cat = pd.cut(y_train, bins=bins, labels=labels)
y_test_cat = pd.cut(y_test, bins=bins, labels=labels)

# 2. Train a classification model (example: Random Forest Classifier)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train_cat)

# 3. Predict on test data
y_pred_cat = clf.predict(X_test)

# 4. Calculate classification metrics
accuracy = accuracy_score(y_test_cat, y_pred_cat)
f1 = f1_score(y_test_cat, y_pred_cat, average='weighted') 
print(f"Accuracy: {accuracy}")
print(f"F1-score: {f1}")
## Write metrics to file
with open("metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n\n")

# 5. Create and display confusion matrix
cm = confusion_matrix(y_test_cat, y_pred_cat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig("age_pred_model_results.png", dpi=120)


