import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import joblib

# Load data from Kaggle
# (Note: You'll need to download the dataset and provide the local path)
data = pd.read_csv("age_prediction_dataset.csv")

# For simplicity, use the existing features without additional engineering
# Split the data into features and target
X = data.drop(['ID', 'Age', 'Age_group'], axis=1)
y = data['Age']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create a Random Forest Regressor model
model = RandomForestRegressor(random_state=42)

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
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, "
                  f"F1 Score = {round(f1, 2)}\n\n")

# 5. Create and display confusion matrix
cm = confusion_matrix(y_test_cat, y_pred_cat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig("age_pred_model_results.png", dpi=120)


# Save the trained model using joblib
filename = 'age_prediction_model.joblib'
joblib.dump(clf, filename)
print(f"Model saved to {filename}")
