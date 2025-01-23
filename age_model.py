import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt
import joblib


def load_data(filepath):
    """Load dataset from a file."""
    return pd.read_csv(filepath)


def preprocess_data(data):
    """Preprocess data by splitting features and target."""
    X = data.drop(['ID', 'Age', 'Age_group'], axis=1)
    y = data['Age']
    return X, y


def train_regressor(X_train, y_train):
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test):
    """Evaluate the performance of the regressor."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return mse, r2


def convert_to_age_groups(y, bins, labels):
    """Convert continuous ages to categorical age groups."""
    return pd.cut(y, bins=bins, labels=labels)


def train_classifier(X_train, y_train_cat):
    """Train a Random Forest Classifier."""
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train_cat)
    return clf


def evaluate_classifier(clf, X_test, y_test_cat, labels):
    """Evaluate the performance of the classifier."""
    y_pred_cat = clf.predict(X_test)
    accuracy = accuracy_score(y_test_cat, y_pred_cat)
    f1 = f1_score(y_test_cat, y_pred_cat, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1}")

    # Save metrics to a file
    with open("metrics.txt", "w") as outfile:
        outfile.write(f"\nAccuracy = {round(accuracy, 2)}, "
                      f"F1 Score = {round(f1, 2)}\n\n")

    # Create and save confusion matrix
    cm = confusion_matrix(y_test_cat, y_pred_cat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig("age_pred_model_results.png", dpi=120)

    return accuracy, f1


def save_model(model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


if __name__ == "__main__":
    # Load data
    data = load_data("age_prediction_dataset.csv")

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate regressor
    regressor = train_regressor(X_train, y_train)
    evaluate_regressor(regressor, X_test, y_test)

    # Define age group bins and labels
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["<18", "18-29", "30-44", "45-59", "60+"]

    # Convert ages to age groups
    y_train_cat = convert_to_age_groups(y_train, bins, labels)
    y_test_cat = convert_to_age_groups(y_test, bins, labels)

    # Train and evaluate classifier
    classifier = train_classifier(X_train, y_train_cat)
    evaluate_classifier(classifier, X_test, y_test_cat, labels)

    # Save the trained classifier
    save_model(classifier, "age_prediction_model.joblib")
