import pytest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from age_model import (
    preprocess_data,
    train_regressor,
    convert_to_age_groups,
    train_classifier
)


@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "ID": [1, 2, 3, 4, 5],
        "Feature1": [10, 20, 30, 40, 50],
        "Feature2": [15, 25, 35, 45, 55],
        "Age": [25, 35, 45, 55, 65],
        "Age_group": [None, None, None, None, None],  # Placeholder
    })
    return data


def test_preprocess_data(sample_data):
    X, y = preprocess_data(sample_data)
    assert "ID" not in X.columns
    assert len(y) == len(sample_data)


def test_train_regressor(sample_data):
    X, y = preprocess_data(sample_data)
    regressor = train_regressor(X, y)
    assert isinstance(regressor, RandomForestRegressor)


def test_convert_to_age_groups(sample_data):
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["<18", "18-29", "30-44", "45-59", "60+"]
    y_cat = convert_to_age_groups(sample_data["Age"], bins, labels)
    assert all(label in labels for label in y_cat.unique())


def test_train_classifier(sample_data):
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["<18", "18-29", "30-44", "45-59", "60+"]
    _, y = preprocess_data(sample_data)
    y_cat = convert_to_age_groups(y, bins, labels)
    classifier = train_classifier(_, y_cat)
    assert isinstance(classifier, RandomForestClassifier)


def test_evaluate_classifier(sample_data):
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["<18", "18-29", "30-44", "45-59", "60+"]
    X, y = preprocess_data(sample_data)
    y_cat = convert_to_age_groups(y, bins, labels)
    classifier = train_classifier(X, y_cat)
    y_pred_cat = classifier.predict(X)
    accuracy = accuracy_score(y_cat, y_pred_cat)
    f1 = f1_score(y_cat, y_pred_cat, average='weighted')
    assert 0 <= accuracy <= 1
    assert 0 <= f1 <= 1
