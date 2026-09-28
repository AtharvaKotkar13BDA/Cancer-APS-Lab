import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )
    y = pd.Series(
        (data.target == 0).astype(int),
        name="malignant"
    )
    return data, X, y


def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test