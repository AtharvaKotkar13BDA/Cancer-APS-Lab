import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def get_probabilities(model, X_test):
    probabilities = model.predict_proba(X_test)
    return probabilities


def create_results(probabilities, y_test):
    results = pd.DataFrame({
        "Actual class": y_test.values,
        "p_malignant": probabilities[:, 0],
        "p_benign": probabilities[:, 1]
    })

    results["Actual label"] = results["Actual class"].map({
        0: "Benign",
        1: "Malignant"
    })

    return results


def evaluate_threshold(y_test, probabilities, threshold):
    y_pred = (
        probabilities[:, 1] >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1score = f1_score(
        y_test,
        y_pred
    )

    return y_pred, accuracy, precision, recall, f1score


def threshold_analysis(y_test, probabilities):
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    result_s = []
    for i in thresholds:
        y_pred, accuracy, precision, recall, f1score = (
            evaluate_threshold(
                y_test,
                probabilities,
                i
            )
        )

        result_s.append({
            "threshold": i,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score
        })

    result_s = pd.DataFrame(result_s)

    return result_s