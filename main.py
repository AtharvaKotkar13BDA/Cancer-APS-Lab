import matplotlib.pyplot as plt
import pandas as pd
from data import (
    load_dataset,
    split_dataset
)

from model import train_model

from evaluation import (
    get_probabilities,
    create_results,
    evaluate_threshold,
    threshold_analysis
)


def main():

    data, X, y = load_dataset()
    print(y.value_counts())
    print("Feature matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("Class names:", data.target_names)



    class_counts = y.value_counts().sort_index()

    class_distribution = pd.DataFrame({
        "Class": data.target_names,
        "Count": class_counts.values,
        "Probability": class_counts.values / len(y)
    })

    print(class_distribution)

    class_distribution.plot(
        x="Class",
        y="Count",
        kind="bar",
        legend=False,
        color=["tomato", "steelblue"]
    )

    plt.ylabel("Number of observations")
    plt.title("Class Distribution")
    plt.xticks(rotation=0)
    plt.show()

    print("Class distribution completed")

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    print("Training size:", len(y_train))
    print("Testing size:", len(y_test))

    print("Training proportions")

    print(
        y_train.value_counts(
            normalize=True
        ).sort_index()
    )

    model = train_model(
        X_train,
        y_train
    )


    probabilities = get_probabilities(
        model,
        X_test
    )

    print(probabilities[:5])

    print(
        probabilities[:5].sum(axis=1)
    )


    results = create_results(
        probabilities,
        y_test
    )

    print(results.head(10))

    print(
        results[
            [
                "Actual label",
                "p_malignant",
                "p_benign"
            ]
        ].head(10)
    )


    threshold = 0.50

    y_pred, accuracy, precision, recall, f1score = (
        evaluate_threshold(
            y_test,
            probabilities,
            threshold
        )
    )

    results["Predicted_malignant"] = y_pred

    results["Predicted_label"] = results[
        "Predicted_malignant"
    ].map({
        1: "Malignant",
        0: "Benign"
    })

    print(
        results[
            [
                "Actual label",
                "p_malignant",
                "Predicted_label"
            ]
        ].head(10)
    )


    for threshold in [0.30, 0.50, 0.70]:

        predictions = (
            results["p_malignant"] >= threshold
        ).astype(int)

        print(
            f"Threshold = {threshold}: "
            f"Predicted malignant cases = "
            f"{predictions.sum()}"
        )


    print(
        "Sklearn Accuracy:",
        accuracy
    )

    print(
        "Sklearn Precision:",
        precision
    )

    print(
        "Sklearn Recall:",
        recall
    )

    print(
        "Sklearn F1 Score:",
        f1score
    )

    result_s = threshold_analysis(
        y_test,
        probabilities
    )

    print("\nThreshold Analysis")

    print(result_s)


if __name__ == "__main__":
    main()