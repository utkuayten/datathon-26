import numpy as np
import pandas as pd

from preprocess import TARGET, prepare_features
from model import train_catboost


def main():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test_x.csv")
    print(train.shape, test.shape)

    y_train = train[TARGET].values
    X_train = train.drop(columns=[TARGET, "student_id"])
    X_test  = test.drop(columns=["student_id"])

    X_train, X_test, cat_features = prepare_features(X_train, X_test, y_train)

    _, test_preds = train_catboost(X_train, y_train, X_test, cat_features)

    submission = pd.DataFrame({
        "student_id": test["student_id"],
        TARGET: np.clip(test_preds, 0, 100),
    })
    print(submission[TARGET].describe())
    submission.to_csv("submission_catboost_nlp.csv", index=False)
    print("Saved: submission_catboost_nlp.csv")


if __name__ == "__main__":
    main()
