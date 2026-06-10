import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

from preprocess import TARGET, prepare_features
from model import train_catboost, train_lightgbm

MODELS = {"catboost": train_catboost, "lightgbm": train_lightgbm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="catboost")
    parser.add_argument("--no-tfidf", action="store_true")
    args = parser.parse_args()

    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test_x.csv")
    print(train.shape, test.shape)

    y_train = train[TARGET].values
    X_train = train.drop(columns=[TARGET, "student_id"])
    X_test  = test.drop(columns=["student_id"])

    X_train, X_test, cat_features = prepare_features(X_train, X_test, y_train, use_tfidf=not args.no_tfidf)

    _, test_preds, metrics = MODELS[args.model](X_train, y_train, X_test, cat_features)

    submission = pd.DataFrame({
        "student_id": test["student_id"],
        TARGET: np.clip(test_preds, 0, 100),
    })
    print(submission[TARGET].describe())
    output_file = f"submissions/submission_{args.model}.csv"
    submission.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {"model": args.model, "timestamp": timestamp, **metrics}
    result_file = f"results/{args.model}_{timestamp}.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {result_file}")


if __name__ == "__main__":
    main()
