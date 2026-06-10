import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from preprocess import TARGET, prepare_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test_x.csv")
    y_train = train[TARGET].values
    X_train = train.drop(columns=[TARGET, "student_id"])
    X_test  = test.drop(columns=["student_id"])
    return X_train, X_test, y_train


def make_objective(X_train, y_train, cat_features, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "iterations": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "task_type": "GPU",
            "verbose": False,
        }

        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_val = y_train[tr_idx], y_train[va_idx]

            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=300,
                verbose=False,
            )
            oof[va_idx] = model.predict(X_val)

        return mean_squared_error(y_train, oof) ** 0.5

    return objective


def main():
    print("Loading and preparing features...")
    X_train, X_test, y_train = load_data()
    X_train, X_test, cat_features = prepare_features(X_train, X_test, y_train)

    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(X_train, y_train, cat_features), n_trials=50,
                   show_progress_bar=True)

    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()