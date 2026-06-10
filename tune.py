import argparse
import subprocess
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from preprocess import TARGET, prepare_features

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_task_type():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "GPU"
    except Exception:
        return "CPU"


TASK_TYPE = get_task_type()
print(f"Task type: {TASK_TYPE}")


def load_data():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test_x.csv")
    y_train = train[TARGET].values
    X_train = train.drop(columns=[TARGET, "student_id"])
    X_test  = test.drop(columns=["student_id"])
    return X_train, X_test, y_train


def make_catboost_objective(X_train, y_train, cat_features, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
            "depth": trial.suggest_int("depth", 3, 5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5, 50, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "bootstrap_type": "Bernoulli",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "task_type": TASK_TYPE,
            "verbose": False,
        }

        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_val = y_train[tr_idx], y_train[va_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, cat_features=cat_features,
                      eval_set=(X_val, y_val), early_stopping_rounds=300, verbose=False)
            oof[va_idx] = model.predict(X_val)

        return mean_squared_error(y_train, oof) ** 0.5

    return objective


def make_lightgbm_objective(X_train, y_train, cat_features, n_splits=5, random_state=42):
    X_train = X_train.copy()
    for c in cat_features:
        X_train[c] = X_train[c].astype("category")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "device": "gpu" if TASK_TYPE == "GPU" else "cpu",
            "verbose": -1,
            "random_state": 42,
        }

        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_val = y_train[tr_idx], y_train[va_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = model.predict(X_val)

        return mean_squared_error(y_train, oof) ** 0.5

    return objective


OBJECTIVES = {
    "catboost": make_catboost_objective,
    "lightgbm": make_lightgbm_objective,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=OBJECTIVES.keys(), default="catboost")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--no-tfidf", action="store_true")
    args = parser.parse_args()

    print("Loading and preparing features...")
    X_train, X_test, y_train = load_data()
    X_train, X_test, cat_features = prepare_features(X_train, X_test, y_train, use_tfidf=not args.no_tfidf)

    objective = OBJECTIVES[args.model](X_train, y_train, cat_features)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest CV RMSE: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
