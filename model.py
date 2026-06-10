import subprocess
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEEDS = [42, 123, 777, 2024, 3407]


def get_task_type():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "GPU"
    except Exception:
        return "CPU"


TASK_TYPE = get_task_type()
print(f"Task type: {TASK_TYPE}")



def train_catboost(X_train, y_train, X_test, cat_features, n_splits=5, random_state=42):
    params = dict(
        iterations=4000, learning_rate=0.02, depth=3,
        loss_function="RMSE", eval_metric="RMSE", task_type=TASK_TYPE, verbose=False,
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof        = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y_train[tr_idx], y_train[va_idx]

        fold_train_pred = np.zeros(len(X_tr))
        fold_val_pred   = np.zeros(len(X_val))
        fold_test_pred  = np.zeros(len(X_test))

        for seed in SEEDS:
            model = CatBoostRegressor(**params, random_seed=seed)
            model.fit(X_tr, y_tr, cat_features=cat_features,
                      eval_set=(X_val, y_val), early_stopping_rounds=300, verbose=False)
            fold_train_pred += model.predict(X_tr)   / len(SEEDS)
            fold_val_pred   += model.predict(X_val)  / len(SEEDS)
            fold_test_pred  += model.predict(X_test) / len(SEEDS)

        oof[va_idx]  = fold_val_pred
        test_preds  += fold_test_pred / kf.n_splits

        train_mse = mean_squared_error(y_tr, fold_train_pred)
        val_mse   = mean_squared_error(y_val, fold_val_pred)
        fold_scores.append({"fold": fold + 1, "train_mse": round(train_mse, 4), "val_mse": round(val_mse, 4)})
        print(f"Fold {fold + 1}: Train MSE={train_mse:.4f}  Val MSE={val_mse:.4f}")

    cv_mse = mean_squared_error(y_train, oof)
    print(f"\nCV MSE: {cv_mse:.4f}  |  CV RMSE: {cv_mse ** 0.5:.4f}")

    print("\nFull train fit...")
    full_test_preds = np.zeros(len(X_test))
    for seed in SEEDS:
        model = CatBoostRegressor(**params, random_seed=seed)
        model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
        full_test_preds += model.predict(X_test) / len(SEEDS)

    metrics = {"params": params, "fold_scores": fold_scores,
               "cv_mse": round(cv_mse, 4), "cv_rmse": round(cv_mse ** 0.5, 4)}
    return oof, full_test_preds, metrics


def train_lightgbm(X_train, y_train, X_test, cat_features, n_splits=5, random_state=42):
    X_train = X_train.copy()
    X_test  = X_test.copy()
    for c in cat_features:
        X_train[c] = X_train[c].astype("category")
        X_test[c]  = X_test[c].astype("category")

    params = dict(
        n_estimators=1000, learning_rate=0.02, num_leaves=31, max_depth=-1,
        reg_alpha=0.1, reg_lambda=5.0, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        device="gpu" if TASK_TYPE == "GPU" else "cpu", verbose=-1,
    )
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof        = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y_train[tr_idx], y_train[va_idx]

        fold_train_pred = np.zeros(len(X_tr))
        fold_val_pred   = np.zeros(len(X_val))
        fold_test_pred  = np.zeros(len(X_test))

        for seed in SEEDS:
            model = lgb.LGBMRegressor(**params, random_state=seed)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)])
            fold_train_pred += model.predict(X_tr)   / len(SEEDS)
            fold_val_pred   += model.predict(X_val)  / len(SEEDS)
            fold_test_pred  += model.predict(X_test) / len(SEEDS)

        oof[va_idx]  = fold_val_pred
        test_preds  += fold_test_pred / kf.n_splits

        train_mse = mean_squared_error(y_tr, fold_train_pred)
        val_mse   = mean_squared_error(y_val, fold_val_pred)
        fold_scores.append({"fold": fold + 1, "train_mse": round(train_mse, 4), "val_mse": round(val_mse, 4)})
        print(f"Fold {fold + 1}: Train MSE={train_mse:.4f}  Val MSE={val_mse:.4f}")

    cv_mse = mean_squared_error(y_train, oof)
    print(f"\nCV MSE: {cv_mse:.4f}  |  CV RMSE: {cv_mse ** 0.5:.4f}")

    print("\nFull train fit...")
    full_test_preds = np.zeros(len(X_test))
    for seed in SEEDS:
        model = lgb.LGBMRegressor(**params, random_state=seed)
        model.fit(X_train, y_train)
        full_test_preds += model.predict(X_test) / len(SEEDS)

    metrics = {"params": params, "fold_scores": fold_scores,
               "cv_mse": round(cv_mse, 4), "cv_rmse": round(cv_mse ** 0.5, 4)}
    return oof, full_test_preds, metrics