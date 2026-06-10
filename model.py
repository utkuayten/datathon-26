import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

SEEDS = [42, 123, 777, 2024, 3407]


def train_catboost(X_train, y_train, X_test, cat_features, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof        = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_val = y_train[tr_idx], y_train[va_idx]

        fold_val_pred  = np.zeros(len(X_val))
        fold_test_pred = np.zeros(len(X_test))

        for seed in SEEDS:
            model = CatBoostRegressor(
                iterations=4000,
                learning_rate=0.02,
                depth=5,
                loss_function="RMSE",
                eval_metric="RMSE",
                random_seed=seed,
                verbose=False,
            )
            model.fit(
                X_tr, y_tr,
                cat_features=cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=300,
                verbose=False,
            )
            fold_val_pred  += model.predict(X_val)  / len(SEEDS)
            fold_test_pred += model.predict(X_test) / len(SEEDS)

        oof[va_idx]  = fold_val_pred
        test_preds  += fold_test_pred / kf.n_splits

        print(f"Fold {fold + 1}: MSE={mean_squared_error(y_val, fold_val_pred):.4f}")

    cv_mse = mean_squared_error(y_train, oof)
    print(f"\nCV MSE: {cv_mse:.4f}  |  CV RMSE: {cv_mse ** 0.5:.4f}")

    return oof, test_preds
