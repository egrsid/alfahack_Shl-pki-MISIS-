"""Первая версия: простой LightGBM без LLM. Запуск: python3 solution.py task.md

Читает train.csv, test.csv, sample_submission.csv из каталога task.md
и пишет submission.csv рядом с ним.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    x_train, x_test = train[features].copy(), test[features].copy()
    for col in features:
        if x_train[col].dtype == object or x_train[col].dtype == bool:
            both = pd.concat([x_train[col], x_test[col]]).astype(str)
            cats = pd.CategoricalDtype(sorted(both.unique()))
            x_train[col] = x_train[col].astype(str).astype(cats)
            x_test[col] = x_test[col].astype(str).astype(cats)
    return x_train, x_test


def main() -> None:
    task_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve().parent
    train = pd.read_csv(task_dir / "train.csv")
    test = pd.read_csv(task_dir / "test.csv")
    sample = pd.read_csv(task_dir / "sample_submission.csv")

    id_col, target = sample.columns[0], sample.columns[1]
    features = [c for c in test.columns if c != id_col and c in train.columns]
    x_train, x_test = prepare(train, test, features)

    y = train[target]
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, n_jobs=8, verbose=-1)

    if is_regression:
        model_cls, labels = lgb.LGBMRegressor, y.to_numpy()
    else:
        encoder = LabelEncoder().fit(y.astype(str))
        model_cls, labels = lgb.LGBMClassifier, encoder.transform(y.astype(str))
        params["class_weight"] = "balanced"

    # быстрая оценка на отложенной части, чтобы видеть качество в логе
    xa, xb, ya, yb = train_test_split(x_train, labels, test_size=0.2, random_state=0)
    check = model_cls(**params).fit(xa, ya)
    pred = check.predict(xb)
    if is_regression:
        print("holdout RMSE:", mean_squared_error(yb, pred) ** 0.5)
    else:
        print("holdout macro F1:", f1_score(yb, pred, average="macro"))

    model = model_cls(**params).fit(x_train, labels)
    result = model.predict(x_test)
    if not is_regression:
        result = encoder.inverse_transform(result.astype(int))

    submission = pd.DataFrame({id_col: test[id_col], target: result})
    submission.to_csv(task_dir / "submission.csv", index=False)
    print("записано:", task_dir / "submission.csv", submission.shape)


if __name__ == "__main__":
    main()
