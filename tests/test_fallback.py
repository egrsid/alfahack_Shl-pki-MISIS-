import numpy as np
import pandas as pd
import pytest
from p3_fixtures import Spec, classification_task, regression_task, write_task
from sklearn.model_selection import StratifiedKFold

import fallback as F
import metrics
import submission

FAST = dict(time_budget=60, threads=1, models=("lightgbm",), max_rounds=300)


def _cfg(**kw):
    return F.FallbackConfig(**{**FAST, **kw})


def test_regression_end_to_end(tmp_path):
    spec, y_test = regression_task(tmp_path, n_train=800, n_test=200)
    cand = F.run(spec, config=_cfg(models=("lightgbm", "catboost", "xgboost")))
    assert submission.check_file(spec) == []
    assert cand.oof.shape == (800,) and cand.test_pred.shape == (200,)
    assert len(cand.meta["fold_scores"]) == cand.meta["n_splits"] == 5
    assert cand.meta["folds"].shape == (800,) and cand.meta["classes"] is None
    assert {r["name"] for r in cand.meta["runs"]} >= {"lightgbm", "catboost", "xgboost"}
    constant = metrics.score("rmse", y_test, np.full(200, y_test.mean()))
    written = pd.read_csv(tmp_path / "submission.csv")["target"]
    assert metrics.score("rmse", y_test, written) < 0.5 * constant  # модель заметно лучше константы


def test_multiclass_string_labels(tmp_path):
    spec, y_test = classification_task(tmp_path, n_classes=4)
    cand = F.run(spec, config=_cfg())
    assert cand.meta["classes"] == ["alpha", "beta", "delta", "gamma"]
    assert cand.test_pred.shape == (300, 4) and np.allclose(cand.test_pred.sum(axis=1), 1)
    out = pd.read_csv(tmp_path / "submission.csv")
    assert set(out["label"]) <= set(cand.meta["classes"]) and submission.check_file(spec) == []
    assert cand.meta["cv_score_post"] >= cand.cv_score  # пост-обработка принимается, только если не хуже
    assert metrics.score("macro_f1", y_test, out["label"]) > 0.5


def test_binary_auc_writes_probabilities(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=2, labels=[0, 1], metric="roc_auc")
    cand = F.run(spec, config=_cfg())
    out = pd.read_csv(tmp_path / "submission.csv")
    assert out["label"].between(0, 1).all() and out["label"].nunique() > 10
    assert np.allclose(out["label"], cand.test_pred[:, 1])
    assert cand.meta["decision"] is None and 0.5 < cand.cv_score <= 1


def test_imbalanced_macro_f1_plans_balanced_variant(tmp_path):
    spec, _ = classification_task(tmp_path, n_classes=5, imbalance=True)
    train = pd.read_csv(tmp_path / "train.csv")
    train.drop(train.index[train["label"] == "eps"][20:]).to_csv(tmp_path / "train.csv", index=False)  # 20 объектов
    data = F.load(spec, _cfg())
    counts = np.bincount(data.y)
    assert counts.max() / counts.min() >= 5
    names = [name for name, _, _ in F._plan(data, F.FallbackConfig())]
    assert names == ["lightgbm", "catboost", "lightgbm_balanced", "xgboost"]


def test_failing_model_does_not_break_others(tmp_path, monkeypatch):
    spec, _ = regression_task(tmp_path)
    original = F._fit_predict

    def flaky(model, *args, **kw):
        if model == "catboost":
            raise RuntimeError("boom")
        return original(model, *args, **kw)
    monkeypatch.setattr(F, "_fit_predict", flaky)
    cand = F.run(spec, config=_cfg(models=("catboost", "lightgbm")))
    assert cand.meta["model"] == "lightgbm" and submission.check_file(spec) == []


def test_all_models_fail_gives_constant(tmp_path, monkeypatch):
    spec, _ = classification_task(tmp_path, n_classes=3)
    monkeypatch.setattr(F, "_fit_predict", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no models")))
    cand = F.run(spec, config=_cfg())
    assert cand.name == "fallback_constant" and submission.check_file(spec) == []


def test_unknown_metric_and_broken_deadline(tmp_path):
    spec, _ = regression_task(tmp_path)
    spec.metric = "some exotic score"

    class Deadline:
        def left(self):
            return 10_000.0

        def stage(self, name):
            raise KeyError(name)
    cand = F.run(spec, Deadline(), config=_cfg())
    assert cand.meta["metric"] == "rmse" and submission.check_file(spec) == []


def test_short_deadline_is_respected(tmp_path):
    spec, _ = classification_task(tmp_path, n_train=3000, n_test=500, n_classes=3)

    class Deadline:
        def left(self):
            return 30.0  # 30 с − резерв 20 с → бюджет упирается в нижнюю границу 20 с
    import time
    start = time.monotonic()
    F.run(spec, Deadline(), config=F.FallbackConfig(threads=1))
    assert time.monotonic() - start < 45 and submission.check_file(spec) == []


def test_tiny_dataset_and_rare_class(tmp_path):
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(30), "x": rng.normal(size=30), "y": ["a"] * 14 + ["b"] * 15 + ["c"]})
    test = pd.DataFrame({"id": range(30, 40), "x": rng.normal(size=10)})
    write_task(tmp_path, train, test, pd.DataFrame({"id": test.id, "y": "a"}))
    spec = Spec(tmp_path, "id", "y", "multiclass", "macro_f1", True)
    cand = F.run(spec, config=_cfg())
    assert cand.test_pred.shape == (10, 3) and submission.check_file(spec) == []


def test_continuous_target_is_regression_whatever_hints_say(tmp_path):
    spec, _ = regression_task(tmp_path)
    spec.task_type, spec.metric = "multiclass", "macro_f1"  # ошибка понимания задачи
    cand = F.run(spec, config=_cfg())
    assert cand.meta["metric"] == "rmse" and cand.oof.ndim == 1 and submission.check_file(spec) == []
    y = pd.Series(np.random.default_rng(0).normal(size=500))
    assert F.infer_task_type(y, "accuracy") == "regression"
    assert F.infer_task_type(pd.Series([0, 1, 2] * 50), "accuracy") == "multiclass"


def test_folds_match_stratified_kfold_with_seed():
    y = np.array([0] * 50 + [1] * 30 + [2] * 20)
    folds = F.make_folds(y, "multiclass", 5, 42)
    for i, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(100), y)):
        assert (folds[valid] == i).all()


def test_guess_spec_from_files(tmp_path):
    regression_task(tmp_path / "reg")
    spec = F.guess_spec(tmp_path / "reg" / "task.md")
    assert (spec.id_col, spec.target, spec.task_type, spec.metric) == ("id", "target", "regression", "rmse")
    classification_task(tmp_path / "clf", n_classes=2, labels=["no", "yes"], metric="ROC AUC")
    spec = F.guess_spec(tmp_path / "clf")
    assert (spec.target, spec.task_type, spec.metric) == ("label", "binary", "roc_auc")


def test_log_target_tried_for_skewed_regression(tmp_path):
    spec, _ = regression_task(tmp_path)
    train = pd.read_csv(tmp_path / "train.csv")
    train["target"] = np.exp(train["target"] / 3)  # положительный скошенный таргет
    train.to_csv(tmp_path / "train.csv", index=False)
    cand = F.run(spec, config=_cfg(write=False))
    assert "lightgbm_log" in {r["name"] for r in cand.meta["runs"]}
    assert not (tmp_path / "submission.csv").exists()  # write=False — файл не трогаем


def test_multi_target_regression(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    df = pd.DataFrame({"id": range(n), "x": x, "t1": 2 * x + rng.normal(scale=.1, size=n), "t2": -x})
    write_task(tmp_path, df.iloc[:200], df.iloc[200:, :2], pd.DataFrame({"id": df.id[200:], "t1": 0.0, "t2": 0.0}))
    spec = F.guess_spec(tmp_path)
    assert spec.target == ["t1", "t2"] and spec.task_type == "regression"
    cand = F.run_multi(spec, config=_cfg())
    assert cand.test_pred.shape == (100, 2) and submission.check_file(spec) == []


def test_n_threads_is_positive():
    assert F.n_threads() >= 1
    assert F.model_threads("lightgbm", 3_000, 8) == 1 and F.model_threads("catboost", 3_000, 8) == 8


@pytest.mark.parametrize("budget_left, expected", [(1000.0, 900.0), (100.0, 80.0), (5.0, 20.0)])
def test_budget_from_deadline(budget_left, expected):
    class Deadline:
        def left(self):
            return budget_left
    assert F._budget(Deadline(), F.FallbackConfig()) == pytest.approx(expected)
