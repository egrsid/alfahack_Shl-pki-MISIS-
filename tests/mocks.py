"""Заглушки модулей команды для тестов агента (вне ZIP).

MockLLM       — вместо llm_client (P2): ответы по сценарию для каждой роли промпта.
StubEvaluator — вместо evaluator.evaluate (P1): K-fold на фиксированном сиде, LightGBM по умолчанию.
StubTuner     — вместо tuning.tune (P1): заданный ответ или маленький перебор сетки тем же evaluator.
SimpleDeadline — вместо budget.Deadline (P1).
SpecStub      — поля TaskSpec из schemas.py (P1), пока его нет в main.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold

from agent import Candidate


@dataclass
class SpecStub:
    task_dir: Path
    id_col: str
    target: str
    task_type: str
    metric: str
    greater_is_better: bool
    feature_notes: dict[str, str] = field(default_factory=dict)
    drop_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    date_cols: list[str] = field(default_factory=list)
    text_cols: list[str] = field(default_factory=list)
    raw_task_text: str = ""
    source: str = "heuristic"


class SimpleDeadline:
    def __init__(self, seconds: float) -> None:
        self.end = time.monotonic() + seconds

    def left(self) -> float:
        return max(0.0, self.end - time.monotonic())

    def stage(self, name: str) -> float:
        return self.left()


class MockLLM:
    """Отдаёт ответы по сценарию: script = {"planner": [...], "coder": [...], "critic": [...]}.

    Роль берётся из первой строки системного промпта («Role: coder.»). Элемент сценария — строка,
    dict (для chat_json), исключение (бросается) или callable(messages). Когда сценарий роли
    исчерпан, отдаётся нейтральный ответ из DEFAULTS.
    """

    offline = True  # не сетевая модель: LLM_DISABLED=1 её не отключает
    DEFAULTS = {"planner": '{"hypotheses": []}', "coder": "Sorry, no code.", "critic": "Nothing to add."}

    def __init__(self, script: dict[str, list] | None = None, tokens: tuple[int, int] = (1000, 200)) -> None:
        self.script = {role: list(items) for role, items in (script or {}).items()}
        self.tokens = tokens
        self.calls: list[tuple[str, list[dict], dict]] = []
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @staticmethod
    def role(messages: list[dict]) -> str:
        found = re.search(r"Role:\s*(\w+)", messages[0]["content"]) if messages else None
        return found.group(1).lower() if found else "unknown"

    def _next(self, messages: list[dict], kw: dict) -> Any:
        role = self.role(messages)
        self.calls.append((role, messages, kw))
        self.usage["prompt_tokens"] += self.tokens[0]
        self.usage["completion_tokens"] += self.tokens[1]
        queue = self.script.get(role, [])
        item = queue.pop(0) if queue else self.DEFAULTS.get(role, "")
        if isinstance(item, BaseException):
            raise item
        return item(messages) if callable(item) else item

    def chat(self, messages: list[dict], **kw: Any) -> str:
        item = self._next(messages, kw)
        return item if isinstance(item, str) else json.dumps(item)

    def chat_json(self, messages: list[dict], schema: dict, **kw: Any) -> dict:
        item = self._next(messages, kw)
        if isinstance(item, str):
            try:
                return json.loads(item)
            except json.JSONDecodeError as error:  # так же ведёт себя клиент P2 после своих попыток починки
                raise ValueError(f"invalid JSON: {error}") from error
        return item

    def count(self, role: str) -> int:
        return sum(1 for r, _, _ in self.calls if r == role)


# ---------------------------------------------------------------- evaluator

def _score(metric: str, y: np.ndarray, pred: np.ndarray, regression: bool) -> float:
    name = metric.lower().replace("-", "_").replace(" ", "_")
    if regression:
        if name == "rmse":
            return float(mean_squared_error(y, pred) ** 0.5)
        if name == "mse":
            return float(mean_squared_error(y, pred))
        if name == "mae":
            return float(mean_absolute_error(y, pred))
        if name == "r2":
            return float(r2_score(y, pred))
        raise ValueError(f"unknown regression metric: {metric}")
    labels = pred.argmax(axis=1)
    if name in ("macro_f1", "f1_macro"):
        return float(f1_score(y, labels, average="macro"))
    if name in ("weighted_f1", "f1_weighted"):
        return float(f1_score(y, labels, average="weighted"))
    if name == "accuracy":
        return float(accuracy_score(y, labels))
    if name in ("logloss", "log_loss"):
        return float(log_loss(y, pred, labels=list(range(pred.shape[1]))))
    if name in ("roc_auc", "auc"):
        return float(roc_auc_score(y, pred[:, 1]) if pred.shape[1] == 2
                     else roc_auc_score(y, pred, multi_class="ovr"))
    raise ValueError(f"unknown classification metric: {metric}")


def _to_model_frame(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train, x_test = x_train.copy(), x_test.copy()
    for col in x_train.columns:
        if x_train[col].dtype == object or isinstance(x_train[col].dtype, pd.CategoricalDtype):
            both = pd.concat([x_train[col], x_test[col]]).astype(object)
            dtype = pd.CategoricalDtype(sorted(both.dropna().astype(str).unique()))
            x_train[col] = x_train[col].astype(object).where(x_train[col].isna(), x_train[col].astype(str)).astype(dtype)
            x_test[col] = x_test[col].astype(object).where(x_test[col].isna(), x_test[col].astype(str)).astype(dtype)
        elif x_train[col].dtype == bool:
            x_train[col], x_test[col] = x_train[col].astype("int8"), x_test[col].astype("int8")
    return x_train, x_test


class StubEvaluator:
    """Заглушка evaluator.evaluate(spec, build_features, params, deadline) -> Candidate.

    params = {"model": "lightgbm|catboost|xgboost", "params": {...}, "target_transform": None|"log1p"}.
    Сплиты и сид фиксированы, поэтому кандидаты сравнимы между собой (в том числе по fold_scores).
    """

    def __init__(self, n_splits: int = 5, seed: int = 42, n_estimators: int = 300,
                 learning_rate: float = 0.05, threads: int = 4) -> None:
        self.n_splits, self.seed = n_splits, seed
        self.n_estimators, self.learning_rate, self.threads = n_estimators, learning_rate, threads
        self.calls = 0
        self._cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def _load(self, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = str(spec.task_dir)
        if key not in self._cache:
            task_dir = Path(spec.task_dir)
            self._cache[key] = (pd.read_csv(task_dir / "train.csv", low_memory=False),
                                pd.read_csv(task_dir / "test.csv", low_memory=False))
        return self._cache[key]

    def _model(self, name: str, params: dict, regression: bool) -> Any:
        if name == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            base = dict(iterations=self.n_estimators, learning_rate=self.learning_rate, random_seed=self.seed,
                        verbose=0, thread_count=self.threads, allow_writing_files=False)
            return (CatBoostRegressor if regression else CatBoostClassifier)(**{**base, **params})
        if name == "xgboost":
            from xgboost import XGBClassifier, XGBRegressor
            base = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.seed,
                        n_jobs=self.threads, tree_method="hist", enable_categorical=True)
            return (XGBRegressor if regression else XGBClassifier)(**{**base, **params})
        from lightgbm import LGBMClassifier, LGBMRegressor
        base = dict(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.seed,
                    n_jobs=self.threads, verbose=-1)
        return (LGBMRegressor if regression else LGBMClassifier)(**{**base, **params})

    def __call__(self, spec: Any, build_features: Any, params: dict, deadline: Any = None) -> Candidate:
        self.calls += 1
        train, test = self._load(spec)
        drop = {spec.id_col, spec.target, *spec.drop_cols}
        feats = [c for c in test.columns if c in train.columns and c not in drop]
        spec_dict = {"task_type": spec.task_type, "metric": spec.metric, "cat_cols": spec.cat_cols,
                     "date_cols": spec.date_cols, "text_cols": spec.text_cols}
        x, x_test = build_features(train[feats].copy(), test[feats].copy(), spec_dict)
        x, x_test = _to_model_frame(x.reset_index(drop=True), x_test.reset_index(drop=True))

        name = (params or {}).get("model") or "lightgbm"
        model_params = dict((params or {}).get("params") or {})
        regression = spec.task_type == "regression"
        target = train[spec.target]
        if regression:
            y = target.to_numpy(dtype=float)
            log_target = (params or {}).get("target_transform") == "log1p" and y.min() > -1
            splitter, n_classes = KFold(self.n_splits, shuffle=True, random_state=self.seed), 0
            oof, test_pred = np.zeros(len(x)), np.zeros(len(x_test))
        else:
            classes = np.array(sorted(target.astype(str).unique()))
            y = np.searchsorted(classes, target.astype(str).to_numpy())
            log_target, n_classes = False, len(classes)
            splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
            oof, test_pred = np.zeros((len(x), n_classes)), np.zeros((len(x_test), n_classes))

        if name == "catboost":
            cats = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
            x, x_test = x.astype({c: str for c in cats}), x_test.astype({c: str for c in cats})
            model_params.setdefault("cat_features", cats)

        fold_scores = []
        for train_idx, valid_idx in splitter.split(x, y):
            model = self._model(name, model_params, regression)
            y_fit = np.log1p(y[train_idx]) if log_target else y[train_idx]
            model.fit(x.iloc[train_idx], y_fit)
            if regression:
                p_valid, p_test = model.predict(x.iloc[valid_idx]), model.predict(x_test)
                if log_target:
                    p_valid, p_test = np.expm1(p_valid), np.expm1(p_test)
            else:  # колонки вероятностей по model.classes_, если в фолде не было какого-то класса
                cls = np.asarray(model.classes_).astype(int)
                p_valid, p_test = np.zeros((len(valid_idx), n_classes)), np.zeros((len(x_test), n_classes))
                p_valid[:, cls], p_test[:, cls] = model.predict_proba(x.iloc[valid_idx]), model.predict_proba(x_test)
            oof[valid_idx] = p_valid
            test_pred += p_test / self.n_splits
            fold_scores.append(_score(spec.metric, y[valid_idx], p_valid, regression))

        return Candidate(name=f"stub_{name}", cv_score=float(np.mean(fold_scores)),
                         cv_std=float(np.std(fold_scores)), oof=oof, test_pred=test_pred,
                         meta={"fold_scores": fold_scores, "model": name, "params": model_params,
                               "target_transform": "log1p" if log_target else None,
                               "n_splits": self.n_splits, "seed": self.seed})


class StubTuner:
    """Заглушка tuning.tune(spec, build_features, cfg, deadline) -> dict (P1).

    result — что вернуть (dict) или исключение (бросается). С evaluator и grid — маленький перебор:
    каждый набор из grid поверх cfg["params"] оценивается тем же evaluator, возвращается лучший
    (для dry-run и живых прогонов, пока у P1 нет настоящего подбора на Optuna).
    """

    def __init__(self, result: Any = None, *, evaluator: Any = None, grid: list[dict] | tuple = (),
                 greater_is_better: bool = False) -> None:
        self.result, self.evaluator, self.grid, self.greater = result, evaluator, list(grid), greater_is_better
        self.calls: list[tuple[dict, float | None]] = []

    def __call__(self, spec: Any, build_features: Any, cfg: dict, deadline: Any = None) -> dict:
        self.calls.append((cfg, deadline.left() if deadline is not None else None))
        if isinstance(self.result, BaseException):
            raise self.result
        if self.evaluator is None:
            return dict(self.result or {})
        best, best_score = dict(cfg.get("params") or {}), None
        for params in self.grid:
            if deadline is not None and deadline.left() < 5:
                break
            trial = {**(cfg.get("params") or {}), **params}
            score = self.evaluator(spec, build_features, {**cfg, "params": trial}, deadline).cv_score
            if best_score is None or (score > best_score if self.greater else score < best_score):
                best, best_score = trial, score
        return best


LIGHTGBM_GRID = [{"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63},
                 {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50},
                 {"n_estimators": 400, "learning_rate": 0.08, "num_leaves": 127, "colsample_bytree": 0.7}]


def identity_features(train: pd.DataFrame, test: pd.DataFrame, spec: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train, test


def make_best(spec: SpecStub, evaluator: StubEvaluator) -> Candidate:
    """Аналог fallback-кандидата P3: модель на исходных признаках, без кода агента."""
    cand = evaluator(spec, identity_features, {"model": "lightgbm"}, None)
    cand.name = "fallback"
    return cand


# ---------------------------------------------------------------- синтетические задачи

def make_regression_task(root: Path, n_train: int = 400, n_test: int = 150, seed: int = 0) -> SpecStub:
    """y = 5·x1·x2 + 0.5·x3 + эффект города + шум: признак x1*x2 даёт крупный выигрыш CV."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({"id": np.arange(n) + 1000, "x1": rng.uniform(-1, 1, n), "x2": rng.uniform(-1, 1, n),
                       "x3": rng.normal(size=n), "city": rng.choice(["a", "b", "c"], n)})
    y = (5 * df.x1 * df.x2 + 0.5 * df.x3 + df.city.map({"a": 0.0, "b": 0.5, "c": -0.5})
         + rng.normal(scale=0.1, size=n))
    df.loc[rng.random(n) < 0.05, "x3"] = np.nan
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(target=y[:n_train]).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[n_train:], "target": 0.0}).to_csv(root / "sample_submission.csv", index=False)
    text = "# Synthetic regression\nPredict `target`. Metric: RMSE (lower is better)."
    (root / "task.md").write_text(text, encoding="utf-8")
    return SpecStub(task_dir=root, id_col="id", target="target", task_type="regression", metric="rmse",
                    greater_is_better=False, cat_cols=["city"], raw_task_text=text,
                    feature_notes={"x1": "первый фактор", "x2": "второй фактор", "x3": "шумовой признак",
                                   "city": "код города"})


def make_multiclass_task(root: Path, n_train: int = 450, n_test: int = 150, seed: int = 1) -> SpecStub:
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    df = pd.DataFrame({"id": np.arange(n), "a": rng.normal(size=n), "b": rng.normal(size=n),
                       "proto": rng.choice(["tcp", "udp", "icmp"], n)})
    score = df.a * df.b + rng.normal(scale=0.05, size=n)
    label = np.where(score > 0.4, "high", np.where(score < -0.4, "low", "mid"))
    root.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].assign(kind=label[:n_train]).to_csv(root / "train.csv", index=False)
    df.iloc[n_train:].to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": df.id.iloc[n_train:], "kind": "mid"}).to_csv(root / "sample_submission.csv", index=False)
    text = "# Synthetic multiclass\nPredict `kind`. Metric: Macro F1 (higher is better)."
    (root / "task.md").write_text(text, encoding="utf-8")
    return SpecStub(task_dir=root, id_col="id", target="kind", task_type="multiclass", metric="macro_f1",
                    greater_is_better=True, cat_cols=["proto"], raw_task_text=text)


# ---------------------------------------------------------------- ответы LLM для сценариев

def fenced(code: str) -> str:
    return f"Here is the code:\n```python\n{code}```\nDone."


def plan(*titles: str, kind: str = "features") -> dict:
    return {"analysis": "x1 and x2 interact; x3 is noise.",
            "hypotheses": [{"id": f"h{i}", "title": title, "kind": kind, "rationale": "test",
                            "expected_gain": "high", "cost": "low"} for i, title in enumerate(titles, 1)]}


def feature_code(body: str, header: str = "") -> str:
    """Модуль по шаблону агента; body — строки внутри цикла по (X_train, X_test) с переменной df."""
    lines = "\n".join("        " + line for line in body.strip().splitlines())
    return (f"import numpy as np\nimport pandas as pd\n{header}\n\n"
            "def build_features(train, test, spec):\n"
            "    X_train, X_test = train.copy(), test.copy()\n"
            "    for df in (X_train, X_test):\n"
            f"{lines}\n"
            "    return X_train, X_test\n\n\n"
            'MODEL = "lightgbm"\nPARAMS = {}\nTARGET_TRANSFORM = None\n')


GOOD_CODE = feature_code('df["x1_x2"] = df["x1"] * df["x2"]')
EXCEPTION_CODE = feature_code('df["oops"] = df["no_such_column"] * 2')
SLOW_CODE = feature_code("while True:\n    pass")
INF_CODE = feature_code('df["ratio"] = df["x1"] / (df["x2"] * 0.0)')
NO_FUNCTION_CODE = "import pandas as pd\n\n\ndef make_features(train, test):\n    return train, test\n"
DROP_ROWS_CODE = ("import pandas as pd\n\n\ndef build_features(train, test, spec):\n"
                  "    return train.iloc[:-5].copy(), test.copy()\n")


def useless_code(name: str) -> str:
    """Монотонное преобразование одного признака: деревьям ничего не даёт."""
    return feature_code(f'df["{name}"] = df["x3"] * 2.0 + 1.0')
