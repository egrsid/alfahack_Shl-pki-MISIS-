"""AutoResearch-агент (P4): LLM-управляемый цикл улучшения лучшего кандидата.

research(spec, best, llm, deadline) -> list[Candidate]:
гипотеза (LLM) → код build_features (LLM) → песочница → оценка харнессом (evaluator) →
решение «принять/отклонить» по шуму CV → память → следующая итерация.

Агент никогда не пишет submission.csv и не роняет процесс: при любом сбое возвращает
уже принятых кандидатов (или []). Модули команды (schemas, llm_client, evaluator, sandbox,
eda, guards) подключаются через адаптеры ниже, поэтому agent.py импортируется и без них.
"""

from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

log = logging.getLogger("agent")

# --- контракты команды (schemas.py у P1, llm_client.py у P2); пока их нет — локальные заменители ---
try:
    from schemas import Candidate
except ImportError:
    @dataclass
    class Candidate:
        name: str
        cv_score: float
        cv_std: float
        oof: np.ndarray
        test_pred: np.ndarray  # в «сырых» единицах (вероятности для классификации)
        meta: dict = field(default_factory=dict)

try:
    from llm_client import LLMUnavailable
except ImportError:
    class LLMUnavailable(Exception):
        """LLM недоступна: сеть, исчерпаны ретраи, Budget has been exceeded."""


class LLMClient(Protocol):
    def chat(self, messages: list[dict], **kw: Any) -> str: ...
    def chat_json(self, messages: list[dict], schema: dict) -> dict: ...


class Deadline(Protocol):
    def left(self) -> float: ...


# evaluator.evaluate(spec, build_features, params, deadline) -> Candidate  (P1)
EvaluateFn = Callable[[Any, Callable, dict, Any], Candidate]
# sandbox.run_code(code, cwd, timeout) -> ExecResult(stdout, stderr, returncode, artifacts)  (P5)
RunCodeFn = Callable[[str, Path, float], Any]
# guards.check_candidate(spec, cand, baseline) -> list[str], пусто = ок  (P5)
GuardFn = Callable[[Any, Candidate, Candidate], list]


@dataclass
class AgentConfig:
    """Все числа агента в одном месте; времена — верхние границы, реально режет Deadline."""

    # итерации и стоп-критерии
    max_iters: int = 12
    hypotheses_per_plan: int = 6
    max_empty_plans: int = 2            # подряд планов без новых гипотез → стоп
    max_consecutive_failures: int = 3
    max_no_improve: int = 5             # плато
    max_tokens: int = 120_000           # доля агента в токенах на задачу (вход + выход)
    min_start_seconds: float = 300.0    # если до конца окна research меньше — не начинаем
    llm_reserve_seconds: float = 60.0   # оценка времени на вызовы LLM за одну итерацию
    # песочница и оценка
    features_timeout: float = 300.0
    evaluate_timeout: float = 900.0
    max_feature_cols: int = 2000
    max_nan_share: float = 0.5          # «NaN-лавина»: доля пропусков выше max(этого, входной + 0.25)
    max_repairs: int = 1                # попыток ремонта по traceback / ошибке проверки
    max_format_fixes: int = 1           # попыток «пришли один блок ```python»
    # правило принятия
    accept_k: float = 1.0               # gain > accept_k · SE шума CV
    min_rel_gain: float = 5e-4          # и gain > min_rel_gain · |cv лучшего|
    default_n_folds: int = 5            # если evaluator не отдал fold_scores
    suspicious_k: float = 8.0           # «чудесный» скачок: gain > suspicious_k · SE ...
    suspicious_rel: float = 0.1         # ... и > 10% от |cv| → перепроверка guards
    only_better_than_best: bool = True  # вернуть только кандидатов не хуже входного best
    # LLM
    plan_max_tokens: int = 1200
    plan_temperature: float = 0.7
    code_max_tokens: int = 2000
    code_temperature: float = 0.2
    critic_max_tokens: int = 300
    use_critic: bool = True
    no_think: bool = True               # дописывать /no_think (Qwen3: без длинного <think>)
    # размеры кусков промпта, символы
    profile_chars: int = 6000
    task_text_chars: int = 1500
    memory_chars: int = 1500
    error_chars: int = 1500
    # файлы
    workdir_name: str = ".agent_work"
    keep_artifacts: bool = False


def _optional(module: str, attr: str) -> Any:
    """Функция модуля команды, если он уже смержен в main; иначе None."""
    try:
        return getattr(importlib.import_module(module), attr, None)
    except ImportError:
        return None


# ============================================================ парсинг ответов LLM

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)
_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)


def strip_think(text: str | None) -> str:
    """Убрать рассуждения Qwen. Незакрытый <think> (ответ обрезан) — убрать только тег."""
    return _THINK_TAG.sub("", _THINK_BLOCK.sub("", text or "")).strip()


def extract_code(text: str | None) -> str | None:
    """Код из ответа кодера: первый блок ```python с build_features, иначе первый python-блок.

    Без ограждений — текст от первого import/def, если в нём есть build_features.
    Незакрытый блок (ответ обрезан по max_tokens) берётся до конца: дальше решит compile().
    """
    text = strip_think(text)
    blocks = [body for lang, body in _FENCE.findall(text) if lang.lower() in ("python", "py", "python3", "")]
    for body in blocks:
        if "def build_features" in body:
            return body.strip() + "\n"
    if blocks and blocks[0].strip():
        return blocks[0].strip() + "\n"
    if "def build_features" in text:
        start = re.search(r"^(import |from |def )", text, re.M)
        return text[start.start():].strip() + "\n" if start else None
    return None


def _clean_json(text: str) -> str:
    """Убрать //, /* */ и # комментарии вне строк и висячие запятые."""
    out, i, in_str = [], 0, False
    while i < len(text):
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif text.startswith("//", i) or ch == "#":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        else:
            out.append(ch)
        i += 1
    return re.sub(r",\s*([}\]])", r"\1", "".join(out))


def _balanced(text: str, start: int) -> str | None:
    """Подстрока от открывающей скобки до парной (с учётом строк в обеих кавычках)."""
    depth, quote, i = 0, "", start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    return None


def _is_payload(value: Any) -> bool:
    """Ответ — объект или список объектов (а не случайный «[1]» из текста вокруг)."""
    return isinstance(value, dict) or (isinstance(value, list) and any(isinstance(v, dict) for v in value))


def _first_json(text: str) -> Any:
    decoder = json.JSONDecoder()
    starts = [i for i, ch in enumerate(text) if ch in "{["][:50]
    for i in starts:
        for variant in (text[i:], _clean_json(text[i:])):
            try:
                value = decoder.raw_decode(variant)[0]
                if _is_payload(value):
                    return value
            except json.JSONDecodeError:
                pass
        chunk = _balanced(text, i)
        if chunk:  # одинарные кавычки, True/False/None — как питоновский литерал
            pythonish = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", chunk)))
            try:
                value = ast.literal_eval(_clean_json(pythonish))
                if _is_payload(value):
                    return value
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                pass
    return None


def extract_json(text: str | None) -> Any:
    """Первый JSON-объект/массив в ответе: сначала из ```json блоков, потом из всего текста."""
    text = strip_think(text)
    fenced = [body for lang, body in _FENCE.findall(text) if lang.lower() in ("json", "")]
    for chunk in [*fenced, text]:
        value = _first_json(chunk)
        if value is not None:
            return value
    raise ValueError("no JSON object found in the answer")


# ============================================================ статическая проверка кода

ALLOWED_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "math", "re", "itertools", "functools", "collections",
                   "datetime", "statistics", "string", "typing", "warnings", "operator", "calendar",
                   "unicodedata", "heapq", "bisect", "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "breakpoint", "input", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_hdf", "to_sql", "to_feather",
                   "to_stata", "to_orc", "to_clipboard", "tofile", "fromfile", "save", "savez", "savez_compressed",
                   "savetxt", "load", "loadtxt", "genfromtxt", "fromregex", "memmap", "system", "popen", "remove",
                   "unlink", "rmtree", "chdir"}
_TOP_LEVEL_OK = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign)


def validate_code(code: str) -> list[str]:
    """Проблемы кода кодера без его исполнения; пустой список — можно запускать в песочнице."""
    try:
        tree = ast.parse(code)
        compile(tree, "<features>", "exec")
    except SyntaxError as error:
        return [f"SyntaxError: {error.msg} (line {error.lineno})"]
    except (ValueError, MemoryError, RecursionError) as error:
        return [f"code does not compile: {error}"]

    problems: list[str] = []
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_features"), None)
    if func is None:
        problems.append("no top-level function `build_features(train, test, spec)`")
    elif len(func.args.posonlyargs) + len(func.args.args) < 3 and func.args.vararg is None:
        problems.append("`build_features` must accept three arguments: (train, test, spec)")

    for node in tree.body:
        docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not isinstance(node, _TOP_LEVEL_OK) and not docstring:
            problems.append(f"top-level `{type(node).__name__}` statement (line {node.lineno}) is not allowed: "
                            "only imports, functions and constants at module level")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.level:
                problems.append("relative imports are not allowed")
                continue
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of `{name}` is not allowed (allowed: numpy, pandas, scipy, sklearn, "
                                    "math, re, collections, itertools, ...)")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            problems.append(f"use of `{node.id}` is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                problems.append(f"access to dunder attribute `{node.attr}` is not allowed")
            elif node.attr in FORBIDDEN_ATTRS or node.attr.startswith("read_"):
                problems.append(f"`.{node.attr}` is not allowed: no file, network or OS access")
    return list(dict.fromkeys(problems))


# белый список гиперпараметров: (min, max) — число в границах; set — одно из значений
MODELS = ("lightgbm", "catboost", "xgboost")
MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "lightgbm": {"n_estimators": (10, 5000), "learning_rate": (1e-3, 0.5), "num_leaves": (2, 1024),
                 "max_depth": (-1, 16), "min_child_samples": (1, 10_000), "min_child_weight": (0.0, 1e3),
                 "subsample": (0.1, 1.0), "subsample_freq": (0, 10), "colsample_bytree": (0.1, 1.0),
                 "reg_alpha": (0.0, 1e3), "reg_lambda": (0.0, 1e3), "min_split_gain": (0.0, 10.0),
                 "max_bin": (15, 1023), "cat_smooth": (0.0, 1e3), "cat_l2": (0.0, 1e3),
                 "max_cat_threshold": (1, 1024), "min_data_per_group": (1, 10_000), "path_smooth": (0.0, 1e3),
                 "class_weight": {"balanced", None}, "extra_trees": {True, False}},
    "catboost": {"iterations": (10, 5000), "learning_rate": (1e-3, 0.5), "depth": (1, 10),
                 "l2_leaf_reg": (0.0, 1e3), "random_strength": (0.0, 100.0), "bagging_temperature": (0.0, 100.0),
                 "border_count": (16, 1024), "one_hot_max_size": (1, 255), "min_data_in_leaf": (1, 10_000),
                 "auto_class_weights": {"Balanced", "SqrtBalanced", None}},
    "xgboost": {"n_estimators": (10, 5000), "learning_rate": (1e-3, 0.5), "max_depth": (1, 16),
                "min_child_weight": (0.0, 1e3), "subsample": (0.1, 1.0), "colsample_bytree": (0.1, 1.0),
                "reg_alpha": (0.0, 1e3), "reg_lambda": (0.0, 1e3), "gamma": (0.0, 100.0), "max_bin": (16, 1024),
                "max_cat_to_onehot": (1, 255)},
}
CONSTANTS = ("MODEL", "PARAMS", "TARGET_TRANSFORM")


def extract_constants(code: str) -> tuple[dict, list[str]]:
    """MODEL / PARAMS / TARGET_TRANSFORM из кода статически (ast.literal_eval), без исполнения."""
    found: dict = {}
    problems: list[str] = []
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in CONSTANTS:
                try:
                    found[target.id] = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError):
                    problems.append(f"{target.id} must be a literal value (no expressions or calls)")
    return found, problems


def normalize_model_config(consts: dict, task_type: str) -> tuple[str, dict, str | None, list[str]]:
    """Привести MODEL/PARAMS/TARGET_TRANSFORM к белому списку; что выкинуто — в notes."""
    notes: list[str] = []
    model = str(consts.get("MODEL") or "lightgbm").lower().strip()
    if model not in MODELS:
        notes.append(f"MODEL={model!r} is not supported, using lightgbm")
        model = "lightgbm"

    raw = consts.get("PARAMS") or {}
    if not isinstance(raw, dict):
        notes.append("PARAMS must be a dict, ignored")
        raw = {}
    allowed, params = MODEL_PARAMS[model], {}
    for key, value in raw.items():
        rule = allowed.get(key)
        if rule is None:
            notes.append(f"PARAMS[{key!r}] is not allowed for {model}, ignored")
        elif isinstance(rule, set):
            if value in rule:
                params[key] = value
            else:
                notes.append(f"PARAMS[{key!r}]={value!r} must be one of {sorted(rule, key=str)}, ignored")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            notes.append(f"PARAMS[{key!r}] must be a number, ignored")
        else:
            low, high = rule
            clipped = min(max(value, low), high)
            if clipped != value:
                notes.append(f"PARAMS[{key!r}]={value} clipped to {clipped}")
            params[key] = int(round(clipped)) if isinstance(low, int) and isinstance(high, int) else float(clipped)

    transform = consts.get("TARGET_TRANSFORM")
    if transform is not None and (transform != "log1p" or task_type != "regression"):
        notes.append(f"TARGET_TRANSFORM={transform!r} is not supported for {task_type}, using None")
        transform = None
    return model, params, transform, notes


def research(spec: Any, best: Candidate | None, llm: LLMClient | None, deadline: Any, *,
             config: AgentConfig | None = None,
             evaluate: EvaluateFn | None = None,
             run_code: RunCodeFn | None = None,
             profile: Callable | None = None,
             guards: GuardFn | None = None,
             data: Any = None) -> list[Candidate]:
    """Принятые кандидаты по убыванию качества CV (первый — лучший) или [].

    Необязательные аргументы — для тестов и для P1: подменить evaluator/песочницу/профиль
    и передать уже загруженные данные. По умолчанию берутся модули команды, если они есть.
    """
    if llm is None:
        log.info("research: LLM нет — исследование пропущено")
        return []
    if os.environ.get("LLM_DISABLED") == "1" and not getattr(llm, "offline", False):
        log.info("research: LLM_DISABLED=1 — исследование пропущено")
        return []
    return []
