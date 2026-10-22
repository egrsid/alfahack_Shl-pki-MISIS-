"""AutoResearch-агент (P4): LLM-управляемый цикл улучшения лучшего кандидата.

research(spec, best, llm, deadline) -> list[Candidate]:
гипотеза (LLM) → код build_features (LLM) → песочница → оценка харнессом (evaluator) →
решение «принять/отклонить» по шуму CV → память → следующая итерация.

Агент никогда не пишет submission.csv и не роняет процесс: при любом сбое возвращает
уже принятых кандидатов (или []). Модули команды (schemas, llm_client, evaluator, sandbox,
eda, guards) подключаются через адаптеры ниже, поэтому agent.py импортируется и без них.
"""

from __future__ import annotations

import importlib
import logging
import os
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
