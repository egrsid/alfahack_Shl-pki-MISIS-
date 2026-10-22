"""ЗАГЛУШКА (P4 для P1). Владелец модуля — P1. В main не мержить: это предложение контракта.

Агент (agent.py, вид гипотезы «tune») вызывает:

    tuning.tune(spec, build_features, cfg, deadline) -> dict

    spec           — TaskSpec задачи;
    build_features — тот же callable, что агент передаёт в evaluator.evaluate: отдаёт готовые признаки
                     текущего лучшего кандидата (код LLM уже исполнен в песочнице);
    cfg            — {"model": "lightgbm|catboost|xgboost", "params": {...стартовые...},
                      "target_transform": None|"log1p", "n_trials": int, "seed": int};
    deadline       — объект с .left() (секунды): подбор обязан уложиться и вернуть лучшее найденное;
    результат      — словарь гиперпараметров модели (подмножество белого списка agent.MODEL_PARAMS).

Ожидания агента к реальной реализации (P1):
- те же сплиты, сид и метрика, что у evaluator.evaluate (иначе найденное не переносится);
- при нехватке времени — вернуть лучшее на момент остановки, а не бросать исключение;
- оценку итоговых параметров агент делает сам через evaluator.evaluate.

Эта заглушка ничего не подбирает: возвращает стартовые параметры, поэтому агент такой «подбор» отклонит
как не давший выигрыша. Это безопасное поведение до появления реального модуля.
"""

from __future__ import annotations

from typing import Any, Callable

STUB = True  # реальный модуль P1 этот флаг не содержит


def tune(spec: Any, build_features: Callable, cfg: dict, deadline: Any) -> dict:
    return dict(cfg.get("params") or {})
