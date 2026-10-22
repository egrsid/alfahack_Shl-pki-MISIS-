# План ветки `feature/metrics-submission` (P3, ML-ядро)

## 1. Цель и границы

**Цель:** два модуля, на которых стоят fallback (P3), evaluator (P1) и агент (P4):
- `metrics.py` — реестр метрик и единый `score(name, y_true, y_pred)`; именно им считается CV везде.
- `submission.py` — сверка с `sample_submission.csv` и запись. **Единственная точка записи `submission.csv`.**

**Не входит:** `fallback.py`, `preprocess.py` (ветка `feature/fallback-pipeline`), понимание задачи (P2), харнесс оценки (P1).

## 2. Контракты

```python
metrics.score(name, y_true, y_pred, decision=None) -> float
metrics.get(name) -> Metric(name, func, greater_is_better, task, needs_proba, tunable)
metrics.resolve(name) -> str            # «Macro F1», «f1-macro», «ROC AUC» → канонический ключ; иначе UnknownMetric
metrics.find_in_text(text) -> str|None  # метрика из task.md (для эвристик P2)
metrics.classes_of(y) -> list[str]      # порядок классов: числовые — по значению, иначе по строке (как np.unique)
metrics.fit_decision / crossfit_decision / to_labels   # порог или веса классов под F1-подобные метрики

submission.write(spec, pred, *, classes=None, decision=None) -> Path   # валидирует и пишет атомарно
submission.write_candidate(spec, cand) -> Path                         # берёт classes/decision из cand.meta
submission.write_dummy(spec) -> Path|None                              # константный сабмит-страховка
submission.validate(spec, frame) -> list[str]                          # пусто = ок
```

Соглашения о форме (совпадают с заглушками P4 в `tests/mocks.py`):
- регрессия — `y_pred` 1-D;
- классификация — `y_true` коды `0..k-1` в порядке `classes_of`, `y_pred` — вероятности `(n, k)`;
  бинарная: допускается 1-D вероятность класса с кодом 1 (положительный — последний класс, как `pos_label` в sklearn);
- метрики по меткам из вероятностей берут `argmax`; `decision` (порог / веса классов) — отдельный пост-процессинг,
  `Candidate.cv_score` считается без него, чтобы кандидаты fallback и агента сравнивались одинаково.

## 3. Шаги
1. `metrics.py`: регрессия (rmse, mse, mae, medae, rmsle, msle, mape, smape, wape, r2), классификация по меткам
   (accuracy, balanced_accuracy, macro/weighted/micro F1, f1, precision, recall, mcc, kappa, qwk) и по вероятностям
   (roc_auc, gini, pr_auc, logloss, brier); синонимы; поиск метрики в тексте; подбор порога/весов классов.
2. `submission.py`: формат строго как в `sample_submission.csv` (колонки, порядок строк по id, число строк),
   метки в исходном виде (int/str/bool как в train), вероятности — если метрика их требует или в образце
   колонки по классам; ремонт NaN/inf с предупреждением; атомарная запись (tmp + `os.replace`).
3. Тесты `tests/test_metrics.py`, `tests/test_submission.py` (вне ZIP).

## 4. Критерии готовности
- `score` совпадает с `sklearn` на всех метриках; неизвестное имя → `UnknownMetric`, P2/fallback откатываются на `default_for`.
- `write` никогда не оставляет битый файл; на любом расхождении с образцом — `SubmissionError` с понятным текстом.
- Работает на регрессии, бинарной (метки и вероятности), мультиклассе со строковыми метками, без id в test.

## 5. Риски
- Положительный класс в бинарной для AUC: берём последний в каноническом порядке (как sklearn) — для строковых
  меток это может не совпасть с замыслом организаторов; P2 может переопределить через spec.
- `sample_submission.csv` с колонками-вероятностями по классам — поддержано по совпадению имён колонок с классами.

## Статус выполнения
_(заполняется по ходу работы)_
