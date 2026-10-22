# План ветки `feature/fallback-pipeline` (P3, ML-ядро)

> Ветка создана от `feature/metrics-submission` (нужны `metrics.py` и `submission.py`); мержится в `main` после неё.

## 1. Цель и границы

**Цель:** `fallback.run(spec, deadline) -> Candidate` — детерминированный пайплайн без LLM, который на любой
табличной задаче за ~15 мин (8 CPU / 32 ГБ) даёт честный CV и пишет `submission.csv` СРАЗУ после первой модели,
перезаписывая его только при улучшении CV. Это страховка от падения LLM, агента и от лимита времени,
и одновременно стартовый `best` для агента P4.

**Входит:** `preprocess.py` (авто-предобработка), `fallback.py` (CV, модели, бленд, запись), тесты.
**Не входит:** `solution.py` (P1 — интеграция), понимание задачи (P2), харнесс `evaluator.py` (P1; когда смержится,
fallback переходит на его сплиты).

## 2. Контракты

```python
fallback.run(spec: TaskSpec, deadline: Deadline | None = None, config: FallbackConfig | None = None) -> Candidate
fallback.guess_spec(task_dir) -> TaskSpec-подобный объект   # эвристики на случай, если understand_task упал
preprocess.prepare(train_x, test_x, spec) -> Prepared(X, X_test, cat_cols, high_card, report)
preprocess.model_frame(prep, X, model) -> (frame, cat_features)    # адаптер под lightgbm / xgboost / catboost
```

`Candidate` (как в `schemas.py` из плана P1): `oof`/`test_pred` в «сырых» единицах (вероятности `(n, k)` по
`meta["classes"]`), `cv_score`/`cv_std` — без пост-обработки (argmax), чтобы сравнение с кандидатами агента было честным.
`meta`: `fold_scores`, `folds` (id фолда для каждой строки train), `model`, `params`, `target_transform`, `n_splits`,
`seed`, `classes`, `decision` (порог/веса классов), `cv_score_post` (CV с decision, кросс-оценка), `members`
(для бленда), `preprocess` (отчёт), `seconds`.

## 3. Шаги
1. `preprocess.py`: типы колонок (число, категория, дата, текст, числа-строками), удаление константных,
   id-подобных, дублирующих колонок и колонок, пустых в test; категории — редкие → `__rare__`, частотное кодирование,
   высокая кардинальность; даты → год/месяц/день/день недели/час/timestamp; текст → длины + TF-IDF→SVD;
   подсказки из `spec.cat_cols/date_cols/text_cols/drop_cols`.
2. `fallback.py`: K-fold (stratified для классификации, сид 42, как у заглушки evaluator P4), LightGBM → CatBoost →
   XGBoost с ранней остановкой и лимитом времени на фолд (колбэки), лог-таргет для регрессии по CV,
   `class_weight=balanced`-вариант для F1-подобных метрик при дисбалансе, жадный бленд (Caruana) по OOF,
   порог/веса классов (`metrics.crossfit_decision`), запись через `submission.write` только при улучшении.
3. Устойчивость: константный сабмит в первые секунды; сбой одной модели не роняет остальные; если не обучилась ни одна —
   кандидат-константа; никакой сети/LLM.
4. Тесты на синтетике (регрессия, бинарная, мультикласс со строковыми метками, пропуски, даты, текст) и замер времени
   на прокси размера public-задач.

## 4. Критерии готовности
- Работает на обеих public-задачах и на синтетике другого типа; укладывается в бюджет (по умолчанию 900 с).
- `submission.csv` валиден после первой модели; при падении любого этапа остаётся лучший записанный.
- Заметно лучше `solution.py` v0 (LightGBM без CV) по CV.

## 5. Риски
- Нет локально данных public-задач → прокси-датасеты того же размера; финальная проверка — когда P1 соберёт `solution.py`.
- CatBoost на мультиклассе медленный → оценка времени по LightGBM, пропуск при нехватке, лимит на фолд.
- Порог/веса классов переобучаются на OOF → берутся только если выигрыш подтверждён на других фолдах.

## Статус выполнения
_(заполняется по ходу работы)_
