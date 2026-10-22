import numpy as np
import pytest

import agent
from mocks import MockLLM, SimpleDeadline, StubEvaluator, make_best, make_multiclass_task, make_regression_task


@pytest.fixture
def reg_task(tmp_path):
    return make_regression_task(tmp_path / "reg")


def test_llm_none_returns_empty(reg_task):
    evaluator = StubEvaluator(n_splits=3, n_estimators=50)
    assert agent.research(reg_task, None, None, SimpleDeadline(600), evaluate=evaluator) == []
    assert evaluator.calls == 0


def test_llm_disabled_env_skips_real_client(reg_task, monkeypatch):
    class RealClient(MockLLM):
        offline = False

    monkeypatch.setenv("LLM_DISABLED", "1")
    llm = RealClient()
    assert agent.research(reg_task, None, llm, SimpleDeadline(600)) == []
    assert llm.calls == []


def test_stub_evaluator_is_deterministic(reg_task, tmp_path):
    evaluator = StubEvaluator(n_splits=3, n_estimators=50)
    first, second = make_best(reg_task, evaluator), make_best(reg_task, evaluator)
    assert first.cv_score == second.cv_score and len(first.meta["fold_scores"]) == 3
    assert first.oof.shape == (400,) and np.isfinite(first.test_pred).all()

    cls = make_best(make_multiclass_task(tmp_path / "cls"), evaluator)
    assert 0 < cls.cv_score <= 1 and cls.oof.shape == (450, 3)
