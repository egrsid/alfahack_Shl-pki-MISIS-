"""Агент в раскладке ZIP: плоские модули, импорт без модулей команды и tests/, только разрешённые пакеты."""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_FILES = ("agent.py", "prompts_agent.py", "memory.py", "error_analysis.py")
TEAM_MODULES = {"schemas", "llm_client", "evaluator", "sandbox", "eda", "guards", "budget"}

# так solution.py (P1) вызывает агента: после fallback, в try/except, submission.csv пишет только сам
CALLER = '''
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import agent


@dataclass
class Spec:
    task_dir: Path
    id_col: str = "id"
    target: str = "y"
    task_type: str = "regression"
    metric: str = "rmse"
    greater_is_better: bool = False


class Deadline:
    def left(self):
        return 3600.0


class Llm:
    def chat(self, messages, **kw):
        raise AssertionError("LLM must not be called")


task_dir = Path.cwd()
pd.DataFrame({"id": range(50), "x": range(50), "y": range(50)}).to_csv(task_dir / "train.csv", index=False)
pd.DataFrame({"id": range(10), "x": range(10)}).to_csv(task_dir / "test.csv", index=False)
pd.DataFrame({"id": range(10), "y": 0.0}).to_csv(task_dir / "submission.csv", index=False)  # fallback
before = (task_dir / "submission.csv").read_bytes()

spec, mode = Spec(task_dir), sys.argv[1]
try:
    found = agent.research(spec, None, None if mode == "no_llm" else Llm(), Deadline())
except Exception as error:
    print("CALLER_CAUGHT", error)
    raise SystemExit(1)
assert found == [], found
assert (task_dir / "submission.csv").read_bytes() == before
print("OK", mode)
'''


def _top_level_imports(path: Path) -> set[str]:
    """Импорты вне try/except и вне функций — они обязаны быть доступны в контейнере."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_agent_modules_import_only_stdlib_numpy_pandas():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "prompts_agent", "memory", "error_analysis"}
    for name in AGENT_FILES:
        extra = _top_level_imports(ROOT / name) - allowed
        assert not extra, f"{name}: unconditional imports {extra}"


def test_no_secrets_or_local_paths_in_agent_modules():
    for name in AGENT_FILES:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "sk-" not in text and "/Users/" not in text and "OPENAI_API_KEY" not in text, name


def _run_flat(tmp_path: Path, mode: str, env_extra: dict) -> subprocess.CompletedProcess:
    for name in AGENT_FILES:
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "caller.py").write_text(CALLER, encoding="utf-8")
    env = {"PATH": os.defpath, **env_extra}
    return subprocess.run([sys.executable, "-s", "caller.py", mode], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=120)


def test_flat_layout_llm_disabled(tmp_path):
    result = _run_flat(tmp_path, "llm", {"LLM_DISABLED": "1"})
    assert result.returncode == 0 and "OK llm" in result.stdout, result.stderr[-2000:]
    assert not any((tmp_path / f"{m}.py").exists() for m in TEAM_MODULES)


def test_flat_layout_no_llm(tmp_path):
    result = _run_flat(tmp_path, "no_llm", {})
    assert result.returncode == 0 and "OK no_llm" in result.stdout, result.stderr[-2000:]


def test_flat_layout_without_evaluator_skips_research(tmp_path):
    """LLM есть, но evaluator.py (P1) ещё не в сборке → агент пропускает исследование, ничего не ломая."""
    result = _run_flat(tmp_path, "llm", {})
    assert result.returncode == 0 and "OK llm" in result.stdout, result.stderr[-2000:]
    assert "evaluator.evaluate" in result.stderr
