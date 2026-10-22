"""Память экспериментов агента (минимальная версия).

Журнал: что пробовали, что приняли/отклонили и почему, что падало. Отдаёт короткую сводку
для промпта и пишет JSONL для отладки. Сжатие и ранжирование «что ещё не пробовали» —
ветка feature/experiment-memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("accepted", "rejected", "failed")


def code_hash(code: str) -> str:
    """Хеш кода без учёта пустых строк и хвостовых пробелов — для поиска повторов."""
    lines = [line.rstrip() for line in code.strip().splitlines() if line.strip()]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def normalize_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", title.lower()).strip()


@dataclass
class Experiment:
    id: int
    hypothesis: str
    kind: str
    status: str  # accepted | rejected | failed
    cv_score: float | None = None
    cv_std: float | None = None
    gain: float | None = None
    code_hash: str = ""
    error: str = ""
    duration_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    ts: float = field(default_factory=time.time)


def _num(value: float | None) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.5g}"


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format(exp: Experiment) -> str:
    head = f"#{exp.id} [{exp.status}] {exp.kind} | {_one_line(exp.hypothesis, 80)}"
    if exp.status == "failed":
        return f"{head} | error: {_one_line(exp.error, 140)}"
    gain = "" if exp.gain is None else f" (gain {exp.gain:+.4g})"
    reason = f" | {_one_line(exp.error, 80)}" if exp.error else ""
    return f"{head} | cv={_num(exp.cv_score)}{gain}{reason}"


class ExperimentMemory:
    def __init__(self) -> None:
        self.experiments: list[Experiment] = []
        self.notes: list[str] = []

    def next_id(self) -> int:
        return len(self.experiments) + 1

    def add(self, exp: Experiment) -> Experiment:
        if exp.status not in STATUSES:
            raise ValueError(f"unknown status: {exp.status}")
        self.experiments.append(exp)
        return exp

    def add_note(self, text: str, keep: int = 3) -> None:
        """Заметка критика; хранятся только последние keep."""
        text = _one_line(text, 600)
        if text:
            self.notes.append(text)
            del self.notes[:-keep]

    def tried_titles(self) -> list[str]:
        return [exp.hypothesis for exp in self.experiments]

    def tried_kinds(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for exp in self.experiments:
            counts[exp.kind] = counts.get(exp.kind, 0) + 1
        return counts

    def find_code(self, digest: str) -> Experiment | None:
        return next((exp for exp in self.experiments if exp.code_hash == digest), None)

    def summary(self, max_chars: int = 1500, recent: int = 8) -> str:
        """Сводка для промпта: последние эксперименты подробно, старые — одной строкой, заметки критика."""
        if not self.experiments and not self.notes:
            return "No experiments yet."
        recent_lines = [_format(exp) for exp in self.experiments[-recent:]]
        older = self.experiments[:-recent] if len(self.experiments) > recent else []
        earlier = ("Earlier: " + "; ".join(f"#{e.id} {e.status} {_one_line(e.hypothesis, 40)}" for e in older)
                   if older else "")
        notes = ["Critic notes:"] + [f"- {note}" for note in self.notes] if self.notes else []

        def render() -> str:
            lines = [earlier] if earlier else []
            if recent_lines:
                lines += ["Experiments (oldest first):", *recent_lines]
            return "\n".join(lines + notes)

        text = render()
        while len(text) > max_chars and (earlier or len(recent_lines) > 1):
            if earlier:
                earlier = ""
            else:
                recent_lines.pop(0)
            text = render()
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    def dump_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for exp in self.experiments:
                row = {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
                       for k, v in asdict(exp).items()}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path
