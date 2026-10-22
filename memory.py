"""Память экспериментов агента.

Журнал: что пробовали, что приняли/отклонили и почему, что падало, что изменилось в коде.
Отдаёт сжатую сводку для промпта (принятые шаги не выкидываются никогда), ловит перефразированные
дубли гипотез, показывает «что ещё не пробовали», пишет и читает JSONL.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable

STATUSES = ("accepted", "rejected", "failed")


def code_hash(code: str) -> str:
    """Хеш кода без учёта пустых строк и хвостовых пробелов — для поиска повторов."""
    lines = [line.rstrip() for line in code.strip().splitlines() if line.strip()]
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def normalize_title(title: str) -> str:
    return re.sub(r"[\W_]+", " ", title.lower()).strip()


def _code_lines(code: str) -> list[str]:
    return [line.rstrip() for line in (code or "").splitlines() if line.strip()]


def code_diff(old: str, new: str, max_lines: int = 15) -> str:
    """Только изменённые строки (+/-) без контекста; max_lines=0 — без обрезки."""
    changed = [line for line in difflib.unified_diff(_code_lines(old), _code_lines(new), lineterm="", n=0)
               if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if max_lines and len(changed) > max_lines:
        changed = changed[:max_lines] + [f"... (+{len(changed) - max_lines} more changed lines)"]
    return "\n".join(changed)


_STRING_LITERAL = re.compile(r"""(['"])((?:\\.|(?!\1).)*)\1""")


def columns_in(code: str, known: Iterable[str]) -> list[str]:
    """Известные колонки, упомянутые в коде строковыми литералами (df["col"]), в порядке появления."""
    known = set(known)
    found = (match.group(2) for match in _STRING_LITERAL.finditer(code or ""))
    return list(dict.fromkeys(name for name in found if name in known))


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
    columns: list[str] = field(default_factory=list)  # исходные колонки, которых касается изменение
    diff: str = ""                                    # сжатый дифф кода относительно базы
    base_hash: str = ""                               # code_hash базы, от которой сделан эксперимент


_FIELDS = {f.name for f in fields(Experiment)}


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
    reason = f" | {_one_line(exp.error, 160)}" if exp.error else ""
    return f"{head} | cv={_num(exp.cv_score)}{gain}{reason}"


class ExperimentMemory:
    def __init__(self, columns: Iterable[str] = ()) -> None:
        self.experiments: list[Experiment] = []
        self.notes: list[str] = []
        self.columns: list[str] = [str(c) for c in columns]  # исходные колонки задачи — для дублей и покрытия

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

    @classmethod
    def load_jsonl(cls, path: str | Path, columns: Iterable[str] = ()) -> "ExperimentMemory":
        """Обратно к dump_jsonl; старый формат (без новых полей) и битые строки не мешают."""
        memory = cls(columns)
        path = Path(path)
        if not path.exists():
            return memory
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                exp = Experiment(**{k: v for k, v in row.items() if k in _FIELDS})
                memory.add(exp)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return memory
