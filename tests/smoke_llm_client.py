"""Ручной smoke-тест: короткий запрос к LLM и проверка кэша.
Тратит немного бюджета, поэтому в автотесты не входит."""
import sys
from pathlib import Path

# добавляем корень проекта в путь, иначе из папки tests не найдётся llm_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient, LLMUnavailable


def main() -> int:
    client = LLMClient()
    question = [{"role": "user", "content": "Ответь одним словом: привет"}]
    try:
        # первый запрос идёт в API, второй такой же должен прийти из кэша
        first = client.chat(question, max_tokens=500)
        second = client.chat(question, max_tokens=500)
    except LLMUnavailable as e:
        print("LLM недоступна:", e)
        return 1
    print("Ответ LLM:", repr(first))
    print("Второй ответ такой же:", first == second)
    print("Расход:", client.usage_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())