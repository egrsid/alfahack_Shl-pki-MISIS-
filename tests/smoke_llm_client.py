"""Ручной smoke-тест: один короткий запрос к LLM.
Тратит бюджет, поэтому в автотесты не входит."""
import sys
from pathlib import Path

# добавляем корень проекта в путь, иначе из папки tests не найдётся llm_client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient, LLMUnavailable


def main() -> int:
    client = LLMClient()
    try:
        answer = client.chat(
            [{"role": "user", "content": "Ответь одним словом: привет"}],
            # токенов берём с запасом: если модель сначала "думает", ответ может обрезаться
            max_tokens=500,
        )
    except LLMUnavailable as e:
        print("LLM недоступна:", e)
        return 1
    print("Ответ LLM:", repr(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())