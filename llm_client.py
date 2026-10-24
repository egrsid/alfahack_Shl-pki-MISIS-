"""Клиент для LLM. Ключ и адрес берём только из переменных окружения."""
from __future__ import annotations

import os

from openai import OpenAI

# модель по умолчанию, на платформе её можно переопределить через OPENAI_MODEL
DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LLMUnavailable(Exception):
    # кидаем её, если с LLM что-то не так: нет ключа, кончился бюджет, упала сеть.
    # тот, кто вызывает клиента, должен её поймать и уйти на запасной вариант без LLM
    pass


def _load_dotenv(path: str = ".env") -> None:
    # читаем .env руками, чтобы локально не экспортить переменные каждый раз.
    # на платформе файла нет, там всё уже лежит в окружении
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # пропускаем пустые строки, комментарии и мусор без "="
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # setdefault, чтобы не затереть переменную, которая уже задана снаружи
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMClient:
    def __init__(self, model: str | None = None, timeout: float = 120.0):
        _load_dotenv()
        self.model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        # сам клиент создаём не сразу, а при первом запросе,
        # чтобы без ключа ничего не падало при простом создании объекта
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMUnavailable("OPENAI_API_KEY не задан")
            # OpenAI() сам читает OPENAI_API_KEY и OPENAI_BASE_URL из окружения.
            # max_retries=0, потому что повторы запросов напишем сами (шаг 2)
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Один запрос к модели, возвращает текст ответа."""
        client = self._get_client()
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )
        except Exception as e:
            raise LLMUnavailable(f"Ошибка запроса к LLM: {e}") from e
        return resp.choices[0].message.content or ""