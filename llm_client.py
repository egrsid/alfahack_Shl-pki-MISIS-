"""Клиент для LLM. Ключ и адрес берём только из переменных окружения."""
from __future__ import annotations

import os
import time

import openai
from openai import OpenAI

# модель по умолчанию, на платформе её можно переопределить через OPENAI_MODEL
DEFAULT_MODEL = "qwen/qwen3.8-27b"

MAX_RETRIES = 3    # сколько всего попыток делаем, если ошибка временная
BASE_DELAY = 2.0   # пауза перед повтором растёт: 2, 4, 8 секунд


class LLMUnavailable(Exception):
    # кидаем её, если с LLM что-то не так: нет ключа, кончился бюджет, упала сеть.
    # тот, кто вызывает клиента, должен её поймать и уйти на запасной вариант без LLM
    pass


def _load_dotenv(path: str = ".env") -> None:
    # читаем .env руками, чтобы локально не exportить переменные каждый раз.
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


def _is_budget_error(e: Exception) -> bool:
    # прокси пишет "Budget has been exceeded", поэтому смотрим по тексту ошибки,
    # а не по коду: код у такой ошибки может быть разным
    return "budget" in str(e).lower()


def _is_retryable(e: Exception) -> bool:
    # временные проблемы, которые имеет смысл повторить:
    # нет связи или таймаут (таймаут входит в APIConnectionError), 429, 5xx
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code == 429 or e.status_code >= 500
    return False


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
            # max_retries=0, потому что повторы теперь делаем сами в chat()
            self._client = OpenAI(timeout=self.timeout, max_retries=0)
        return self._client

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Один запрос к модели, возвращает текст ответа."""
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
                # content может быть None, поэтому подстраховываемся пустой строкой
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                # бюджет кончился: повторять бессмысленно, сразу сдаёмся
                if _is_budget_error(e):
                    raise LLMUnavailable(f"Бюджет LLM исчерпан: {e}") from e
                # ошибка не временная (например, неверный ключ): тоже сдаёмся сразу
                if not _is_retryable(e):
                    raise LLMUnavailable(f"Ошибка запроса к LLM: {e}") from e
                # это была последняя попытка, ждать уже незачем
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(BASE_DELAY * 2 ** attempt)

        raise LLMUnavailable(
            f"LLM не ответила после {MAX_RETRIES} попыток: {last_error}"
        ) from last_error