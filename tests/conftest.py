import os
import sys
from pathlib import Path

# модули решения лежат плоско в корне репозитория
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# в тестах реальный Qwen не вызывается никогда; MockLLM помечен offline и этим флагом не отключается
os.environ.setdefault("LLM_DISABLED", "1")
