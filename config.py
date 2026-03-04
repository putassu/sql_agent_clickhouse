a = 3
import logging

# Настройки LLM (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# Параметры логики
LOG_LEVEL = logging.INFO
CONFIDENCE_THRESHOLD = 0.7
NUM_RETRIES = 3
MAX_HUMAN_RETRIES = 3

# Версия схемы (для промпта)
DB_SCHEMA_VERSION = """
Table: FINAL_KPI_DATA
Columns: 
  - CALDAY (Date)
  - CALMONTH (String, format YYYYMM)
  - KPI_ID (String) - ID показателя
  - SEGMENT_ID (String) - ID компании/завода
  - KF_TYPE (String) - 'ACT' (Факт), 'CUMM' (Накопленный итог)
  - VALUE (Float) - Значение
"""
