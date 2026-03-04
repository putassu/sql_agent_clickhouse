import requests
import json
import pandas as pd
import logging
from config import OLLAMA_URL, MODEL_NAME
from string import Template

logger = logging.getLogger("Utils")

def create_prompt(input_values,PROMPT):
    t = Template(PROMPT)
    return t.substitute(**input_values)

def call_llm(prompt: str, input_text: str, is_json: bool = False):
    """Универсальный синхронный вызов Ollama."""
    # full_prompt = prompt.format(input_text=input_text, task=input_text)
    full_prompt = create_prompt({"input_text":input_text},prompt)
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "format": "json" if is_json else ""
    }
    
    try:
        # Используем requests.post вместо aiohttp
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status() # Вызывает исключение при ошибках HTTP (4xx, 5xx)
        
        result = response.json()
        text = result['response'].strip()
        
        if is_json:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Очистка текста от лишних символов, если модель ошиблась в формате
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end != 0:
                    return json.loads(text[start:end])
                raise
        return text
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise

def call_deepseek_v3(prompt, input_text, response_model):
    """Синхронная логика вызова основной модели."""
    # Проверяем, является ли response_model классом Pydantic
    is_pydantic = hasattr(response_model, "model_validate")
    
    res = call_llm(prompt, input_text, is_json=is_pydantic)
    
    if is_pydantic:
        # Если пришел словарь, валидируем его через Pydantic
        return response_model.model_validate(res)
    
    # Если просили строку, возвращаем как есть
    return str(res)

def call_qwen_coder_32b(prompt, data, task):
    """Синхронная заглушка для Sandbox."""
    # Для DataFrame используем to_string(), как в оригинале
    data_str = data.to_string() if isinstance(data, pd.DataFrame) else str(data)
    return call_llm(prompt, f"Data: {data_str}\nTask: {task}", is_json=False)

def hybrid_search_thesaurus(term: str, domain: str):
    """Синхронная моковая база данных."""
    thesaurus = {
        "FFP": {"id": "KPI_001", "name": "Fatal Frequency Rate", "category": "KPI", "score": 0.98},
        "ЯГОК": {"id": "SEG_099", "name": "Яковлевский ГОК", "category": "SEGMENT", "score": 0.95},
        "НИ": {"id": "CUMM_MODE", "name": "Накопленный итог", "category": "KF_TYPE", "score": 0.55},
    }
    term_up = term.upper()
    # Простой поиск по ключам
    res = next((v for k, v in thesaurus.items() if k in term_up or term_up in k), None)
    if res:
        return [{**res}]
    return []

def validate_sql_with_explain(sql: str):
    """Синхронная имитация валидации SQL."""
    if not sql or "SELECT" not in sql.upper():
        return False, "Query must be a valid SELECT statement"
    return True, ""

def execute_clickhouse_query(sql: str):
    """Синхронная имитация получения данных из ClickHouse."""
    data = {
        "CALMONTH": ["202409", "202410", "202411", "202412", "202501"],
        "VALUE": [0.10, 0.25, 0.38, 0.50, 0.08],
        "KF_TYPE": ["CUMM"] * 5
    }
    return pd.DataFrame(data)

def extract_date_logic_py(text: str):
    """Синхронный хелпер для дат."""
    return {"start": "202409", "end": "202502"}

import json
import pyperclip

filename = 'company_segment.json'

try:
    with open(filename, 'r', encoding='utf-8') as file:
        companies = json.load(file)
        
        # Собираем все имена в один список
        names = []
        for company in companies:
            if "name" in company:
                names.append(company["name"])
        
        # Объединяем имена через перенос строки
        result_text = "\n".join(names)
        
        # Копируем результат в буфер обмена
        pyperclip.copy(result_text)
        
        print(f"Успешно! {len(names)} наименований скопировано в буфер обмена.")
                
except FileNotFoundError:
    print(f"Ошибка: Файл '{filename}' не найден.")
except json.JSONDecodeError:
    print("Ошибка: Файл имеет неверный формат JSON.")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")

