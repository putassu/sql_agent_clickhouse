import requests
import base64
import io
import pandas as pd

# Константа с описанием логики (ваша переменная)
STRATEGY_DESCRIPTION = """
В этой таблице нужно делать финальную выборку для ответа на поставленный вопрос.

Важные столбцы в этой таблице:
...
"""

def calculate_in_sandbox(dfs: list[pd.DataFrame], df_names: list[str], calculation_task: str) -> dict:
    input_files = []
    data_previews = []
    
    # 1. Подготавливаем файлы и генерируем превью данных для промпта
    for df, name in zip(dfs, df_names):
        filename = f"{name}.csv"
        
        # Сохраняем в CSV для песочницы
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        base64_encoded = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')
        input_files.append({
            "name": filename,
            "mount_path": f"inputs/{filename}",
            "content_base64": base64_encoded
        })
        
        # Генерируем превью (первые 3 строки) для LLM
        preview_text = df.head(3).to_string(index=False)
        columns_info = ", ".join(df.columns)
        
        data_previews.append(f"--- Файл: inputs/{filename} ---\nСтолбцы: {columns_info}\nПример данных (первые 3 строки):\n{preview_text}\n")
        
    joined_previews = "\n".join(data_previews)
    
    # 2. Формируем подробный промпт
    prompt = f"""
Ты — опытный Data Scientist. Твоя задача — написать Python-скрипт для выполнения аналитической задачи.

ЗАДАЧА: 
{calculation_task}

ОПИСАНИЕ СТРУКТУРЫ ДАННЫХ (БИЗНЕС-ЛОГИКА):
{STRATEGY_DESCRIPTION}

ВАЖНЫЕ ОСОБЕННОСТИ ДАННЫХ (КРИТИЧНО ДЛЯ РАБОТЫ):
1. ВСЕ значения в CSV файлах имеют строковый тип (string), даже если это числа (например, '10.5' или '2024').
2. Пропущенные значения в базе представлены не как классические NULL/NaN, а как пустые строки `""`. 
3. Перед выполнением любых математических операций (sum, mean, corr и т.д.) или фильтраций (например, `df['IS_KF_ACT_NULL'] == 1`) ОБЯЗАТЕЛЬНО конвертируй нужные столбцы в числовой формат.
4. Рекомендуемый способ конвертации: `pd.to_numeric(df['column_name'], errors='coerce')`, пустые строки при этом станут NaN, которые легко обработать через `.fillna(0)` или игнорировать агрегатными функциями.

ПРЕДОСТАВЛЕННЫЕ ФАЙЛЫ И ПРИМЕРЫ ДАННЫХ:
{joined_previews}

ТРЕБОВАНИЯ К КОДУ:
- Прочитай нужные файлы из папки `inputs/` с помощью pandas (например, `pd.read_csv('inputs/STRFSTRATEGY01.csv', dtype=str)` - принудительно читай как строки, чтобы ничего не потерять).
- Обработай данные с учетом бизнес-логики и особенностей типов.
- Выведи финальный результат в консоль с помощью `print()`. Вывод должен быть понятным и содержать только ответ на задачу.
- Верни ТОЛЬКО готовый к выполнению Python код, без markdown-разметки (без ```python ... ```) и без лишних комментариев.
"""

    # 3. Получаем код от Qwen
    generated_code = call_qwen_coder(prompt)
    
    # Очистка разметки (на всякий случай)
    generated_code = generated_code.strip()
    if generated_code.startswith("```python"):
        generated_code = generated_code[9:]
    if generated_code.endswith("```"):
        generated_code = generated_code[:-3]
        
    print("Сгенерированный код для проверки:\n", generated_code)
    
    # 4. Выполняем в песочнице
    sandbox_url = "http://10.114.86.136:8000/run_sandbox_task"
    payload = {
        "task_id": "analytics_task",
        "language": "python",
        "code": generated_code.strip(),
        "libraries": ["pandas", "numpy", "scipy"],
        "input_files": input_files
    }
    
    try:
        response = requests.post(sandbox_url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if result.get('status') == 'success':
            print("\nУСПЕХ! Результат вычислений:\n", result.get('output'))
        else:
            print("\nОШИБКА В ПЕСОЧНИЦЕ:\n", result.get('error'))
            
        return result
    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None
