# 📊 Analytic Assistant API (LangGraph + FastAPI)

Интеллектуальный аналитический агент с поддержкой "Human-in-the-loop". Система извлекает сущности (KPI, Компании, Периоды) из текстовых запросов, запрашивает подтверждение у пользователя и выполняет финальный анализ данных.

---

## 🚀 Основные возможности
- **Двухэтапная обработка**: Сначала извлечение параметров, затем генерация ответа после подтверждения.
- **Умное разрешение сущностей**: Поиск подходящих KPI и Компаний в базе знаний.
- **Сохранение контекста**: Использование `MemorySaver` для ведения цепочки диалога через `thread_id`.
- **Интерактивный UI**: Современный фронтенд на Tailwind CSS с виджетами выбора.

---

## 🛠 Технологический стек
- **Backend**: FastAPI (Python 3.10+)
- **Graph Logic**: LangGraph (LangChain)
- **State Management**: Pydantic v2
- **Frontend**: HTML5, Tailwind CSS, Lucide Icons

---

## 📦 Установка и запуск

### 1. Среда и зависимости
```bash
# Клонируйте репозиторий (если есть) или создайте проект
python -m venv venv
source venv/bin/activate  # для Windows: venv\Scripts\activate

pip install fastapi uvicorn langgraph langchain pandas pydantic
```

### 2. Настройка CORS (Важно!)
Убедитесь, что в вашем `main.py` добавлены настройки CORS, иначе фронтенд не сможет достучаться до API:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Запуск сервера
```bash
uvicorn main:app --reload --port 8000
```

---

## 📡 API Эндпоинты

### 1. `POST /ask`
Инициация запроса. Граф доходит до узла коррекции и останавливается.
- **Query params**: `query` (текст), `thread_id` (ID сессии).
- **Response**: `ChatResponse` (список найденных сущностей и вопрос от LLM).

### 2. `POST /confirm`
Подтверждение выбранных сущностей или текстовое уточнение.
- **Body**: `SelectionRequest`
```json
{
  "thread_id": "demo_id",
  "selected_ids": ["kpi_001", "comp_002"],
  "text_feedback": null
}
```
- **Response**: `FinalResponse` (готовый результат анализа или SQL-запроса).

---

## 🧩 Схемы данных (Pydantic)

```python
class EntityItem(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    score: Optional[float] = None
    # ... поля для пропущенных сущностей

class ChatResponse(BaseModel):
    thread_id: str
    status: str
    categories: Dict[str, List[EntityItem]]
    ask_user: Optional[str]
```

---

## ⚠️ Решение частых проблем

### Ошибка: `Type is not msgpack serializable: DataFrame`
**Причина**: Вы пытаетесь сохранить объект Pandas DataFrame в `State` графа.
**Решение**: В узле, который генерирует данные, преобразуйте DataFrame в словарь:
`return {"data": df.to_dict(orient="records")}` или обнуляйте поле: `return {"df_field": None}`.

### Ошибка: `AttributeError: 'ResolvedEntity' object has no attribute 'items'`
**Причина**: `resolved_entities` — это Pydantic модель, а не словарь.
**Решение**: Используйте `old_entities_dict.model_dump().items()` (для Pydantic v2) или `.dict().items()` (для v1).

---

## 🎨 Фронтенд
Файл `index.html` находится в корне проекта. Просто откройте его в браузере. Убедитесь, что переменная `API_BASE` в скрипте совпадает с адресом вашего FastAPI сервера (по умолчанию `http://localhost:8000`).
