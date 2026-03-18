import uuid
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver

# Импортируй свои функции и логику из своего скрипта
# (Предположим, они в файле logic.py или в этом же файле)
# Из твоего кода нам нужны: AgentState, workflow, и все узлы.

app = FastAPI(title="Analytic Assistant API")

from fastapi.middleware.cors import CORSMiddleware

# Добавь это сразу после создания app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Разрешает запросы с любого адреса (для тестов)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Схемы данных для API ---

class EntityItem(BaseModel):
    # Поля для найденных сущностей (KPI, COMPANY)
    id: Optional[str] = None
    name: Optional[str] = None
    source_term: Optional[str] = None
    score: Optional[float] = None
    reason: Optional[str] = None
    
    # Поля для MISSING сущностей
    term: Optional[str] = None
    expected_type: Optional[str] = None
    message: Optional[str] = None

# Модель ответа от API
class ChatResponse(BaseModel):
    thread_id: str
    status: str
    ask_user: Optional[str] = None # Тот самый вопрос от LLM, если он есть
    # Словарь, где ключ - категория (KPI, COMPANY...), а значение - список сущностей
    categories: Dict[str, List[EntityItem]] = Field(default_factory=dict)
    answer: Optional[str] = None

# Модель для подтверждения от пользователя
class SelectionRequest(BaseModel):
    thread_id: str
    selected_ids: Optional[List[str]] = None # Список ID из чекбоксов
    text_feedback: Optional[str] = None # Если юзер написал текст вместо выбора

# Модель ответа (финальный результат)
class FinalResponse(BaseModel):
    status: str
    answer: str # Финальный ответ от LLM (результат анализа или SQL)

# --- Настройка Графа (Синхронная) ---

memory = MemorySaver()

# ВАЖНО: Модифицируй human_in_the_loop_node, чтобы он ничего не спрашивал через input()
def human_in_the_loop_node(state: AgentState):
    """
    Этот узел теперь не спрашивает input(). 
    Он запускается автоматически ПОСЛЕ того, как пользователь 
    прислал данные через API эндпоинт /confirm.
    """
    logger.info("--- HUMAN CORRECTION NODE ---")
    
    # К этому моменту в state.user_feedback и state.resolved_entities 
    # уже лежат данные, которые мы записали в эндпоинте /confirm.
    
    feedback = state.user_feedback
    entities = state.resolved_entities

    # Если пользователь просто подтвердил (нажал ОК), 
    # мы гарантируем, что у всех сущностей confidence = 1.0, 
    # чтобы пройти валидацию в conditional_edges.
    if feedback == "CONFIRMED":
        for e in entities:
            e.confidence = 1.0
            
    print(f"Обработка фидбека: {feedback}")
    
    return {
        "user_feedback": feedback,
        "resolved_entities": entities,
        "human_retry_count": state.human_retry_count + 1
    }

def should_continue_after_human(state: AgentState):
    # Если пользователь написал текстовый фидбек (исправление), 
    # возвращаемся в начало на перепарсинг интента.
    if state.user_feedback and state.user_feedback != "CONFIRMED":
        return "reparse_intent"
    
    # Если всё подтверждено — идем генерить SQL
    return "generate_sql"

# В сборке графа:
workflow.add_conditional_edges(
    "human_correction",
    should_continue_after_human,
    {
        "reparse_intent": "intent_parser",  # Возврат в начало
        "generate_sql": "sql_generator"    # Вперед к данным
    }
)

# Собираем граф (используй свой существующий код сборки, но с прерыванием)
# workflow = StateGraph(AgentState)
# ... (добавление всех твоих узлов) ...

# Обязательно добавь interrupt_before перед узлом, где нужно подтверждение
compiled_graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["human_correction"] 
)

# --- Эндпоинты API ---

@app.post("/ask", response_model=ChatResponse)
def ask(query: str, thread_id: str = "demo_id"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Прогон графа
    for event in app.stream({"raw_query": query}, config):
        pass
        
    state = app.get_state(config)
    
    # Получаем словарь из стейта. Если его нет, отдаем пустой словарь
    entities_dict = state.values.get("resolved_entities", {})
    
    return ChatResponse(
        thread_id=thread_id,
        status="awaiting_user",
        ask_user=state.values["intent"].ask_user if state.values.get("intent") else None,
        categories=entities_dict # FastAPI сам разложит это по полочкам
    )

@app.post("/confirm", response_model=FinalResponse)
async def confirm_selection(request: SelectionRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # 1. Получаем текущее состояние графа
    current_state = app.get_state(config)
    if not current_state.values:
        return {"status": "error", "answer": "Сессия не найдена"}

    # 2. Обработка ВЫБОРА ИЗ СПИСКА (selected_ids)
    if request.selected_ids:
        # Берем текущий словарь сущностей (KPI, COMPANY и т.д.)
        old_entities_dict = current_state.values.get("resolved_entities", {})
        new_entities_dict = {}

        # Проходим по категориям и оставляем только то, что выбрал юзер
        for category, items in old_entities_dict.items():
            if category == "MISSING":
                continue # Пропущенные не фильтруем по ID
                
            filtered_items = [
                item for item in items 
                if str(item.get("id")) in request.selected_ids
            ]
            
            if filtered_items:
                # Ставим score 1.0, чтобы SQL-генератор не сомневался
                for item in filtered_items:
                    item["score"] = 1.0
                new_entities_dict[category] = filtered_items

        # ОБНОВЛЯЕМ СТЕЙТ: записываем отфильтрованные сущности и ставим статус CONFIRMED
        app.update_state(config, {
            "resolved_entities": new_entities_dict,
            "user_feedback": "CONFIRMED"
        }, as_node="human_correction")

    # 3. Обработка ТЕКСТОВОГО ИСПРАВЛЕНИЯ
    elif request.text_feedback:
        app.update_state(config, {
            "user_feedback": request.text_feedback
        }, as_node="human_correction")

        # 4. ЗАПУСКАЕМ ГРАФ ДАЛЬШЕ
    try:
        for event in compiled_graph.stream(None, config, stream_mode="values"):
            pass # Нам не нужно сохранять event, мы возьмем весь стейт ниже
    except Exception as e:
        print(f"Ошибка при завершении графа: {e}")
        return FinalResponse(status="error", answer="Ошибка при генерации ответа")

    # 5. ПРОВЕРЯЕМ СТАТУС ГРАФА ПОСЛЕ ВОЗОБНОВЛЕНИЯ
    state_after_resume = compiled_graph.get_state(config)
    
    # Если граф снова прервался и ждет ввода от пользователя
    if state_after_resume.next and "human_correction" in state_after_resume.next:
        entities_dict = state_after_resume.values.get("resolved_entities", {})
        intent_obj = state_after_resume.values.get("intent")
        ask_user_text = intent_obj.ask_user if intent_obj else "Пожалуйста, уточните сущности:"
        
        # Возвращаем тот же ответ, что и в /ask, чтобы фронт снова показал чекбоксы
        return {
            "thread_id": request.thread_id,
            "status": "awaiting_confirmation", # или awaiting_user
            "ask_user": ask_user_text,
            "categories": entities_dict
        }

    # 6. Если граф дошел до конца (END)
    answer = state_after_resume.values.get("final_analysis", "Анализ завершен успешно")

    return FinalResponse(
        status="success",
        answer=answer
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
