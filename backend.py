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

# --- Схемы данных для API ---

class ChatRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None

class SelectionRequest(BaseModel):
    thread_id: str
    selected_ids: List[str] # Список ID, которые выбрал юзер в чекбоксах
    feedback: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    status: str # "need_confirmation" или "completed"
    entities: Optional[List[Dict]] = None # Для выбора
    answer: Optional[str] = None

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

@app.post("/ask")
def ask(query: str, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Запуск
    for event in app.stream({"raw_query": query}, config):
        pass

    state = app.get_state(config)
    entities_dict = state.values.get("resolved_entities", {})
    
    # Формируем структуру для фронта
    return {
        "status": "need_confirmation",
        "ask_user": state.values["intent"].ask_user if state.values.get("intent") else None,
        "categories": {
            "KPI": entities_dict.get("KPI", []),
            "COMPANY": entities_dict.get("COMPANY", []),
            "SEGMENT": entities_dict.get("COMPANY_SEGMENT", []),
            "MISSING": entities_dict.get("MISSING", [])
        }
    }

@app.post("/confirm")
def confirm(thread_id: str, selected_ids: list[str] = None, text_feedback: str = None):
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    
    # 1. Если это выбор из чекбоксов
    if selected_ids:
        old_entities_dict = state.values.get("resolved_entities", {})
        new_entities_dict = {}

        for category, items in old_entities_dict.items():
            if category == "MISSING":
                continue # Пропущенные сущности не идут в SQL, их должен исправить текст
            
            # Фильтруем список внутри категории
            filtered_items = [item for item in items if item.get("id") in selected_ids]
            
            if filtered_items:
                # Ставим всем выбранным score 1.0 для надежности
                for item in filtered_items:
                    item["score"] = 1.0
                new_entities_dict[category] = filtered_items
        
        # Обновляем стейт: подменяем старый словарь на отфильтрованный
        app.update_state(config, {
            "resolved_entities": new_entities_dict,
            "user_feedback": "CONFIRMED"
        }, as_node="human_correction")

    # 2. Если это текстовое исправление
    elif text_feedback:
        app.update_state(config, {
            "user_feedback": text_feedback
        }, as_node="human_correction")

    # Пробуждаем граф
    for event in app.stream(None, config):
        pass
        
    final_state = app.get_state(config)
    return {"answer": final_state.values.get("final_analysis", "Готово")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
