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

@app.post("/ask", response_model=ChatResponse)
def ask_question(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Начальный запуск
    # Если это новый запрос, передаем raw_query
    initial_input = {"raw_query": req.query}
    
    # Запускаем до первого прерывания или до конца
    for event in compiled_graph.stream(initial_input, config=config):
        pass # Просто прокручиваем узлы

    # Проверяем состояние после остановки
    state = compiled_graph.get_state(config)
    
    # Если граф остановился перед human_correction, значит нужны подтверждения
    if state.next and "human_correction" in state.next:
        return ChatResponse(
            thread_id=thread_id,
            status="need_confirmation",
            entities=[e.dict() for e in state.values.get("resolved_entities", []) if e.confidence < 0.9]
        )

    # Если граф дошел до конца
    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        answer=state.values.get("final_analysis")
    )

@app.post("/confirm", response_model=ChatResponse)
def confirm_entities(req: SelectionRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # 1. Получаем текущее состояние
    current_state = compiled_graph.get_state(config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Thread not found")

    # 2. Обновляем сущности (ставим confidence 1.0 тем, кого выбрал юзер)
    entities = current_state.values.get("resolved_entities", [])
    for e in entities:
        if e.db_id in req.selected_ids:
            e.confidence = 1.0 # Подтверждаем
    
    # 3. Записываем обновленное состояние и фидбек
    compiled_graph.update_state(config, {
        "resolved_entities": entities,
        "user_feedback": req.feedback
    })

    # 4. Продолжаем выполнение (передаем None, чтобы продолжить с точки прерывания)
    for event in compiled_graph.stream(None, config=config):
        pass

    # 5. Возвращаем финальный результат
    final_state = compiled_graph.get_state(config)
    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        answer=final_state.values.get("final_analysis")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
