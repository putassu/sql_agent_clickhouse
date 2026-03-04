import logging
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import pandas as pd
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json

# Импорты ваших будущих модулей (моки) - теперь подразумевается, что они синхронные
from config import (
    a,
    LOG_LEVEL, 
    CONFIDENCE_THRESHOLD, 
    NUM_RETRIES,
    MAX_HUMAN_RETRIES, 
    DB_SCHEMA_VERSION
)
from utils import (
    call_deepseek_v3,        # Синхронный вызов основной логической модели
    call_qwen_coder_32b,     # Синхронный вызов модели для Sandbox
    hybrid_search_thesaurus, # Синхронный поиск по векторной БД + BM25
    execute_clickhouse_query,# Синхронное выполнение
    validate_sql_with_explain,
    extract_date_logic_py    # Python-хелпер для дат
)
from prompts import (
    INTENT_PROMPT,
    SQL_SELECT_TABLES_PROMPT, 
    SQL_GEN_PROMPT, 
    SANDBOX_PROMPT, 
    SYNTHESIS_PROMPT
)

# Настройка логирования
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnalyticAgent")

# --- Модели данных ---

class ResolvedEntity(BaseModel):
    original_term: str
    official_name: str
    db_id: str
    category: Literal["KPI", "SEGMENT", "STAFF_GROUP", "KF_TYPE", "PERIOD"]
    confidence: float

class QueryIntent(BaseModel):
    dashboard_domain: str = Field(description="Домен: Травматизм, Финансы и т.д.")
    metrics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    periods: List[Dict[str, Any]] = Field(default_factory=list)
    need_sandbox: bool = False
    calculation_task: Optional[str] = None

class AgentState(BaseModel):
    """Состояние графа."""
    raw_query: str
    user_feedback: Optional[str] = None
    intent: Optional[QueryIntent] = None
    resolved_entities: List[ResolvedEntity] = []
    sql_query: Optional[str] = None
    sql_error: Optional[str] = None
    sql_data: Optional[Any] = None # pd.DataFrame
    final_analysis: Optional[str] = None
    retry_count: int = 0
    human_retry_count: int = 0
    selected_tables: List[str] = Field(default_factory=list)

# --- Узлы графа (Nodes) ---

def intent_parser_node(state: AgentState):
    """Узел 1: Извлечение намерений через DeepSeek-V3."""
    logger.info("Starting Intent Parsing...")
    
    # Если есть фидбек от пользователя, объединяем его с запросом
    query = state.raw_query
    if state.user_feedback:
        query = f"Original query: {query}. User correction: {state.user_feedback}"

    logger.info(f"[SENDED PROMPT]: {query}")

    response = call_deepseek_v3(
        prompt=INTENT_PROMPT,
        input_text=query,
        response_model=QueryIntent
    )
    if isinstance(response, dict):
        formatted_response = json.dumps(response, indent=4, ensure_ascii=False)
    elif isinstance(response, str):
        try:
            formatted_response = json.loads(response)
            formatted_response = json.dumps(formatted_response, indent=4, ensure_ascii=False)
        except Exception as ex:
            logger.critical(f"response NOT PARSEBLE: {response}")
    else:
        formatted_response = response
    logger.info(f"[RECEIVED RESPONSE]: {formatted_response}")
    
    return {"intent": response, "user_feedback": state.user_feedback}

def entity_resolver_node(state: AgentState):
    """Узел 2: Семантический поиск ID в тезаурусе (RAG)."""
    logger.info("Resolving Entities via Hybrid Search...")
    
    # Собираем всё, что нужно найти (метрики + компании)
    to_resolve = state.intent.metrics + state.intent.entities

    logger.info(f"[TO RESOLVE]: {to_resolve}")

    resolved_list = []
    for term in to_resolve:
        # Ищем в векторной БД с метаданными домена
        matches = hybrid_search_thesaurus(term, domain=state.intent.dashboard_domain)
        if matches:
            # Берем лучший результат
            best_match = matches[0]
            resolved_list.append(ResolvedEntity(
                original_term=term,
                official_name=best_match['name'],
                db_id=best_match['id'],
                category=best_match['category'],
                confidence=best_match['score']
            ))

    logger.info(f"[RESOLVED]: {resolved_list}")

    return {"resolved_entities": resolved_list}

def human_in_the_loop_node(state: AgentState):
    # Фильтруем только то, в чем сомневаемся
    uncertain = [e for e in state.resolved_entities if e.confidence < CONFIDENCE_THRESHOLD]
    
    print("\n" + "="*50)
    print("УТОЧНЕНИЕ ПАРАМЕТРОВ:")
    for e in uncertain:
        print(f" - Термин '{e.original_term}' похож на '{e.official_name}' (Уверенность: {e.confidence*100:.0f}%)")
    
    if not uncertain:
        print("Для всех терминов нашлось соответствие")

    user_input = input("\nВведите исправление (например: 'НИ это Накопленный Итог') или 'ок' для подтверждения: ")
    
    # Если пользователь просто подтвердил, искусственно поднимаем скор всем сущностям, 
    # чтобы пройти проверку на следующем цикле
    new_entities = state.resolved_entities
    if user_input.lower() in ['ок', 'подтверждаю', '']:
        for e in new_entities:
            e.confidence = 1.0
        feedback = "User confirmed all entities."
    else:
        feedback = user_input

    return {
        "user_feedback": feedback, 
        "resolved_entities": new_entities,
        "human_retry_count": state.human_retry_count + 1
    }

def sql_tables_selector_node(state: AgentState):
    """Узел 3: Выбор необходимых таблиц и обрезка схемы БД, чтобы не подавать все таблицы"""
    logger.info("Select tables...")
    prompt_context = {
        "intent": state.intent.model_dump(),
        "entities": [e.model_dump() for e in state.resolved_entities],
        "schema": DB_SCHEMA_VERSION,
        "previous_error": state.sql_error
    }
    tables = call_deepseek_v3(
        prompt=SQL_SELECT_TABLES_PROMPT,
        input_text=str(prompt_context),
        response_model=str # Возвращает список таблиц
    )
    return {"selected_tables": tables}
    

def sql_generator_node(state: AgentState):
    """Узел 3: Генерация SQL на основе разрешенных сущностей."""
    logger.info("Generating SQL Query...")
    
    prompt_context = {
        "intent": state.intent.model_dump(),
        "entities": [e.model_dump() for e in state.resolved_entities],
        "schema": DB_SCHEMA_VERSION,
        "previous_error": state.sql_error
    }
    
    logger.info(f"[SQL GENERATION PROMPT]: {json.dumps(prompt_context, indent=4, ensure_ascii=False)}")
    
    sql = call_deepseek_v3(
        prompt=SQL_GEN_PROMPT,
        input_text=str(prompt_context),
        response_model=str # Возвращает чистый SQL
    )
    logger.info(f"[SQL]: {sql}")
    return {"sql_query": sql, "retry_count": state.retry_count + 1}

def sql_validator_node(state: AgentState):
    """Узел 4: Проверка SQL через EXPLAIN."""
    logger.info("Validating SQL...")
    is_valid, error_msg = validate_sql_with_explain(state.sql_query)
    
    if not is_valid:
        logger.warning(f"SQL Validation failed: {error_msg}")
        return {"sql_error": error_msg}
    return {"sql_error": None}

def execution_node(state: AgentState):
    """Узел 5: Выполнение запроса в ClickHouse."""
    logger.info("Executing SQL in ClickHouse...")
    df = execute_clickhouse_query(state.sql_query)
    return {"sql_data": df}

def sandbox_node(state: AgentState):
    """Узел 6: Сложная аналитика в Python (Qwen-Coder)."""
    logger.info("Running Sandbox Analytics (Qwen)...")
    logger.info(f"[TASK]: {state.intent.calculation_task}")
    logger.info(f"[SQL DATA]: {state.sql_data[:300]}")
    # Передаем данные в Qwen для написания и выполнения кода
    analysis_result = call_qwen_coder_32b(
        prompt=SANDBOX_PROMPT,
        data=state.sql_data,
        task=state.intent.calculation_task
    )
    logger.info(f"[ANALYSIS RESULT]: {analysis_result}")
    return {"final_analysis": analysis_result}

def synthesizer_node(state: AgentState):
    """Узел 7: Финальный ответ."""
    logger.info("Synthesizing Final Answer...")
    
    data_to_show = state.final_analysis if state.final_analysis else state.sql_data.to_string()
    
    answer = call_deepseek_v3(
        prompt=SYNTHESIS_PROMPT,
        input_text=f"Query: {state.raw_query}\nData: {data_to_show}",
        response_model=str
    )
    return {"final_analysis": answer}

# --- Логика ветвления (Conditional Edges) ---

def should_ask_user(state: AgentState):
    if state.human_retry_count >= MAX_HUMAN_RETRIES:
        print("\n[SYSTEM]: Превышено число попыток уточнения. Попробую продолжить с тем, что есть...")
        return "generate_sql"
    """Проверка уверенности в сущностях."""
    if not state.resolved_entities:
        return "ask_user"
    low_confidence = any(e.confidence < CONFIDENCE_THRESHOLD for e in state.resolved_entities)
    if low_confidence and not state.user_feedback:
        return "ask_user"
    return "generate_sql"

def check_sql_status(state: AgentState):
    """Проверка валидности SQL."""
    if state.sql_error:
        if state.retry_count >= NUM_RETRIES:
            return "fail_end"
        return "regenerate_sql"
    return "execute_query"

def check_sandbox_needed(state: AgentState):
    """Нужен ли Sandbox."""
    if state.intent.need_sandbox:
        return "run_sandbox"
    return "synthesize"

# --- Сборка Графа ---

workflow = StateGraph(AgentState)

# Добавляем узлы
workflow.add_node("intent_parser", intent_parser_node)
workflow.add_node("entity_resolver", entity_resolver_node)
workflow.add_node("human_correction", human_in_the_loop_node)
workflow.add_node("sql_generator", sql_generator_node)
workflow.add_node("sql_validator", sql_validator_node)
workflow.add_node("data_executor", execution_node)
workflow.add_node("sandbox_analytics", sandbox_node)
workflow.add_node("synthesizer", synthesizer_node)

# Устанавливаем ребра
workflow.set_entry_point("intent_parser")
workflow.add_edge("intent_parser", "entity_resolver")

workflow.add_conditional_edges(
    "entity_resolver",
    should_ask_user,
    {
        "ask_user": "human_correction",
        "generate_sql": "sql_generator"
    }
)

workflow.add_edge("human_correction", "intent_parser") # Возврат на уточнение
workflow.add_edge("sql_generator", "sql_validator")

workflow.add_conditional_edges(
    "sql_validator",
    check_sql_status,
    {
        "regenerate_sql": "sql_generator",
        "execute_query": "data_executor",
        "fail_end": END
    }
)

workflow.add_conditional_edges(
    "data_executor",
    check_sandbox_needed,
    {
        "run_sandbox": "sandbox_analytics",
        "synthesize": "synthesizer"
    }
)

workflow.add_edge("sandbox_analytics", "synthesizer")
workflow.add_edge("synthesizer", END)

# Компиляция
app = workflow.compile()

# --- Запуск (Main Loop) ---

def main():
    print("--- Агент-Аналитик (DeepSeek + Qwen + LangGraph) ---")
    user_query = input("Введите ваш запрос: ")
    
    initial_state = AgentState(raw_query=user_query)
    
    # Используем метод stream вместо astream для синхронного запуска
    for event in app.stream(initial_state):
        for node_name, output in event.items():
            logger.info(f"Finished node: {node_name}")
            # Здесь можно выводить промежуточные результаты для отладки
            if "final_analysis" in output and node_name == "synthesizer":
                print(f"\n[ОТВЕТ]: {output['final_analysis']}")

if __name__ == "__main__":
    main()
