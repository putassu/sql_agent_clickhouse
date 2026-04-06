import logging
import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from utils import get_current_date, get_current_month, get_current_month_name, get_current_quarter, get_current_year
import pandas as pd
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
from rag_production import ResumableSearchEngine

engine = ResumableSearchEngine()

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
    execute_clickhouse_query,# Синхронное выполнение
    validate_sql_with_explain
)
from prompts import (
    INTENT_PROMPT,
    SQL_SELECT_TABLES_PROMPT, 
    SQL_GEN_PROMPT, 
    SANDBOX_PROMPT, 
    SYNTHESIS_PROMPT,
    ENTITY_RESOLVER_PROMPT
)

# Настройка логирования
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnalyticAgent")

# --- Модели данных ---

# class ResolvedEntity(BaseModel):
#     original_term: str
#     official_name: str
#     db_id: str
#     category: Literal["KPI", "SEGMENT", "STAFF_GROUP", "KF_TYPE", "PERIOD"]
#     confidence: float

class ResolvedEntity(BaseModel):
    KPI: List[Dict[str, Any]] = Field(default_factory=list)
    COMPANY: List[Dict[str, Any]] = Field(default_factory=list)
    COMPANY_SEGMENT: List[Dict[str, Any]] = Field(default_factory=list)
    MISSING_ENTITIES: List[Dict[str, Any]] = Field(default_factory=list)

class QueryIntent(BaseModel):
    dashboard_domain: str = Field(description="Домен: Травматизм, Финансы и т.д.")
    metrics: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    periods: List[Dict[str, Any]] = Field(default_factory=list)
    need_sandbox: bool = False
    calculation_task: Optional[str] = None
    ask_user: Optional[str] = None

class AgentState(BaseModel):
    """Состояние графа."""
    raw_query: str
    user_feedback: Optional[str] = None
    intent: Optional[QueryIntent] = None
    resolved_entities: Optional[ResolvedEntity] = None
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

    input_args = {"input_text": query,
                  "CURRENT_DATE": get_current_date(),
    "CURRENT_YEAR": get_current_year(),
    "CURRENT_MONTH": get_current_month(),
    "CURRENT_MONTH_NAME": get_current_month_name(),
    "CURRENT_QUARTER": get_current_quarter()}

    logger.info(f"[SENDED PROMPT]: {query}")

    response = call_deepseek_v3(
        prompt=INTENT_PROMPT,
        input_args=input_args,
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
    query = state.raw_query
    if state.user_feedback:
        query = f"{query}. {state.user_feedback}"

    # Собираем всё, что нужно найти (метрики + компании)
    kpis = [{"item_types":["KPI"], "term":term} for term in state.intent.metrics]
    others = [{"item_types":["COMPANY", "COMPANY_SEGMENT"], "term":term} for term in state.intent.entities]
    to_resolve = kpis + others
    logger.info(f"[TO RESOLVE]: {to_resolve}")
    kpi_user_message_matches = engine.hybrid_search(query=query, top_k=30, item_types=["KPI"]) #домен часто обнаруживается ошибочно, пришлось его убрать
    company_user_message_matches = engine.hybrid_search(query=query, top_k=30, item_types=["COMPANY", "COMPANY_SEGMENT"]) #если ищем юр.лица и структуры, domains не нужен, он только у KPI
    # print(company_user_message_matches)
    print('KPI MATCHES:')
    print(query, [state.intent.dashboard_domain])
    print(kpi_user_message_matches)
    print('------------------------------------------------')
    print('COMPANY MATCHES:')
    print(company_user_message_matches)
    print('------------------------------------------------')
    resolved_list = []
    all_entity_matches = kpi_user_message_matches + company_user_message_matches
    for item in to_resolve:
        term = item["term"]
        item_types = item["item_types"]
        # Ищем в векторной БД с метаданными домена
        entity_matches = engine.hybrid_search(query=term, top_k=20, item_types=item_types)
        all_entity_matches += entity_matches
        print(f'{term} MATCHES:')
        print(entity_matches)
        print('------------------------------------------------')

    input_args = {"user_query": query,
                  "to_resolve": to_resolve,
    "candidates_list": all_entity_matches,
    "num_candidates": len(all_entity_matches)}

    response = call_deepseek_v3(
        prompt=ENTITY_RESOLVER_PROMPT,
        input_args=input_args,
        response_model=ResolvedEntity
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
        print(type(response))

    logger.info(f"[RESOLVED]: {formatted_response}")

    return {"resolved_entities": formatted_response}

def human_in_the_loop_node(state: AgentState):
    # Фильтруем только то, в чем сомневаемся
    # uncertain = [e for e in state.resolved_entities if e.confidence < CONFIDENCE_THRESHOLD]

    print("\n" + "="*50)
    print("УТОЧНЕНИЕ ПАРАМЕТРОВ:")
    # for e in uncertain:
    #     print(f" - Термин '{e.original_term}' похож на '{e.official_name}' (Уверенность: {e.confidence*100:.0f}%)")

    # if not uncertain:
    #     print("Для всех терминов нашлось соответствие")
    ask_user = state.intent.ask_user
    new_entities = state.resolved_entities
    if ask_user:
        print("ЕСТЬ ASK_USER, ЗАПРАШИВАЮ INPUT")
        user_input = input(ask_user)
        feedback = user_input
        print(f"получил сообщение пользователя, вот оно: {feedback}")
    if state.resolved_entities:
        print("ЕСТЬ RESOLVED_ENTITIES, ЗАПРАШИВАЮ INPUT")
        user_input = input("\nПередайте JSON с отмеченными сущностями: ")
        feedback = '' #бэкенд чата должен также передавать feedback - юзер может что-то написать
        new_entities = json.loads(user_input)

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
        "schema": DB_SCHEMA_VERSION,"previous_error": state.sql_error
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
    """Узел 4: Проверка SQL (выполнение, проверка синтаксиса и наличия данных)."""
    logger.info("Validating SQL and checking for empty results...")
    
    try:
        # Пытаемся выполнить сгенерированный запрос
        df = query_clickhouse(state.sql_query)
        
        # Если запрос отработал без ошибок, но вернул $$0$$ строк (пустой датафрейм)
        if df.empty:
            error_msg = (
                "SQL_ERROR: Запрос вернул пустой результат. Возможно, нет данных за выбранный период. "
                "Попробуй другие фильтры. Например, если нет данных по продажам за год, "
                "но есть нарастающий итог на 12-й месяц года (что то же самое) — "
                "попробуй более широкие фильтры времени (возьми все записи по этому году, "
                "а последующие узлы сами разберутся, в какой строчке лежит ответ)."
            )
            logger.warning(f"SQL Validation failed (Empty Result): {error_msg}")
            return {"sql_error": error_msg}
        
        # Если данные есть, возвращаем отсутствие ошибок. 
        # (Опционально можно сразу сохранить данные в state, чтобы не делать запрос дважды)
        df_dict = df.astype(str).to_dict(orient="records")
        return {"sql_error": None, "sql_data": df_dict}
        
    except Exception as e:
        # Перехватываем реальные ошибки синтаксиса SQL от ClickHouse
        error_msg = f"SQL_ERROR: Ошибка выполнения запроса (синтаксис или схема данных): {str(e)}"
        logger.warning(f"SQL Validation failed (Execution Error): {error_msg}")
        return {"sql_error": error_msg}


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
    low_confidence = any(e["score"] < CONFIDENCE_THRESHOLD for e in state.resolved_entities.KPI)
    if low_confidence and not state.user_feedback:
        return "ask_user"
    if state.intent.ask_user:
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

workflow.add_conditional_edges(
    "intent_parser",
    should_ask_user,
    {
        "ask_user": "human_correction",
        "generate_sql": "entity_resolver"
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
