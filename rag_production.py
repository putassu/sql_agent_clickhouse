import json
import os
import pickle
import uuid
import logging
import hashlib
from typing import List, Optional, Dict, Any, Set

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchAny
)
from rank_bm25 import BM25Okapi
import tiktoken

import httpx  # Добавьте в импорты в самом верху файла
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # Чтобы не спамило варнингами

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    QDRANT_HOST = "10.x.x.x"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "corporate_thesaurus"
    API_BASE_URL = "https://your-corp-api/v1"
    API_KEY = "your-api-key"
    EMBEDDING_MODEL = "bge-m3"
    VECTOR_SIZE = 1024
    
    DATA_FILES = {
        "kpi.json": "KPI",
        "company.json": "COMPANY",
        "company_segment.json": "COMPANY_SEGMENT"
    }
    
    # Файлы состояния
    PROGRESS_FILE = "indexing_progress.jsonl" # Храним тут тексты и метаданные для BM25
    BM25_STORAGE_PATH = "bm25_local_index.pkl"
    RRF_K = 60

class ResumableSearchEngine:
    def __init__(self):
        self.client = OpenAI(base_url=Config.API_BASE_URL, api_key=Config.API_KEY)
        self.qdrant = QdrantClient(host=Config.QDRANT_HOST, port=Config.QDRANT_PORT)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bm25_data = self._load_bm25()

    # Создаем HTTP-клиент, который игнорирует проверку SSL
        unsafe_client = httpx.Client(verify=False) 
        
        self.client = OpenAI(
            base_url=Config.API_BASE_URL, 
            api_key=Config.API_KEY,
            http_client=unsafe_client  # <--- Передаем этот клиент сюда
        )
        
        # # Для Qdrant (на случай, если там тоже SSL)
        # self.qdrant = QdrantClient(
        #     host=Config.QDRANT_HOST, 
        #     port=Config.QDRANT_PORT,
        #     https=False # Если используете http://
        # )
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.bm25_data = self._load_bm25()

    def _generate_deterministic_id(self, parent_id: str, text: str) -> str:
        """Создает уникальный UUID на основе контента, чтобы избежать дублей."""
        unique_str = f"{parent_id}_{text}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

    def _get_embedding(self, text: str) -> List[float]:
        """Получение вектора через API с обработкой ошибок."""
        try:
            response = self.client.embeddings.create(input=text, model=Config.EMBEDDING_MODEL)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Ошибка API при обработке текста '{text[:30]}...': {e}")
            raise

    def _load_bm25(self):
        if os.path.exists(Config.BM25_STORAGE_PATH):
            with open(Config.BM25_STORAGE_PATH, "rb") as f:
                return pickle.load(f)
        return None

    def _get_indexed_ids_from_qdrant(self) -> Set[str]:
        """Собирает все ID чанков, которые уже есть в Qdrant."""
        indexed_ids = set()
        try:
            # Используем scroll для получения всех ID без векторов
            offset = None
            while True:
                res, offset = self.qdrant.scroll(
                    collection_name=Config.COLLECTION_NAME,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                    offset=offset
                )
                for point in res:
                    indexed_ids.add(str(point.id))
                if offset is None:
                    break
            logger.info(f"Найдено {len(indexed_ids)} существующих записей в Qdrant.")
        except Exception as e:
            logger.warning(f"Коллекция не найдена или пуста: {e}")
        return indexed_ids

    def index_data(self, clean_start: bool = False):
        """
        Индексация с поддержкой дозаписи.
        :param clean_start: Если True, удалит старую базу и начнет с нуля.
        """
        if clean_start:
            logger.info("Удаление старой коллекции и файла прогресса...")
            try: self.qdrant.delete_collection(Config.COLLECTION_NAME)
            except: pass
            if os.path.exists(Config.PROGRESS_FILE): os.remove(Config.PROGRESS_FILE)

        # 1. Создаем коллекцию если нет
        try:
            self.qdrant.get_collection(Config.COLLECTION_NAME)
        except:
            self.qdrant.create_collection(
                collection_name=Config.COLLECTION_NAME,
                vectors_config=VectorParams(size=Config.VECTOR_SIZE, distance=Distance.COSINE)
            )

        # 2. Собираем уже обработанные ID
        indexed_chunk_ids = self._get_indexed_ids_from_qdrant()
        
        # 3. Открываем файл прогресса для дозаписи (BM25 метаданные)
        progress_data = []
        if os.path.exists(Config.PROGRESS_FILE):
            with open(Config.PROGRESS_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    progress_data.append(json.loads(line))

        # 4. Основной цикл индексации
        for file_path, item_type in Config.DATA_FILES.items():
            if not os.path.exists(file_path): continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                items = json.load(f)
            
            logger.info(f"Обработка файла {file_path}...")
            
            for item in items:
                if item.get("checked") is False: continue
                
                parent_id = item.get("ID")
                variations = list(set(filter(None, [item.get("name")] + item.get("tags", []))))
                
                for text_variant in variations:
                    chunk_id = self._generate_deterministic_id(parent_id, text_variant)
                    
                    # Пропускаем, если этот чанк уже в Qdrant
                    if chunk_id in indexed_chunk_ids:
                        continue

                    # Получаем эмбеддинг
                    vector = self._get_embedding(text_variant)
                    
                    payload = {
                        "parent_id": parent_id,
                        "item_type": item_type,
                        "domain": item.get("domain", []),
                        "text_variant": text_variant,
                        "full_data": item
                    }
                    
                    # Пишем в Qdrant
                    self.qdrant.upsert(
                        collection_name=Config.COLLECTION_NAME,
                        points=[PointStruct(id=chunk_id, vector=vector, payload=payload)]
                    )
                    
                    # Пишем в файл прогресса (для BM25)
                    with open(Config.PROGRESS_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    
                    progress_data.append(payload)
                    indexed_chunk_ids.add(chunk_id)

        # 5. Пересобираем BM25 на основе накопленного прогресса
        if progress_data:
            logger.info("Сборка BM25 индекса из файла прогресса...")
            bm25_corpus = [ [str(t) for t in self.tokenizer.encode(p['text_variant'].lower())] for p in progress_data ]
            bm25_engine = BM25Okapi(bm25_corpus)
            
            with open(Config.BM25_STORAGE_PATH, "wb") as f:
                pickle.dump({"engine": bm25_engine, "meta": progress_data}, f)
            
            self.bm25_data = {"engine": bm25_engine, "meta": progress_data}
            logger.info("Индексация успешно завершена.")
        else:
            logger.info("Новых данных для индексации не обнаружено.")

    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5, 
        item_types: List[str] = None, 
        domains: List[str] = None
    ) -> List[Dict]:
        """
        Гибридный поиск с фильтрацией и RRF агрегацией.
        $$score = \frac{1}{k + rank}$$
        """
        if not self.bm25_data:
            raise ValueError("Индекс BM25 не найден. Сначала запустите index_data().")

        # --- 1. Векторный поиск ---
        must_filters = []
        if item_types:
            must_filters.append(FieldCondition(key="item_type", match=MatchAny(any=item_types)))
        if domains:
            must_filters.append(FieldCondition(key="domain", match=MatchAny(any=domains)))
            
        q_filter = Filter(must=must_filters) if must_filters else None
        
        query_vector = self._get_embedding(query)
        vector_hits = self.qdrant.search(
            collection_name=Config.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=q_filter,
            limit=top_k * 3  # Берем с запасом для агрегации
        )

        # --- 2. BM25 Поиск ---
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25_data["engine"].get_scores(query_tokens)
        
        # Ручная фильтрация BM25 (так как он локальный)
        bm25_hits = []
        for i, meta in enumerate(self.bm25_data["meta"]):
            if item_types and meta['item_type'] not in item_types:
                continue
            if domains and not any(d in meta['domain'] for d in domains):
                continue
            if bm25_scores[i] > 0:
                bm25_hits.append({"idx": i, "score": bm25_scores[i]})
        
        # Сортируем и берем топ
        bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)[:top_k * 3]

        # --- 3. Агрегация RRF (Reciprocal Rank Fusion) + Max-Pooling ---
        # Мы группируем по parent_id, так как у одной сущности много тегов/векторов
        entity_rrf_scores = {} # parent_id -> final_rrf_score
        entity_payloads = {}   # parent_id -> full_data

        # Добавляем результаты векторного поиска
        for rank, hit in enumerate(vector_hits):
            pid = hit.payload['parent_id']
            score = 1.0 / (Config.RRF_K + rank + 1)
            entity_rrf_scores[pid] = max(entity_rrf_scores.get(pid, 0), score)
            entity_payloads[pid] = hit.payload['full_data']

        # Добавляем результаты BM25
        for rank, hit in enumerate(bm25_hits):
            meta = self.bm25_data["meta"][hit['idx']]
            pid = meta['parent_id']
            score = 1.0 / (Config.RRF_K + rank + 1)
            entity_rrf_scores[pid] = max(entity_rrf_scores.get(pid, 0), score)
            if pid not in entity_payloads:
                entity_payloads[pid] = meta['full_data']

        # --- 4. Финализация ---
        results = []
        # Сортируем по итоговому RRF скору
        sorted_entities = sorted(entity_rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for pid, final_score in sorted_entities:
            output = entity_payloads[pid].copy()
            output['search_score'] = round(final_score, 4)
            results.append(output)
            
        return results

if __name__ == "__main__":
    engine = ResumableSearchEngine()
    
    # Запуск: если упадет, просто запустите снова.
    # Если нужно все очистить и начать с нуля: engine.index_data(clean_start=True)
    engine.index_data(clean_start=False) 

