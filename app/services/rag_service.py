import os
import re
import json
import hashlib
from typing import Optional, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import settings

PARENT_STORE_PATH = os.path.join(settings.chroma_path, "parent_store.json")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Remove common PDF extraction artifacts before chunking."""
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # isolated page numbers
    text = re.sub(r'-\n', '', text)                                # hyphenated line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)                        # excess blank lines
    text = re.sub(r'[ \t]{2,}', ' ', text)                        # multiple spaces/tabs
    return text.strip()


# ---------------------------------------------------------------------------
# VectorDB singleton manager
# ---------------------------------------------------------------------------

class VectorDBManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._db = None
            instance._embeddings = None
            instance._cross_encoder = None
            instance._parent_store = None
            instance._bm25_cache = {}
            cls._instance = instance
        return cls._instance

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embeddings_model,
                model_kwargs={"device": "cpu"}
            )
        return self._embeddings

    @property
    def db(self):
        if self._db is None:
            self._db = Chroma(
                persist_directory=settings.chroma_path,
                embedding_function=self.embeddings
            )
        return self._db

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(settings.rerank_model)
        return self._cross_encoder

    @property
    def parent_store(self) -> dict:
        if self._parent_store is None:
            if os.path.exists(PARENT_STORE_PATH):
                with open(PARENT_STORE_PATH, 'r', encoding='utf-8') as f:
                    self._parent_store = json.load(f)
            else:
                self._parent_store = {}
        return self._parent_store

    def save_parent_store(self):
        os.makedirs(os.path.dirname(os.path.abspath(PARENT_STORE_PATH)), exist_ok=True)
        with open(PARENT_STORE_PATH, 'w', encoding='utf-8') as f:
            json.dump(self._parent_store, f, ensure_ascii=False, indent=2)

    def reset_connection(self):
        self._db = None
        self._parent_store = None
        self._bm25_cache = {}

    def add_child_documents(self, children: list, parents: dict) -> int:
        """Persist parent metadata and add child chunks to ChromaDB."""
        self._parent_store = self.parent_store
        self._parent_store.update(parents)
        self.save_parent_store()
        self.db.add_documents(children)
        self._bm25_cache = {}  # invalidate BM25 index cache
        return len(children)

    def get_bm25_index(self, topic: Optional[str] = None):
        """Build (or return cached) BM25 index over child chunks for a topic."""
        cache_key = topic or "__all__"
        if cache_key not in self._bm25_cache:
            collection = self.db._collection
            if collection.count() == 0:
                return None, [], []

            query_params = {"include": ["documents", "metadatas"]}
            if topic:
                query_params["where"] = {"topic": topic}

            results = collection.get(**query_params)
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            if not docs:
                return None, [], []

            tokenized = [doc.lower().split() for doc in docs]
            self._bm25_cache[cache_key] = (BM25Okapi(tokenized), docs, metas)

        return self._bm25_cache[cache_key]


vector_manager = VectorDBManager()


# ---------------------------------------------------------------------------
# Ingestion: parent-child chunking with semantic splitter
# ---------------------------------------------------------------------------

def _compute_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def _create_parent_chunks(documents: list) -> list:
    """
    Split documents into semantically coherent parent chunks.
    Uses SemanticChunker with a fallback to RecursiveCharacterTextSplitter
    for chunks that exceed the max parent size.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(
            vector_manager.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85
        )
        raw = splitter.split_documents(documents)
        # Enforce hard max to avoid sending huge blobs to the LLM
        max_size = settings.parent_chunk_size
        fallback = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=100)
        result = []
        for chunk in raw:
            if len(chunk.page_content) > max_size * 1.5:
                result.extend(fallback.split_documents([chunk]))
            else:
                result.append(chunk)
        return result
    except Exception:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter.split_documents(documents)


def process_pdf_to_vector_db(file_path: str, topic: str) -> int:
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()

    # 1. Clean text in each page
    for doc in raw_documents:
        doc.page_content = _clean_text(doc.page_content)
    raw_documents = [d for d in raw_documents if len(d.page_content.strip()) > 50]

    # 2. Semantic parent chunks
    parent_docs = _create_parent_chunks(raw_documents)

    # 3. Child chunks (small, for precise retrieval)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
        add_start_index=True
    )

    source_name = os.path.basename(file_path)

    # Load existing content hashes for this topic to skip duplicates
    existing_store = vector_manager.parent_store
    existing_hashes = {
        v.get("content_hash")
        for v in existing_store.values()
        if v.get("topic") == topic
    }

    child_docs: List[Document] = []
    parent_store_update: dict = {}
    seen_hashes: set = set()

    for i, parent in enumerate(parent_docs):
        parent_hash = _compute_hash(parent.page_content)

        # 4. Deduplication: skip parents whose content was already indexed
        if parent_hash in existing_hashes or parent_hash in seen_hashes:
            continue
        seen_hashes.add(parent_hash)

        parent_id = f"{topic}_{parent_hash}"
        page_num = parent.metadata.get("page", i)

        # 5. Rich metadata stored alongside parent content
        parent_store_update[parent_id] = {
            "content": parent.page_content,
            "topic": topic,
            "source": source_name,
            "page": page_num,
            "content_hash": parent_hash
        }

        # 6. Create children with back-reference to their parent
        children = child_splitter.split_documents([parent])
        for j, child in enumerate(children):
            child.metadata.update({
                "topic": topic,
                "source": source_name,
                "page": page_num,
                "parent_id": parent_id,
                "chunk_position": j,
                "total_siblings": len(children)
            })
            child_docs.append(child)

    if not child_docs:
        return 0

    return vector_manager.add_child_documents(child_docs, parent_store_update)


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(results_lists: list, k: int = 60) -> list:
    """Merge multiple ranked lists into one via Reciprocal Rank Fusion."""
    scores: dict = {}
    doc_map: dict = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = _compute_hash(doc.page_content)
            if doc_id not in scores:
                scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids]


def _bm25_search(query: str, topic: Optional[str], k: int) -> list:
    """Keyword-based BM25 search over child chunks."""
    bm25, docs, metas = vector_manager.get_bm25_index(topic)
    if bm25 is None:
        return []

    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    return [
        Document(page_content=docs[idx], metadata=metas[idx] or {})
        for idx in top_indices
        if scores[idx] > 0
    ]


def _generate_query_variants(query: str) -> list:
    """Ask the LLM for 2 alternative phrasings to improve recall."""
    try:
        from app.services.llm_service import get_llm
        llm = get_llm()
        prompt = (
            "Genera exactamente 2 reformulaciones alternativas de esta pregunta para buscar "
            "en un manual de autoescuela española. Solo las reformulaciones, una por línea, "
            "sin numeración ni explicaciones.\n"
            f"Pregunta: {query}"
        )
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        lines = [l.strip() for l in content.strip().split('\n') if l.strip()]
        return [query] + lines[:2]
    except Exception:
        return [query]


def _fetch_parent_documents(child_docs: list) -> list:
    """Retrieve unique parent documents from the store given a list of children."""
    store = vector_manager.parent_store
    seen: list = []
    parents: list = []

    for child in child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in seen and parent_id in store:
            seen.append(parent_id)
            data = store[parent_id]
            parents.append(Document(
                page_content=data["content"],
                metadata={
                    "topic": data.get("topic", ""),
                    "source": data.get("source", ""),
                    "page": data.get("page", 0),
                    "parent_id": parent_id
                }
            ))
    return parents


def _cross_encode_rerank(query: str, docs: list, top_k: int) -> list:
    """Rerank documents using a cross-encoder for precise relevance scoring."""
    try:
        pairs = [(query, doc.page_content) for doc in docs]
        scores = vector_manager.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
    except Exception:
        return docs[:top_k]


# ---------------------------------------------------------------------------
# Main search entry point
# ---------------------------------------------------------------------------

def search_in_vector_db(
    query: str,
    topic: Optional[str] = None,
    k: Optional[int] = None,
    use_multi_query: bool = True,
    use_rerank: bool = True
) -> list:
    if k is None:
        k = settings.default_search_k

    fetch_k = k * settings.fetch_k_multiplier

    # Multi-query expansion
    query_variants = _generate_query_variants(query) if use_multi_query else [query]

    filter_dict = {"topic": topic} if topic else None
    all_child_results: list = []

    for variant in query_variants:
        # MMR vector search (diversity-aware)
        try:
            mmr_results = vector_manager.db.max_marginal_relevance_search(
                variant,
                k=fetch_k,
                fetch_k=fetch_k * 2,
                filter=filter_dict
            )
        except Exception:
            mmr_results = vector_manager.db.similarity_search(
                variant, k=fetch_k, filter=filter_dict
            )

        # BM25 keyword search
        bm25_results = _bm25_search(variant, topic, fetch_k)

        # Fuse both ranked lists
        fused = _reciprocal_rank_fusion([mmr_results, bm25_results])
        all_child_results.extend(fused)

    # Deduplicate children across variants
    seen_hashes: set = set()
    unique_children: list = []
    for doc in all_child_results:
        h = _compute_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_children.append(doc)

    # Expand to parent context
    parent_docs = _fetch_parent_documents(unique_children)
    if not parent_docs:
        parent_docs = unique_children  # fallback for legacy (non-parent-child) DBs

    # Cross-encoder reranking
    if use_rerank and len(parent_docs) > k:
        return _cross_encode_rerank(query, parent_docs, k)

    return parent_docs[:k]


# ---------------------------------------------------------------------------
# Admin utilities
# ---------------------------------------------------------------------------

def reset_vector_db():
    vector_manager.db.delete_collection()
    if os.path.exists(PARENT_STORE_PATH):
        os.remove(PARENT_STORE_PATH)
    vector_manager.reset_connection()
    return True


def get_all_topics():
    collection = vector_manager.db._collection

    if collection.count() == 0:
        return []

    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas", [])

    topics = set()
    for meta in metadatas:
        if meta and "topic" in meta:
            topics.add(meta["topic"])

    return sorted(list(topics))
