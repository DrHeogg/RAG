from typing import List
from llama_index.core import Document, VectorStoreIndex
from vector_store import get_index

def ingest_docs(documents: List[Document]) -> None:
    """
    Добавляет документы в существующую коллекцию Qdrant.
    В LlamaIndex 0.11.x используем повторное создание индекса
    поверх того же storage_context (данные просто дозапишутся
    в тот же vector store).
    """
    base_index = get_index()  # индекс, связанный с вашей коллекцией Qdrant
    VectorStoreIndex.from_documents(
        documents,
        storage_context=base_index.storage_context,
        # service_context можно не передавать, если вы настраивали Settings.embed_model глобально
        # service_context=base_index.service_context,
    )
