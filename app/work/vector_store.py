from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from os import getenv
from llama_index.core.settings import Settings
from os import getenv
from dotenv import load_dotenv
load_dotenv("/home/app/work/.env")

QDRANT_URL = getenv("QDRANT_URL")
QDRANT_API_KEY = getenv("QDRANT_API_KEY")
COLLECTION = getenv("QDRANT_COLLECTION", "docs")
EMB_MODEL = getenv("EMB_MODEL", "intfloat/multilingual-e5-base")

assert QDRANT_URL, "QDRANT_URL is not set"
assert COLLECTION, "QDRANT_COLLECTION is not set"
assert EMB_MODEL, "EMB_MODEL is not set"

qdrant_client = QdrantClient(url=QDRANT_URL)
embed_model = HuggingFaceEmbedding(model_name=EMB_MODEL, normalize=True)
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

Settings.embed_model = embed_model

_index = None

def get_index():
    """Возвращает VectorStoreIndex, привязанный к нашей коллекции Qdrant."""
    global _index
    if _index is None:
        _index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
    return _index