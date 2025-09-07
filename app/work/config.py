import os
from dotenv import load_dotenv

# для контейнера путь к .env может быть таким:
load_dotenv("/home/app/work/.env")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")

# Эмбеддинги (должны совпадать с тем, что использовалось при индексации)
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-base")
EMB_DEVICE = os.getenv("EMB_DEVICE", "cpu")

# Поиск
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))  
MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "6000"))  

# YaGPT (Yandex Cloud)
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
YANDEX_SERV_API_KEY = os.getenv("YANDEX_SERV_API_KEY")
YANDEX_MODEL = os.getenv("YANDEX_MODEL", "yandexgpt")  

# Чат
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "5"))  
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты ассистент, отвечающий по предоставленному контексту. Если в контексте ответа нет, скажи об этом."
)
