RAG App (LlamaIndex + Qdrant + YaGPT)

Retrieval-Augmented Generation: индексируем локальные документы (PDF/DOCX/TXT/сканы), храним эмбеддинги в Qdrant, достаём top-k релевантных фрагментов и генерируем ответ через YaGPT, учитывая историю диалога.

Что внутри

Ингест: парсинг (включая OCR), чанкинг, эмбеддинги (sentence-transformers), запись в Qdrant.

Ретраивер: top-k по косинусной близости, порог score_threshold, фильтры по метаданным.

Генерация: YaGPT получает вопрос + контекст + сжатую историю чата.

Docker: app (Python) + qdrant. Конфиг — через .env.

Стек

LlamaIndex · Qdrant · sentence-transformers · PyPDF2/pdfplumber/pytesseract · YaGPT
