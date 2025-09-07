FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    poppler-utils \
    tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /home/app/work

ENV QDRANT_URL=http://qdrant:6333 \
    QDRANT_COLLECTION=docs_mvp \
    EMB_MODEL=intfloat/multilingual-e5-base

EXPOSE 8888

CMD ["jupyter","lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--ServerApp.token=","--ServerApp.password=","--ServerApp.root_dir=/home/app/work"]
