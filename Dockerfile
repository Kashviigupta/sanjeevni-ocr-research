FROM python:3.11-slim

# Tesseract is a SYSTEM binary, not a pip package. hin = Devanagari, needed for
# documents from government facilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-hin libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Built from the repo root, not ml/, because the clinical ontology in shared/ is not
# optional: the dialogue engine refuses to start without it. A context of ml/ cannot
# reach it, which is exactly how this image shipped without one.
WORKDIR /app
COPY ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/pyproject.toml .
COPY ml/src/ src/
COPY ml/data/ data/
COPY shared/ shared/
RUN pip install --no-cache-dir -e .

# Stateless service, no database, nothing persisted - so nothing here can leak a record.
EXPOSE 8100
CMD ["uvicorn", "sanjeevani_ml.service.main:app", "--host", "0.0.0.0", "--port", "8100"]
