# --- Étape 1 : Builder ---
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances dans un dossier local pour faciliter le transfert
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Étape 2 : Runner ---
FROM python:3.10-slim AS runner

WORKDIR /app


COPY --from=builder /root/.local /root/.local
COPY src/ ./src/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/train.py"]
