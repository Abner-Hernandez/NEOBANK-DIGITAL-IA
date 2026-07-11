# syntax=docker/dockerfile:1

# ============================================================
# Stage 1: builder — instala dependencias Python (incluye
# herramientas de compilación que NO queremos en la imagen final)
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ============================================================
# Stage 2: runtime — imagen final, liviana, sin compiladores
# ============================================================
FROM python:3.12-slim

WORKDIR /app

# libgomp1: requerido en runtime por faiss-cpu (usa OpenMP)
# curl: usado por el HEALTHCHECK para verificar /health
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

# Copiamos únicamente los paquetes ya instalados del stage builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiamos solo el código fuente necesario en runtime (no tests/, no docs/)
COPY src/ ./src/
COPY scripts/ ./scripts/

# Directorios que se montarán como volúmenes en docker-compose, pero deben
# existir con los permisos correctos por si el contenedor corre sin volumen
RUN mkdir -p /app/logs /app/vectorstore /app/Documentacion && \
    chown -R appuser:appuser /app /home/appuser/.local

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
