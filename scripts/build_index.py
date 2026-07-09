"""
build_index.py
---------------
Script de indexación offline. Conecta todo el pipeline de Fase 1 (ingesta +
chunking) con la construcción del índice vectorial FAISS.

Uso:
    python -m scripts.build_index

Este script se corre manualmente cada vez que se agregan/modifican documentos
en Documentacion/. En Fase 4 lo integraremos opcionalmente a un job de CI/CD.
"""

import sys
import time

from src.config import settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.pipeline import run_ingestion
from src.rag.vectorstore import build_vectorstore, save_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    inicio = time.time()

    try:
        settings.validate()
    except (EnvironmentError, FileNotFoundError) as exc:
        logger.error(f"Configuración inválida, abortando indexación: {exc}")
        sys.exit(1)

    logger.info("=== Iniciando construcción del índice ===")

    documentos = run_ingestion()
    if not documentos:
        logger.error("La ingesta no produjo documentos. Verifica Documentacion/. Abortando.")
        sys.exit(1)

    chunks = chunk_documents(documentos)

    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)

    duracion = time.time() - inicio
    logger.info(
        f"=== Índice construido exitosamente en {duracion:.1f}s | "
        f"{len(documentos)} documentos -> {len(chunks)} fragmentos indexados ==="
    )


if __name__ == "__main__":
    main()
