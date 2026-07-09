"""
build_index.py
---------------
Script de indexacion. Por defecto corre en modo INCREMENTAL: detecta que
archivos de Documentacion/ son nuevos o cambiaron (por hash) y solo
embebe esos, dejando intacto (sin gastar cuota de API) todo lo que ya
estaba indexado.

Uso:
    python -m scripts.build_index              # incremental (recomendado)
    python -m scripts.build_index --rebuild     # reconstruccion completa desde cero

Se corre manualmente cada vez que se agregan/modifican documentos en
Documentacion/. En Fase 4 se integrara opcionalmente a un job de CI/CD.
"""

import argparse
import sys
import time

from src.config import settings
from src.ingestion.chunking import chunk_documents
from src.ingestion.pipeline import run_ingestion
from src.rag.vectorstore import build_vectorstore, save_vectorstore, sync_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexación del corpus de Alura Agente")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Fuerza reconstrucción completa desde cero, ignorando el manifest existente.",
    )
    args = parser.parse_args()

    inicio = time.time()

    try:
        settings.validate()
    except (EnvironmentError, FileNotFoundError) as exc:
        logger.error(f"Configuración inválida, abortando indexación: {exc}")
        sys.exit(1)

    modo = "RECONSTRUCCIÓN COMPLETA" if args.rebuild else "INCREMENTAL"
    logger.info(f"=== Iniciando indexación (modo: {modo}) ===")

    documentos = run_ingestion()
    if not documentos:
        logger.error("La ingesta no produjo documentos. Verifica Documentacion/. Abortando.")
        sys.exit(1)

    chunks = chunk_documents(documentos)

    if args.rebuild:
        vectorstore = build_vectorstore(chunks)
    else:
        vectorstore = sync_vectorstore(chunks)

    save_vectorstore(vectorstore)

    duracion = time.time() - inicio
    logger.info(
        f"=== Indexación completada en {duracion:.1f}s | modo: {modo} | "
        f"{len(documentos)} documentos -> {len(chunks)} fragmentos en el corpus ==="
    )


if __name__ == "__main__":
    main()
