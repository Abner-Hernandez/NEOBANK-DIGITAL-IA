"""
pipeline.py
-----------
Orquestador de ingesta. Recorre recursivamente RAW_DATA_DIR (incluyendo
subcarpetas por categoría), detecta la extensión de cada archivo y
despacha al loader correspondiente desde LOADER_REGISTRY.

Diseñado para ser el único punto de entrada de la Fase 1:
    from src.ingestion.pipeline import run_ingestion
    documentos = run_ingestion()
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.config import settings
from src.ingestion.loaders import LOADER_REGISTRY
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_ingestion(raw_dir: Path | None = None) -> List[Document]:
    """
    Recorre el directorio de datos crudos, procesa cada archivo soportado
    y retorna la lista consolidada de Documents listos para chunking/embeddings.

    Args:
        raw_dir: directorio a procesar. Por defecto usa settings.RAW_DATA_DIR.

    Returns:
        Lista de Document de todos los archivos procesados exitosamente.
    """
    raw_root = raw_dir or settings.RAW_DATA_DIR

    if not raw_root.exists():
        raise FileNotFoundError(f"No existe el directorio de datos: {raw_root}")

    all_documents: List[Document] = []
    archivos_no_soportados: List[str] = []
    archivos_con_error: List[str] = []

    archivos = [f for f in raw_root.rglob("*") if f.is_file()]
    logger.info(f"Iniciando ingesta: {len(archivos)} archivo/s encontrado/s en {raw_root}")

    for filepath in archivos:
        extension = filepath.suffix.lower()
        loader_fn = LOADER_REGISTRY.get(extension)

        if loader_fn is None:
            archivos_no_soportados.append(filepath.name)
            logger.warning(f"Extensión no soportada, se omite: {filepath.name}")
            continue

        try:
            documentos = loader_fn(filepath, raw_root)
            all_documents.extend(documentos)
        except Exception as exc:
            archivos_con_error.append(filepath.name)
            logger.error(f"Error procesando {filepath.name}: {exc}", exc_info=True)

    logger.info(
        f"Ingesta finalizada: {len(all_documents)} documento/s generados | "
        f"{len(archivos_no_soportados)} omitido/s | {len(archivos_con_error)} con error"
    )

    if archivos_con_error:
        logger.warning(f"Archivos con error: {archivos_con_error}")

    return all_documents


if __name__ == "__main__":
    # Permite correr `python -m src.ingestion.pipeline` para pruebas rápidas
    docs = run_ingestion()
    print(f"\nTotal de documentos procesados: {len(docs)}")
    if docs:
        print("\n--- Muestra del primer documento ---")
        print(f"Contenido: {docs[0].page_content[:200]}...")
        print(f"Metadata: {docs[0].metadata}")
