"""
chunking.py
-----------
Fragmenta los Documents narrativos (txt/pdf) en chunks aptos para embeddings,
preservando la metadata original (categoria, tipo_archivo, source) en cada
fragmento resultante. Las filas de CSV NO se fragmentan: ya son unidades
atómicas de información desde la Fase 1 (ver src/ingestion/loaders.py).
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Tipos de archivo que SÍ se fragmentan. CSV se excluye deliberadamente:
# cada fila ya es una unidad semántica completa generada en loaders.py.
TIPOS_FRAGMENTABLES = {"txt", "pdf"}

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Aplica fragmentación condicional:
    - txt/pdf -> se dividen con RecursiveCharacterTextSplitter
    - csv     -> pasan sin modificar (ya son atómicos)

    Args:
        documents: lista de Document provenientes de run_ingestion().
        chunk_size: tamaño objetivo de cada fragmento en caracteres.
        chunk_overlap: superposición entre fragmentos consecutivos.

    Returns:
        Lista consolidada de Document listos para generar embeddings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # prioriza cortes por párrafo/oración
    )

    fragmentables = [d for d in documents if d.metadata.get("tipo_archivo") in TIPOS_FRAGMENTABLES]
    no_fragmentables = [d for d in documents if d.metadata.get("tipo_archivo") not in TIPOS_FRAGMENTABLES]

    chunks = splitter.split_documents(fragmentables) if fragmentables else []

    # Metadata de trazabilidad: útil para depurar calidad de retrieval en Fase 2
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    resultado = chunks + no_fragmentables

    logger.info(
        f"Chunking finalizado: {len(fragmentables)} doc/s narrativos -> {len(chunks)} chunk/s "
        f"| {len(no_fragmentables)} doc/s (csv) sin modificar "
        f"| total corpus final: {len(resultado)}"
    )

    return resultado


if __name__ == "__main__":
    # Permite correr `python -m src.ingestion.chunking` para inspección rápida
    from src.ingestion.pipeline import run_ingestion

    docs = run_ingestion()
    chunks = chunk_documents(docs)

    print(f"\nDocumentos originales: {len(docs)}")
    print(f"Fragmentos finales (post-chunking): {len(chunks)}")
    if chunks:
        muestra = chunks[0]
        print("\n--- Muestra de un fragmento ---")
        print(f"Contenido ({len(muestra.page_content)} chars): {muestra.page_content[:200]}...")
        print(f"Metadata: {muestra.metadata}")
