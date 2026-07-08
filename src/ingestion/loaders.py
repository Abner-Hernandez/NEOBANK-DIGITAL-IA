"""
loaders.py
----------
Un loader especializado por tipo de archivo. Todos retornan una lista de
langchain_core.documents.Document, para que el resto del pipeline (chunking,
embeddings, vectorstore) sea agnóstico al formato original.

Cada Document lleva metadata enriquecida (source, categoria, tipo_archivo)
para poder citar la fuente exacta cuando el agente responda.
"""

from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _categoria_desde_ruta(filepath: Path, raw_root: Path) -> str:
    """
    Deriva la categoría del documento a partir de la subcarpeta donde vive,
    ej. Documentacion/politicas_internas/doc1.pdf -> categoria = "politicas_internas".
    Si el archivo está en la raíz de Documentacion/, se marca como "general".
    """
    try:
        relative = filepath.relative_to(raw_root)
        return relative.parts[0] if len(relative.parts) > 1 else "general"
    except ValueError:
        return "general"


def load_txt(filepath: Path, raw_root: Path) -> List[Document]:
    """Carga un archivo .txt como un único Document con metadata de categoría."""
    loader = TextLoader(str(filepath), encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["categoria"] = _categoria_desde_ruta(filepath, raw_root)
        doc.metadata["tipo_archivo"] = "txt"

    logger.info(f"TXT cargado: {filepath.name} ({len(docs)} documento/s)")
    return docs


def load_pdf(filepath: Path, raw_root: Path) -> List[Document]:
    """
    Carga un .pdf con PyPDFLoader. Genera un Document por página, lo cual
    es ideal para citar "página X del documento Y" en las respuestas del agente.
    """
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()

    for doc in docs:
        doc.metadata["categoria"] = _categoria_desde_ruta(filepath, raw_root)
        doc.metadata["tipo_archivo"] = "pdf"

    logger.info(f"PDF cargado: {filepath.name} ({len(docs)} página/s)")
    return docs


def load_csv(filepath: Path, raw_root: Path) -> List[Document]:
    """
    Carga un .csv narrativo con pandas. Asume que el contenido relevante
    es texto libre distribuido en columnas; concatena todas las columnas
    de texto de cada fila en un único Document para preservar contexto.

    NOTA: si tus CSVs tienen una estructura fija conocida (ej. siempre
    columnas "pregunta,respuesta"), en Fase 1.1 ajustamos esta función
    para un tratamiento más preciso por columna.
    """
    df = pd.read_csv(filepath, encoding="utf-8")
    docs: List[Document] = []
    categoria = _categoria_desde_ruta(filepath, raw_root)

    for idx, row in df.iterrows():
        contenido = "\n".join(f"{col}: {val}" for col, val in row.items())
        docs.append(
            Document(
                page_content=contenido,
                metadata={
                    "source": str(filepath),
                    "fila": idx,
                    "categoria": categoria,
                    "tipo_archivo": "csv",
                },
            )
        )

    logger.info(f"CSV cargado: {filepath.name} ({len(docs)} fila/s)")
    return docs


# Mapa extensión -> función loader. Fácil de extender (ej. .docx en el futuro).
LOADER_REGISTRY = {
    ".txt": load_txt,
    ".pdf": load_pdf,
    ".csv": load_csv,
}
