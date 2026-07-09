"""
test_loaders.py
----------------
Smoke tests de la Fase 1. Validan que la ingesta corre sin errores
y que la metadata se asigna correctamente. No son tests exhaustivos
de contenido (eso depende de tus documentos reales), sino de que el
pipeline no se rompe silenciosamente.
"""

import sys
from pathlib import Path

# Permite correr pytest desde la raíz del proyecto sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.pipeline import run_ingestion  # noqa: E402


def test_run_ingestion_returns_documents():
    """La ingesta debe retornar al menos un Document del archivo de prueba."""
    docs = run_ingestion()
    assert len(docs) > 0, "La ingesta no generó ningún documento"


def test_documents_have_required_metadata():
    """Cada Document debe tener categoria y tipo_archivo en su metadata."""
    docs = run_ingestion()
    for doc in docs:
        assert "categoria" in doc.metadata
        assert "tipo_archivo" in doc.metadata


def test_categoria_derivada_de_subcarpeta():
    """Un archivo en data/raw/politicas_internas/ debe tener esa categoría."""
    docs = run_ingestion()
    categorias = {doc.metadata["categoria"] for doc in docs}
    assert "politicas_internas" in categorias
