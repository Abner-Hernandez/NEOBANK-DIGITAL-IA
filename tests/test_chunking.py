"""
test_chunking.py
-----------------
Valida la lógica de fragmentación condicional: txt/pdf se dividen,
csv permanece intacto, y la metadata se preserva correctamente.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from src.ingestion.chunking import chunk_documents  # noqa: E402


def test_csv_documents_are_not_split():
    """Un Document de tipo csv debe pasar sin fragmentarse, sin importar su tamaño."""
    doc_csv = Document(
        page_content="col1: valor1\ncol2: " + ("x" * 5000),  # deliberadamente largo
        metadata={"tipo_archivo": "csv", "categoria": "general"},
    )
    resultado = chunk_documents([doc_csv], chunk_size=1000, chunk_overlap=150)

    assert len(resultado) == 1
    assert resultado[0].page_content == doc_csv.page_content


def test_txt_document_is_split_when_long():
    """Un Document txt más largo que chunk_size debe generar múltiples fragmentos."""
    texto_largo = "Esta es una oración de prueba. " * 100  # ~3200 caracteres
    doc_txt = Document(
        page_content=texto_largo,
        metadata={"tipo_archivo": "txt", "categoria": "general", "source": "test.txt"},
    )
    resultado = chunk_documents([doc_txt], chunk_size=1000, chunk_overlap=150)

    assert len(resultado) > 1
    for chunk in resultado:
        assert chunk.metadata["tipo_archivo"] == "txt"
        assert chunk.metadata["categoria"] == "general"
        assert "chunk_id" in chunk.metadata


def test_mixed_corpus_preserves_all_documents():
    """El total de fragmentos + csv intactos nunca debe perder documentos."""
    doc_csv = Document(page_content="fila corta", metadata={"tipo_archivo": "csv"})
    doc_txt_corto = Document(
        page_content="Texto breve que no necesita fragmentarse.",
        metadata={"tipo_archivo": "txt"},
    )
    resultado = chunk_documents([doc_csv, doc_txt_corto])

    assert len(resultado) == 2  # ninguno se pierde, ninguno se fragmenta innecesariamente
