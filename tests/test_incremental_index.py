"""
test_incremental_index.py
--------------------------
Valida el ciclo completo de indexación incremental: archivo nuevo se
embebe, archivo sin cambios se omite, archivo modificado se re-embebe
(reemplazando sus chunks viejos), y archivo eliminado se borra del índice.
Usa embeddings falsos deterministas — sin llamadas reales a la API.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

import src.rag.vectorstore as vectorstore_module  # noqa: E402
from src.rag.manifest import load_manifest  # noqa: E402


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Reemplaza get_embeddings() en todo el módulo vectorstore por una versión falsa."""
    fake = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(vectorstore_module, "get_embeddings", lambda: fake)


def _make_doc(path: Path, content: str, categoria: str = "general") -> Document:
    return Document(page_content=content, metadata={"source": str(path), "categoria": categoria})


def test_sync_incremental_full_cycle(tmp_path):
    vs_path = tmp_path / "vectorstore" / "faiss_index"

    archivo1 = tmp_path / "doc1.txt"
    archivo2 = tmp_path / "doc2.txt"
    archivo1.write_text("Contenido original del documento 1", encoding="utf-8")
    archivo2.write_text("Contenido del documento 2", encoding="utf-8")

    chunks_v1 = [
        _make_doc(archivo1, "Contenido original del documento 1"),
        _make_doc(archivo2, "Contenido del documento 2"),
    ]

    # --- Paso 1: primera corrida, ambos archivos son nuevos ---
    vs = vectorstore_module.sync_vectorstore(
        chunks_v1, vectorstore_path=vs_path, batch_size=10, delay_seconds=0
    )
    assert len(vs.docstore._dict) == 2

    manifest = load_manifest(vs_path.parent / "_manifest.json")
    assert str(archivo1) in manifest
    assert str(archivo2) in manifest

    # --- Paso 2: segunda corrida SIN cambios -> no debe re-embeber nada ---
    vs2 = vectorstore_module.sync_vectorstore(
        chunks_v1, vectorstore_path=vs_path, batch_size=10, delay_seconds=0
    )
    assert len(vs2.docstore._dict) == 2  # mismo tamaño, nada duplicado

    # --- Paso 3: modificar archivo1 -> debe re-embeberse (reemplazando su chunk viejo) ---
    archivo1.write_text("Contenido MODIFICADO del documento 1", encoding="utf-8")
    chunks_v2 = [
        _make_doc(archivo1, "Contenido MODIFICADO del documento 1"),
        _make_doc(archivo2, "Contenido del documento 2"),
    ]
    vs3 = vectorstore_module.sync_vectorstore(
        chunks_v2, vectorstore_path=vs_path, batch_size=10, delay_seconds=0
    )
    assert len(vs3.docstore._dict) == 2  # sigue siendo 2, no se duplicó
    contenidos = [doc.page_content for doc in vs3.docstore._dict.values()]
    assert "Contenido MODIFICADO del documento 1" in contenidos
    assert "Contenido original del documento 1" not in contenidos  # el viejo ya no está

    # --- Paso 4: eliminar archivo2 del corpus -> debe borrarse del índice ---
    chunks_v3 = [_make_doc(archivo1, "Contenido MODIFICADO del documento 1")]
    vs4 = vectorstore_module.sync_vectorstore(
        chunks_v3, vectorstore_path=vs_path, batch_size=10, delay_seconds=0
    )
    assert len(vs4.docstore._dict) == 1
    manifest_final = load_manifest(vs_path.parent / "_manifest.json")
    assert str(archivo2) not in manifest_final
    assert str(archivo1) in manifest_final
