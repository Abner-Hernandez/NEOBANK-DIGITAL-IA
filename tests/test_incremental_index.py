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


def test_sync_bootstraps_existing_index_without_manifest(tmp_path, monkeypatch):
    """
    Reproduce el bug real: un índice ya existía en disco (construido con
    código viejo, sin manifest), y luego se corre sync_vectorstore por
    primera vez. Los archivos YA indexados NO deben duplicarse — deben
    reconocerse via el propio vectorstore y solo registrarse en el manifest.
    """
    vs_path = tmp_path / "vectorstore" / "faiss_index"

    archivo1 = tmp_path / "doc1.txt"
    archivo1.write_text("Contenido ya indexado previamente", encoding="utf-8")
    chunk = _make_doc(archivo1, "Contenido ya indexado previamente")

    # Simula un índice preexistente construido SIN pasar por sync (sin manifest)
    embeddings = DeterministicFakeEmbedding(size=64)
    from langchain_community.vectorstores import FAISS as FaissDirect

    indice_viejo = FaissDirect.from_documents([chunk], embeddings)
    vs_path.parent.mkdir(parents=True, exist_ok=True)
    indice_viejo.save_local(str(vs_path))
    # deliberadamente NO se crea _manifest.json, para reproducir el bug

    # Ahora corremos sync_vectorstore con ese mismo archivo (sin cambios reales)
    vs = vectorstore_module.sync_vectorstore(
        [chunk], vectorstore_path=vs_path, batch_size=10, delay_seconds=0
    )

    assert len(vs.docstore._dict) == 1  # NO debe duplicarse a 2

    manifest = load_manifest(vs_path.parent / "_manifest.json")
    assert str(archivo1) in manifest  # se registró retroactivamente
