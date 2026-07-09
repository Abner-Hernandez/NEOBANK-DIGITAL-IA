"""
test_retriever.py
------------------
Valida la lógica de filtrado por categoría del retriever usando un
FAISS índice construido con embeddings falsos deterministas (sin llamadas
reales a la API de Gemini). Esto nos permite probar la lógica de negocio
(filtro, validación, top-k) de forma rápida, gratuita y determinista.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from src.rag.retriever import get_available_categories, get_retriever, retrieve_documents  # noqa: E402


@pytest.fixture
def fake_vectorstore() -> FAISS:
    """Construye un FAISS pequeño con 2 categorías, sin llamadas a red."""
    documentos = [
        Document(
            page_content="Los empleados tienen 15 días de vacaciones al año.",
            metadata={"categoria": "politicas_internas", "source": "vacaciones.txt"},
        ),
        Document(
            page_content="El horario de oficina es de 9am a 6pm de lunes a viernes.",
            metadata={"categoria": "politicas_internas", "source": "horario.txt"},
        ),
        Document(
            page_content="El producto de ahorro premium ofrece 5% de interés anual.",
            metadata={"categoria": "productos", "source": "ahorro.txt"},
        ),
    ]
    embeddings = DeterministicFakeEmbedding(size=64)
    return FAISS.from_documents(documentos, embeddings)


def test_get_available_categories_returns_all_distinct(fake_vectorstore):
    categorias = get_available_categories(fake_vectorstore)
    assert categorias == ["politicas_internas", "productos"]


def test_retriever_without_filter_searches_full_corpus(fake_vectorstore):
    resultados = retrieve_documents(fake_vectorstore, "vacaciones", k=3)
    assert len(resultados) == 3  # ve los 3 documentos, sin restricción


def test_retriever_with_valid_categoria_filters_results(fake_vectorstore):
    resultados = retrieve_documents(
        fake_vectorstore, "información", k=3, categoria="politicas_internas"
    )
    assert len(resultados) == 2  # solo los 2 de esa categoría
    for doc in resultados:
        assert doc.metadata["categoria"] == "politicas_internas"


def test_retriever_with_invalid_categoria_raises_value_error(fake_vectorstore):
    with pytest.raises(ValueError, match="no existe en el corpus"):
        get_retriever(fake_vectorstore, categoria="categoria_inexistente")
