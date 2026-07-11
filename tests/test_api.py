"""
test_api.py
-----------
Valida los endpoints de la API usando FastAPI TestClient. El vectorstore
se inyecta directamente en app.state (bypaseando el lifespan real, que
requeriría una API key real), usando embeddings falsos deterministas.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from src.api.main import app  # noqa: E402
from src.config import settings  # noqa: E402
import src.api.main as main_module  # noqa: E402
import src.rag.chain as chain_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    documentos = [
        Document(
            page_content="Los empleados tienen 15 días de vacaciones al año.",
            metadata={"categoria": "politicas_internas", "source": "vacaciones.txt"},
        ),
        Document(
            page_content="El producto de ahorro premium ofrece 5% de interés anual.",
            metadata={"categoria": "productos", "source": "ahorro.txt"},
        ),
    ]
    embeddings = DeterministicFakeEmbedding(size=64)
    vectorstore = FAISS.from_documents(documentos, embeddings)

    respuesta_falsa = "Los empleados tienen 15 días de vacaciones al año."
    llm_falso = FakeListChatModel(responses=[respuesta_falsa])
    monkeypatch.setattr(chain_module, "get_llm", lambda temperature=0.2: llm_falso)

    # El lifespan real corre al entrar al TestClient y llamaría a settings.validate()
    # + load_vectorstore() reales (fallarían sin API key real). Los reemplazamos
    # para que el lifespan cargue nuestro vectorstore falso en vez del real.
    monkeypatch.setattr(main_module, "load_vectorstore", lambda: vectorstore)
    monkeypatch.setattr(settings, "validate", lambda: None)

    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_index_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["index_loaded"] is True


def test_categories_returns_distinct_values(client):
    response = client.get("/categories")
    assert response.status_code == 200
    assert response.json()["categories"] == ["politicas_internas", "productos"]


def test_query_returns_answer_and_sources(client):
    response = client.post("/query", json={"question": "¿cuántos días de vacaciones tengo?"})
    assert response.status_code == 200

    body = response.json()
    assert body["question"] == "¿cuántos días de vacaciones tengo?"
    assert "15 días" in body["answer"]
    assert len(body["sources"]) > 0
    assert "source" in body["sources"][0]
    assert "categoria" in body["sources"][0]


def test_query_with_invalid_categoria_returns_400(client):
    response = client.post(
        "/query", json={"question": "algo", "categoria": "categoria_inexistente"}
    )
    assert response.status_code == 400


def test_query_with_valid_categoria_filters_sources(client):
    response = client.post(
        "/query", json={"question": "productos de ahorro", "categoria": "productos"}
    )
    assert response.status_code == 200
    body = response.json()
    for fuente in body["sources"]:
        assert fuente["categoria"] == "productos"


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422  # falla validación de Pydantic (min_length=1)


def test_health_without_index_returns_index_loaded_false(monkeypatch):
    """Si el índice no cargó (ej. falta API key), /health debe reportarlo, no crashear."""

    def _falla_carga():
        raise FileNotFoundError("Índice no encontrado (simulado para el test)")

    monkeypatch.setattr(main_module, "load_vectorstore", _falla_carga)
    monkeypatch.setattr(settings, "validate", lambda: None)

    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["index_loaded"] is False
