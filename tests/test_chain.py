"""
test_chain.py
--------------
Valida la cadena RAG completa (retriever -> prompt -> LLM) usando un LLM
falso determinista y embeddings falsos, sin llamadas reales a Gemini.
Esto prueba la PLOMERÍA de la cadena (que el contexto llegue al prompt,
que la respuesta y las fuentes se retornen correctamente), no la calidad
de las respuestas del LLM real (eso se valida manualmente con tu API key).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

import src.rag.chain as chain_module  # noqa: E402


@pytest.fixture
def fake_vectorstore() -> FAISS:
    documentos = [
        Document(
            page_content="Los empleados tienen 15 días de vacaciones al año.",
            metadata={
                "categoria": "politicas_internas",
                "source": "Documentacion/politicas_internas/vacaciones.txt",
            },
        ),
        Document(
            page_content="El producto de ahorro premium ofrece 5% de interés anual.",
            metadata={"categoria": "productos", "source": "Documentacion/productos/ahorro.txt"},
        ),
    ]
    embeddings = DeterministicFakeEmbedding(size=64)
    return FAISS.from_documents(documentos, embeddings)


@pytest.fixture
def fake_llm(monkeypatch):
    """Reemplaza get_llm() por un modelo falso con respuesta fija y determinista."""
    respuesta_esperada = "Según los documentos, los empleados tienen 15 días de vacaciones."
    llm_falso = FakeListChatModel(responses=[respuesta_esperada])
    monkeypatch.setattr(chain_module, "get_llm", lambda temperature=0.2: llm_falso)
    return respuesta_esperada


def test_ask_returns_answer_question_and_context(fake_vectorstore, fake_llm):
    resultado = chain_module.ask(fake_vectorstore, "¿cuántos días de vacaciones tengo?")

    assert resultado["question"] == "¿cuántos días de vacaciones tengo?"
    assert resultado["answer"] == fake_llm
    assert len(resultado["context"]) > 0
    assert all(isinstance(doc, Document) for doc in resultado["context"])


def test_ask_with_categoria_filter_restricts_context(fake_vectorstore, fake_llm):
    resultado = chain_module.ask(
        fake_vectorstore, "cuéntame sobre productos", categoria="productos"
    )

    assert len(resultado["context"]) == 1
    assert resultado["context"][0].metadata["categoria"] == "productos"


def test_format_docs_includes_source_and_categoria():
    docs = [
        Document(
            page_content="Contenido de prueba.",
            metadata={"source": "archivo.txt", "categoria": "general"},
        )
    ]
    texto = chain_module._format_docs(docs)

    assert "archivo.txt" in texto
    assert "general" in texto
    assert "Contenido de prueba." in texto
