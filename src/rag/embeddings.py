"""
embeddings.py
-------------
Factory del modelo de embeddings de Google Gemini. Centralizado en una sola
función para que, si en el futuro cambias de proveedor de embeddings,
solo se toque este archivo — nada más en el proyecto depende de la librería
específica de Google.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Retorna una instancia configurada del modelo de embeddings de Gemini.
    Usa GOOGLE_API_KEY y GEMINI_EMBEDDING_MODEL desde la configuración central.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )
