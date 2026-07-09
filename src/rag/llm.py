"""
llm.py
------
Factory del modelo de chat Gemini. Aislado en su propio modulo por la misma
razon que embeddings.py: si el nombre del modelo vuelve a cambiar (como ya
paso con gemini-2.0-flash, discontinuado el 1-jun-2026), solo se toca aqui.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Retorna una instancia configurada del modelo de chat Gemini.

    temperature=0.2 por defecto: para un agente de respuestas basadas en
    documentos internos, priorizamos respuestas consistentes y ancladas
    a las fuentes por encima de creatividad/variabilidad.
    """
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
    )
