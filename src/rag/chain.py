"""
chain.py
--------
Cadena RAG completa usando LCEL (LangChain Expression Language): conecta
el retriever (Fase 2.2) con el prompt y el LLM Gemini (Fase 2.3).

Diseno: la cadena retorna tanto la respuesta como los documentos fuente
usados para generarla (via RunnableParallel), no solo el texto. Esto es
clave para la API (Fase 2.4): el frontend podra mostrar "fuente: politica_
vacaciones.txt, categoria: politicas_internas" junto a cada respuesta,
en vez de una respuesta sin trazabilidad.
"""

from typing import List, Optional, TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.rag.llm import get_llm
from src.rag.retriever import DEFAULT_TOP_K, get_retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """Eres el asistente virtual de Alura Agente. Respondes preguntas \
basándote EXCLUSIVAMENTE en el contexto proporcionado, que proviene de documentos \
internos de la organización.

Reglas estrictas:
1. Si la respuesta no está contenida en el contexto, di explícitamente que no \
tienes esa información en los documentos disponibles. NUNCA inventes datos.
2. Responde siempre en español, de forma clara y concisa.
3. Si el contexto contiene información parcial o ambigua, indícalo en vez de \
completar los vacíos con suposiciones.
4. No menciones que eres un modelo de lenguaje ni expliques tu proceso de \
razonamiento; responde de forma directa y natural.

Contexto:
{context}
"""

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


class RAGResponse(TypedDict):
    """Estructura de salida de la cadena RAG."""

    question: str
    answer: str
    context: List[Document]


def _format_docs(docs: List[Document]) -> str:
    """
    Formatea los documentos recuperados como texto plano para el prompt,
    incluyendo su fuente y categoría para que el LLM pueda referenciarlos
    si es relevante (ej. "según la política de vacaciones...").
    """
    bloques = []
    for doc in docs:
        fuente = doc.metadata.get("source", "desconocida")
        categoria = doc.metadata.get("categoria", "general")
        bloques.append(
            f"[Fuente: {fuente} | Categoría: {categoria}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(bloques)


def build_rag_chain(
    vectorstore: FAISS,
    k: int = DEFAULT_TOP_K,
    categoria: Optional[str] = None,
    temperature: float = 0.2,
):
    """
    Construye la cadena RAG completa: retriever -> prompt -> LLM.

    Retorna un Runnable que, al invocarse con un string (la pregunta),
    produce un dict con:
        - "question": la pregunta original
        - "context": lista de Document usados como fuente (para citar)
        - "answer": la respuesta generada por el LLM

    Args:
        vectorstore: índice FAISS ya cargado.
        k: cantidad de fragmentos a recuperar por consulta.
        categoria: si se especifica, restringe la búsqueda a esa categoría.
        temperature: temperatura del LLM (ver src.rag.llm.get_llm).
    """
    retriever = get_retriever(vectorstore, k=k, categoria=categoria)
    llm = get_llm(temperature=temperature)

    cadena_generacion = (
        RunnablePassthrough.assign(context=(lambda x: _format_docs(x["context"])))
        | _prompt
        | llm
        | StrOutputParser()
    )

    cadena_completa = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=cadena_generacion)

    logger.info(
        f"Cadena RAG construida | k={k} | categoria={categoria or 'todas'} | "
        f"temperature={temperature}"
    )

    return cadena_completa


def ask(
    vectorstore: FAISS,
    question: str,
    k: int = DEFAULT_TOP_K,
    categoria: Optional[str] = None,
) -> RAGResponse:
    """Atajo funcional: construye la cadena y responde una pregunta en un solo paso."""
    chain = build_rag_chain(vectorstore, k=k, categoria=categoria)
    return chain.invoke(question)  # type: ignore[return-value]


if __name__ == "__main__":
    # Prueba rápida: `python -m src.rag.chain "tu pregunta aquí"`
    import sys

    from src.rag.vectorstore import load_vectorstore

    if len(sys.argv) < 2:
        print('Uso: python -m src.rag.chain "tu pregunta"')
        sys.exit(1)

    pregunta = sys.argv[1]
    vs = load_vectorstore()

    resultado = ask(vs, pregunta)

    print(f"\nPregunta: {resultado['question']}\n")
    print(f"Respuesta:\n{resultado['answer']}\n")
    print("Fuentes utilizadas:")
    for doc in resultado["context"]:
        print(f"  - {doc.metadata.get('source')} (categoría: {doc.metadata.get('categoria')})")
