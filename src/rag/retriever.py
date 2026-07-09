"""
retriever.py
------------
Configura el retriever sobre el vectorstore FAISS, con soporte opcional de
filtrado por categoria. Util cuando el corpus tiene multiples dominios
tematicos (ej. "politicas_internas", "productos", "soporte") y se quiere
restringir la busqueda a uno solo en vez de buscar en los 415 fragmentos.
"""

from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_K = 4


def get_available_categories(vectorstore: FAISS) -> List[str]:
    """
    Inspecciona el vectorstore y retorna las categorias distintas presentes
    en la metadata de los documentos indexados. Util para validar filtros
    antes de aplicarlos, y para exponer las categorias disponibles en la
    API (Fase 2.4).
    """
    categorias = {
        doc.metadata.get("categoria", "general")
        for doc in vectorstore.docstore._dict.values()  # type: ignore[attr-defined]
    }
    return sorted(categorias)


def get_retriever(
    vectorstore: FAISS,
    k: int = DEFAULT_TOP_K,
    categoria: Optional[str] = None,
) -> VectorStoreRetriever:
    """
    Construye un retriever configurado sobre el vectorstore.

    Args:
        vectorstore: indice FAISS ya cargado (via load_vectorstore()).
        k: cantidad de fragmentos mas relevantes a recuperar por consulta.
        categoria: si se especifica, restringe la busqueda a documentos
            cuya metadata "categoria" coincida exactamente. Debe ser una
            categoria real presente en el corpus (se valida contra
            get_available_categories antes de aplicar el filtro).

    Returns:
        Un VectorStoreRetriever listo para usarse en la cadena RAG.

    Raises:
        ValueError: si `categoria` no existe en el corpus indexado.
    """
    search_kwargs = {"k": k}

    if categoria is not None:
        disponibles = get_available_categories(vectorstore)
        if categoria not in disponibles:
            raise ValueError(
                f"Categoría '{categoria}' no existe en el corpus indexado. "
                f"Categorías disponibles: {disponibles}"
            )
        search_kwargs["filter"] = {"categoria": categoria}
        logger.info(f"Retriever configurado con filtro de categoría: '{categoria}'")
    else:
        logger.info("Retriever configurado sin filtro (busca en todo el corpus)")

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    k: int = DEFAULT_TOP_K,
    categoria: Optional[str] = None,
) -> List[Document]:
    """
    Atajo funcional para recuperar documentos relevantes sin construir
    explícitamente un objeto retriever. Útil para debugging rápido y para
    inspeccionar qué fragmentos se recuperarían antes de pasarlos al LLM.
    """
    retriever = get_retriever(vectorstore, k=k, categoria=categoria)
    return retriever.invoke(query)


if __name__ == "__main__":
    # Inspección rápida: `python -m src.rag.retriever "tu pregunta aquí"`
    import sys

    from src.rag.vectorstore import load_vectorstore

    if len(sys.argv) < 2:
        print('Uso: python -m src.rag.retriever "tu pregunta"')
        sys.exit(1)

    pregunta = sys.argv[1]
    vs = load_vectorstore()

    print(f"\nCategorías disponibles: {get_available_categories(vs)}\n")

    resultados = retrieve_documents(vs, pregunta)
    print(f"Fragmentos recuperados para: '{pregunta}'\n")
    for i, doc in enumerate(resultados, 1):
        print(f"--- Fragmento {i} (categoria={doc.metadata.get('categoria')}) ---")
        print(doc.page_content[:200])
        print()
