"""
main.py
-------
API FastAPI de Alura Agente. Expone la cadena RAG (Fase 2.3) como endpoints
REST para que la app web pueda consumirlos.

Diseno clave de arranque: el indice FAISS se carga UNA SOLA VEZ al iniciar
el proceso (via lifespan), no en cada request. Si el indice no existe o
falla al cargar, la API sigue arrancando (para que /health sea consultable
y reporte el problema), pero /query y /categories responden 503 hasta que
se resuelva.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CategoriesResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from src.config import settings
from src.rag.chain import build_rag_chain
from src.rag.retriever import get_available_categories
from src.rag.vectorstore import load_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el índice FAISS una sola vez, al arrancar el proceso."""
    try:
        settings.validate()
        app.state.vectorstore = load_vectorstore()
        logger.info("Vectorstore cargado exitosamente al iniciar la API")
    except Exception as exc:
        logger.error(f"No se pudo cargar el vectorstore al iniciar: {exc}")
        app.state.vectorstore = None

    yield

    logger.info("Apagando Alura Agente API")


app = FastAPI(
    title="Alura Agente API",
    description="API RAG para consultar documentos internos en lenguaje natural.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: en producción, define ALLOWED_ORIGINS en .env con el dominio real
# de tu app web (ej. "https://miapp.com"). Por defecto "*" para desarrollo local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_vectorstore():
    """Dependencia interna: valida que el índice esté cargado antes de servir la request."""
    if app.state.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El índice no está disponible. Verifica que vectorstore/ exista "
                "(corre `python -m scripts.build_index`) y reinicia la API."
            ),
        )
    return app.state.vectorstore


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health() -> HealthResponse:
    """Verifica que la API esté viva y si el índice está cargado correctamente."""
    return HealthResponse(
        status="ok",
        app_env=settings.APP_ENV,
        index_loaded=app.state.vectorstore is not None,
    )


@app.get("/categories", response_model=CategoriesResponse, tags=["Consulta"])
def categories() -> CategoriesResponse:
    """Lista las categorías disponibles en el corpus indexado."""
    vectorstore = _require_vectorstore()
    return CategoriesResponse(categories=get_available_categories(vectorstore))


@app.post("/query", response_model=QueryResponse, tags=["Consulta"])
def query(payload: QueryRequest) -> QueryResponse:
    """
    Responde una pregunta en lenguaje natural basándose en los documentos
    internos indexados, opcionalmente restringida a una categoría.
    """
    vectorstore = _require_vectorstore()

    try:
        chain = build_rag_chain(vectorstore, k=payload.k, categoria=payload.categoria)
        resultado = chain.invoke(payload.question)
    except ValueError as exc:
        # Ej.: categoría inexistente (ver src.rag.retriever.get_retriever)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Error al procesar consulta '{payload.question}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error interno al procesar la consulta."
        )

    fuentes = [
        SourceInfo(
            source=doc.metadata.get("source", "desconocida"),
            categoria=doc.metadata.get("categoria", "general"),
        )
        for doc in resultado["context"]
    ]

    return QueryResponse(
        question=resultado["question"], answer=resultado["answer"], sources=fuentes
    )
