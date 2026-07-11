"""
schemas.py
----------
Modelos Pydantic de request/response de la API. Separados del código de
rutas para que el "contrato" de la API sea explícito y fácil de consumir
desde la documentación automática de FastAPI (/docs).
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, description="Pregunta en lenguaje natural sobre los documentos internos"
    )
    categoria: Optional[str] = Field(
        None, description="Si se especifica, restringe la búsqueda a esta categoría"
    )
    k: int = Field(
        4, ge=1, le=20, description="Cantidad de fragmentos a recuperar como contexto"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "¿cuál es la política de vacaciones?"},
                {
                    "question": "¿qué producto de ahorro ofrecen?",
                    "categoria": "productos",
                    "k": 3,
                },
            ]
        }
    }


class SourceInfo(BaseModel):
    source: str = Field(..., description="Ruta del archivo fuente usado para la respuesta")
    categoria: str = Field(..., description="Categoría del documento fuente")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]


class CategoriesResponse(BaseModel):
    categories: List[str]


class HealthResponse(BaseModel):
    status: str
    app_env: str
    index_loaded: bool
