"""
vectorstore.py
--------------
Construccion, persistencia y carga del indice vectorial FAISS.

Diseno clave: la construccion (build_vectorstore) es una operacion offline,
costosa, que se corre desde scripts/build_index.py. La carga (load_vectorstore)
es la operacion runtime, rapida, que usa la API en cada arranque.

RESILIENCIA ANTE RATE LIMITS (free tier de Gemini):
El free tier de embeddings tiene un limite bajo de requests por minuto.
Con corpus de cientos de fragmentos, generar embeddings uno tras otro sin
control dispara errores 429 (RESOURCE_EXHAUSTED) a mitad de camino.

Estrategia de tres capas:
1. Procesamiento por lotes pequenos con pausa proactiva entre ellos
   (evita disparar el limite en primer lugar, en vez de solo reaccionar).
2. Reintentos con backoff exponencial ante 429, sin reintentar otros errores.
3. Checkpointing: el indice se guarda en disco tras cada lote exitoso, y si
   el proceso se corta, la siguiente corrida retoma desde el ultimo lote
   guardado en vez de volver a embeber todo desde cero.
"""

import json
import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.rag.embeddings import get_embeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 10
DEFAULT_DELAY_BETWEEN_BATCHES_SECONDS = 8.0
CHECKPOINT_FILENAME = "_checkpoint.json"


def _es_error_de_rate_limit(exc: BaseException) -> bool:
    """
    Determina si una excepcion corresponde a un 429/RESOURCE_EXHAUSTED de
    la API de Gemini. Solo reintentamos ESTE tipo de error: otros errores
    (ej. API key invalida, modelo inexistente) deben fallar de inmediato
    en vez de reintentarse ciegamente 6 veces perdiendo tiempo.
    """
    mensaje = str(exc).lower()
    return (
        "429" in mensaje
        or "resource_exhausted" in mensaje
        or "rate limit" in mensaje
        or "quota" in mensaje
    )


@retry(
    retry=retry_if_exception(_es_error_de_rate_limit),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=90),
    reraise=True,
)
def _embed_batch_con_reintentos(
    vectorstore: Optional[FAISS], batch: List[Document], embeddings
) -> FAISS:
    """
    Embebe un lote de documentos. Si vectorstore es None, crea uno nuevo
    (primer lote); si ya existe, agrega el lote al indice existente.
    Reintenta con backoff exponencial SOLO ante errores de rate limit.
    """
    if vectorstore is None:
        return FAISS.from_documents(batch, embeddings)
    vectorstore.add_documents(batch)
    return vectorstore


def _cargar_checkpoint(checkpoint_path: Path) -> int:
    """Retorna cuantos documentos ya fueron indexados en una corrida anterior."""
    if not checkpoint_path.exists():
        return 0
    try:
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        return data.get("documentos_procesados", 0)
    except (json.JSONDecodeError, OSError):
        logger.warning("Checkpoint corrupto o ilegible, se ignora y se reinicia desde 0")
        return 0


def _guardar_checkpoint(checkpoint_path: Path, documentos_procesados: int) -> None:
    checkpoint_path.write_text(
        json.dumps({"documentos_procesados": documentos_procesados}), encoding="utf-8"
    )


def build_vectorstore(
    documents: List[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_BETWEEN_BATCHES_SECONDS,
    vectorstore_path: Optional[Path] = None,
) -> FAISS:
    """
    Construye (o retoma) un indice FAISS a partir de los chunks del corpus,
    procesando en lotes pequenos con checkpointing para ser resiliente a
    los limites del free tier de Gemini.

    Si una corrida anterior fue interrumpida, esta funcion detecta el
    checkpoint guardado y continua desde el primer documento no procesado,
    en vez de volver a gastar cuota re-embebiendo documentos ya indexados.

    Args:
        documents: chunks del corpus (deben venir en orden deterministico).
        batch_size: cuantos documentos se embeben por llamada a la API.
        delay_seconds: pausa proactiva entre lotes para no saturar el RPM.
        vectorstore_path: donde se persiste el indice incrementalmente.

    Returns:
        El FAISS vectorstore completo, con todos los documentos indexados.
    """
    if not documents:
        raise ValueError("No se puede construir un vectorstore sin documentos")

    target_path = vectorstore_path or settings.VECTORSTORE_DIR
    target_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target_path.parent / CHECKPOINT_FILENAME

    embeddings = get_embeddings()

    inicio = _cargar_checkpoint(checkpoint_path)
    vectorstore: Optional[FAISS] = None

    if inicio > 0 and target_path.exists():
        logger.info(
            f"Checkpoint encontrado: retomando desde el documento {inicio}/{len(documents)} "
            f"(evita re-embeber lo ya indexado)"
        )
        vectorstore = FAISS.load_local(
            str(target_path), embeddings, allow_dangerous_deserialization=True
        )
    elif inicio > 0:
        logger.warning(
            "Checkpoint indica progreso previo pero no se encontró el índice en disco. "
            "Reiniciando desde 0."
        )
        inicio = 0

    pendientes = documents[inicio:]
    if not pendientes:
        logger.info("Todos los documentos ya estaban indexados según el checkpoint.")
        return vectorstore  # type: ignore[return-value]

    total_lotes = (len(pendientes) + batch_size - 1) // batch_size
    logger.info(
        f"Indexando {len(pendientes)} documento/s restantes en {total_lotes} lote/s "
        f"de tamaño {batch_size} (pausa de {delay_seconds}s entre lotes)"
    )

    procesados = inicio
    for i in range(0, len(pendientes), batch_size):
        lote = pendientes[i : i + batch_size]
        numero_lote = (i // batch_size) + 1

        try:
            vectorstore = _embed_batch_con_reintentos(vectorstore, lote, embeddings)
        except Exception as exc:
            # Agotamos los reintentos o fue un error no relacionado a rate limit.
            # Guardamos el progreso hecho hasta ahora antes de propagar el error,
            # así la próxima corrida retoma desde aquí en vez de perder todo.
            if vectorstore is not None:
                save_vectorstore(vectorstore, target_path)
                _guardar_checkpoint(checkpoint_path, procesados)
            logger.error(
                f"Fallo irrecuperable en lote {numero_lote}/{total_lotes} tras reintentos. "
                f"Progreso guardado: {procesados} documento/s. Error: {exc}"
            )
            raise

        procesados += len(lote)
        save_vectorstore(vectorstore, target_path)
        _guardar_checkpoint(checkpoint_path, procesados)

        logger.info(
            f"Lote {numero_lote}/{total_lotes} indexado | "
            f"progreso total: {procesados}/{len(documents)}"
        )

        # Pausa proactiva entre lotes (no aplica tras el último lote)
        if i + batch_size < len(pendientes):
            time.sleep(delay_seconds)

    # Indexación completa: limpiamos el checkpoint para que la próxima
    # ejecución de build_index (con documentos nuevos) empiece limpia.
    checkpoint_path.unlink(missing_ok=True)
    logger.info("Índice FAISS construido y guardado exitosamente (checkpoint limpiado)")

    return vectorstore  # type: ignore[return-value]


def save_vectorstore(vectorstore: FAISS, path: Optional[Path] = None) -> None:
    """Persiste el indice FAISS en disco (carpeta con index.faiss + index.pkl)."""
    target = path or settings.VECTORSTORE_DIR
    target.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target))


def load_vectorstore(path: Optional[Path] = None) -> FAISS:
    """
    Carga un indice FAISS previamente construido. Requiere
    allow_dangerous_deserialization=True porque FAISS usa pickle
    internamente. Es seguro aqui porque el indice lo generamos
    nosotros mismos (no proviene de una fuente externa no confiable).
    """
    target = path or settings.VECTORSTORE_DIR

    if not target.exists():
        raise FileNotFoundError(
            f"No existe un índice FAISS en {target}. "
            f"Ejecuta primero: python -m scripts.build_index"
        )

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(target), embeddings, allow_dangerous_deserialization=True
    )
    logger.info(f"Índice FAISS cargado desde: {target}")

    return vectorstore
