"""
vectorstore.py
--------------
Construccion, sincronizacion, persistencia y carga del indice FAISS.

Dos modos de indexacion:
- sync_vectorstore(): INCREMENTAL (recomendado). Compara hashes de archivo
  contra el manifest y solo embebe documentos nuevos o modificados. Es el
  modo por defecto de scripts/build_index.py.
- build_vectorstore(): RECONSTRUCCION COMPLETA desde cero. Util para el
  primer build, o si se quiere forzar un re-index total (--rebuild).

Ambos comparten la misma logica de resiliencia ante rate limits del free
tier de Gemini: procesamiento por lotes pequenos, pausa proactiva entre
lotes, reintentos con backoff exponencial ante 429, y checkpointing en
disco para no perder progreso si el proceso se corta a mitad de camino.
"""

import time
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.config import settings
from src.rag.embeddings import get_embeddings
from src.rag.manifest import (
    MANIFEST_FILENAME,
    compute_file_hash,
    load_manifest,
    save_manifest,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 10
DEFAULT_DELAY_BETWEEN_BATCHES_SECONDS = 8.0
CHECKPOINT_FILENAME = "_checkpoint.json"


def _es_error_de_rate_limit(exc: BaseException) -> bool:
    """Solo reintentamos 429/RESOURCE_EXHAUSTED; otros errores fallan de inmediato."""
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
    """Embebe un lote; crea el vectorstore si no existe aún, o agrega al existente."""
    if vectorstore is None:
        return FAISS.from_documents(batch, embeddings)
    vectorstore.add_documents(batch)
    return vectorstore


def _procesar_lotes(
    vectorstore: Optional[FAISS],
    pendientes: List[Document],
    embeddings,
    target_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    delay_seconds: float,
    ya_procesados: int = 0,
) -> FAISS:
    """
    Núcleo compartido de indexación por lotes con checkpointing. Usado tanto
    por build_vectorstore (corpus completo) como por sync_vectorstore
    (solo documentos nuevos/modificados).
    """
    total_lotes = (len(pendientes) + batch_size - 1) // batch_size
    logger.info(
        f"Indexando {len(pendientes)} documento/s en {total_lotes} lote/s "
        f"de tamaño {batch_size} (pausa de {delay_seconds}s entre lotes)"
    )

    procesados = ya_procesados
    for i in range(0, len(pendientes), batch_size):
        lote = pendientes[i : i + batch_size]
        numero_lote = (i // batch_size) + 1

        try:
            vectorstore = _embed_batch_con_reintentos(vectorstore, lote, embeddings)
        except Exception as exc:
            if vectorstore is not None:
                save_vectorstore(vectorstore, target_path)
            logger.error(
                f"Fallo irrecuperable en lote {numero_lote}/{total_lotes} tras reintentos. "
                f"Progreso guardado: {procesados} documento/s de este batch. Error: {exc}"
            )
            raise

        procesados += len(lote)
        save_vectorstore(vectorstore, target_path)

        logger.info(f"Lote {numero_lote}/{total_lotes} indexado | progreso: {procesados}")

        if i + batch_size < len(pendientes):
            time.sleep(delay_seconds)

    return vectorstore  # type: ignore[return-value]


def _get_ids_for_source(vectorstore: FAISS, source: str) -> List[str]:
    """Encuentra los ids internos de FAISS de todos los chunks de un archivo fuente."""
    return [
        doc_id
        for doc_id, doc in vectorstore.docstore._dict.items()  # type: ignore[attr-defined]
        for _ in [None]
        if doc.metadata.get("source") == source
    ]


def sync_vectorstore(
    chunks: List[Document],
    vectorstore_path: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_BETWEEN_BATCHES_SECONDS,
) -> FAISS:
    """
    Sincroniza el indice FAISS de forma INCREMENTAL: solo embebe documentos
    cuyo archivo fuente es nuevo o cambió (por hash), reutiliza el resto,
    y borra del indice los chunks de archivos que ya no existen.

    Args:
        chunks: TODOS los chunks del corpus actual (salida de chunk_documents,
            sobre el corpus completo — la función internamente decide cuáles
            de estos ya están indexados y cuáles faltan).
        vectorstore_path: ubicación del índice persistido.
        batch_size, delay_seconds: ver _procesar_lotes.

    Returns:
        El FAISS vectorstore actualizado.
    """
    target_path = vectorstore_path or settings.VECTORSTORE_DIR
    target_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target_path.parent / MANIFEST_FILENAME
    checkpoint_path = target_path.parent / CHECKPOINT_FILENAME

    manifest = load_manifest(manifest_path)
    embeddings = get_embeddings()

    vectorstore: Optional[FAISS] = None
    if target_path.exists():
        vectorstore = FAISS.load_local(
            str(target_path), embeddings, allow_dangerous_deserialization=True
        )

    # Agrupar chunks actuales por archivo fuente
    chunks_por_fuente: dict[str, List[Document]] = {}
    for chunk in chunks:
        fuente = chunk.metadata.get("source", "desconocida")
        chunks_por_fuente.setdefault(fuente, []).append(chunk)

    fuentes_actuales = set(chunks_por_fuente)
    fuentes_en_manifest = set(manifest)

    nuevos = [f for f in fuentes_actuales if f not in manifest]
    cambiados = [
        f
        for f in fuentes_actuales
        if f in manifest and compute_file_hash(Path(f)) != manifest[f]["hash"]
    ]
    eliminados = list(fuentes_en_manifest - fuentes_actuales)
    sin_cambios = fuentes_actuales - set(nuevos) - set(cambiados)

    logger.info(
        f"Diagnóstico de cambios: {len(nuevos)} nuevo/s | {len(cambiados)} modificado/s | "
        f"{len(eliminados)} eliminado/s | {len(sin_cambios)} sin cambios (se omiten)"
    )

    # Borrar del índice: chunks viejos de archivos modificados + archivos eliminados
    if vectorstore is not None:
        for fuente in cambiados + eliminados:
            ids_viejos = _get_ids_for_source(vectorstore, fuente)
            if ids_viejos:
                vectorstore.delete(ids=ids_viejos)
                logger.info(f"Eliminados {len(ids_viejos)} chunk/s viejos de: {fuente}")

    # Reunir chunks a embeber: solo nuevos + modificados
    pendientes: List[Document] = []
    for fuente in nuevos + cambiados:
        pendientes.extend(chunks_por_fuente[fuente])

    if pendientes:
        vectorstore = _procesar_lotes(
            vectorstore, pendientes, embeddings, target_path, checkpoint_path,
            batch_size, delay_seconds,
        )
    elif vectorstore is not None and eliminados:
        # Solo hubo eliminaciones, sin nada nuevo que embeber: igual persistimos
        save_vectorstore(vectorstore, target_path)

    if vectorstore is None:
        raise ValueError(
            "No hay documentos para indexar y no existe un índice previo. "
            "Verifica que Documentacion/ tenga archivos."
        )

    # Actualizar manifest: agregar nuevos/modificados, quitar eliminados
    for fuente in nuevos + cambiados:
        manifest[fuente] = {"hash": compute_file_hash(Path(fuente))}
    for fuente in eliminados:
        manifest.pop(fuente, None)
    save_manifest(manifest_path, manifest)

    if not pendientes and not eliminados:
        logger.info("Índice ya estaba al día — ningún archivo nuevo o modificado.")

    checkpoint_path.unlink(missing_ok=True)
    return vectorstore


def build_vectorstore(
    documents: List[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
    delay_seconds: float = DEFAULT_DELAY_BETWEEN_BATCHES_SECONDS,
    vectorstore_path: Optional[Path] = None,
) -> FAISS:
    """
    Reconstruye el índice FAISS DESDE CERO con todos los documentos dados,
    ignorando cualquier manifest o índice previo. Uso: primer build, o
    forzar un re-index total (ej. tras cambiar el modelo de embeddings).
    Para el uso cotidiano (agregar/modificar documentos), usa sync_vectorstore.
    """
    if not documents:
        raise ValueError("No se puede construir un vectorstore sin documentos")

    target_path = vectorstore_path or settings.VECTORSTORE_DIR
    target_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target_path.parent / MANIFEST_FILENAME
    checkpoint_path = target_path.parent / CHECKPOINT_FILENAME

    embeddings = get_embeddings()
    vectorstore = _procesar_lotes(
        None, documents, embeddings, target_path, checkpoint_path,
        batch_size, delay_seconds,
    )

    # Reconstruye el manifest desde cero para que quede consistente con sync futuro
    manifest = {}
    for doc in documents:
        fuente = doc.metadata.get("source", "desconocida")
        if fuente not in manifest and Path(fuente).exists():
            manifest[fuente] = {"hash": compute_file_hash(Path(fuente))}
    save_manifest(manifest_path, manifest)

    checkpoint_path.unlink(missing_ok=True)
    logger.info("Índice FAISS reconstruido completamente desde cero")
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: Optional[Path] = None) -> None:
    """Persiste el índice FAISS en disco (carpeta con index.faiss + index.pkl)."""
    target = path or settings.VECTORSTORE_DIR
    target.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target))


def load_vectorstore(path: Optional[Path] = None) -> FAISS:
    """Carga un índice FAISS previamente construido."""
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
