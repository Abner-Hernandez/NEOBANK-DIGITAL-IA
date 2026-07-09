"""
manifest.py
-----------
Rastrea que archivos fuente ya fueron indexados, mediante un hash MD5 de
su contenido. Esto permite detectar archivos nuevos, modificados o
eliminados entre corridas de build_index, y es la base de la indexacion
incremental (src/rag/vectorstore.py:sync_vectorstore).

El manifest se guarda como JSON junto al indice FAISS:
    vectorstore/faiss_index/../_manifest.json
"""

import hashlib
import json
from pathlib import Path
from typing import Dict

ManifestDict = Dict[str, Dict[str, str]]  # {source_path: {"hash": "..."}}

MANIFEST_FILENAME = "_manifest.json"


def compute_file_hash(path: Path) -> str:
    """Hash MD5 del contenido del archivo. Suficiente para detección de cambios
    (no es un uso criptográfico, no nos importa resistencia a colisiones)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_manifest(manifest_path: Path) -> ManifestDict:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_manifest(manifest_path: Path, manifest: ManifestDict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
