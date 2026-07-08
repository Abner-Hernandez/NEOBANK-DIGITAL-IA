"""
logger.py
---------
Configuración centralizada de logging. En un servidor OCI sin acceso
interactivo, los logs de archivo son la única ventana para diagnosticar
problemas — por eso esto se diseña desde la Fase 1, no se improvisa después.
"""

import logging
import sys
from pathlib import Path

from src.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger configurado con dos handlers:
    - Consola (stdout): útil en desarrollo local y en `docker logs` / journalctl en OCI.
    - Archivo rotativo en LOG_DIR: persistencia para auditoría e incident response.
    """
    logger = logging.getLogger(name)

    # Evita duplicar handlers si get_logger se llama varias veces con el mismo name
    if logger.handlers:
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Handler de consola ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Handler de archivo ---
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file: Path = settings.LOG_DIR / "alura_agente.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
