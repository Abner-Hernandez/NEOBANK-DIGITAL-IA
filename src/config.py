"""
config.py
---------
Punto único de configuración del proyecto. Carga variables de entorno
y expone rutas absolutas basadas en la raíz del proyecto (no hardcodeadas),
para que el mismo código funcione igual en tu máquina local y en OCI Compute.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Raíz del proyecto = carpeta que contiene /src, /data, /logs, etc.
# Calculada dinámicamente para no depender de rutas absolutas del sistema.
BASE_DIR = Path(__file__).resolve().parent.parent

# Carga el archivo .env ubicado en la raíz del proyecto.
load_dotenv(BASE_DIR / ".env")


class Settings:
    """Agrupa toda la configuración de la aplicación en un solo objeto."""

    # --- Entorno ---
    APP_ENV: str = os.getenv("APP_ENV", "local")

    # --- Google Gemini ---
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
    GEMINI_EMBEDDING_MODEL: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"
    )

    # --- Rutas de datos (siempre relativas a BASE_DIR) ---
    # "Documentacion" es la carpeta fuente de archivos txt/pdf/csv que el
    # agente indexará (convención definida en el repo NEOBANK-DIGITAL-IA).
    RAW_DATA_DIR: Path = BASE_DIR / os.getenv("RAW_DATA_DIR", "Documentacion")
    VECTORSTORE_DIR: Path = BASE_DIR / os.getenv(
        "VECTORSTORE_DIR", "vectorstore/faiss_index"
    )

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: Path = BASE_DIR / os.getenv("LOG_DIR", "logs")

    @classmethod
    def validate(cls) -> None:
        """
        Valida que la configuración crítica esté presente antes de arrancar
        cualquier proceso. Falla rápido y con un mensaje claro, en vez de
        dejar que el error explote más adelante dentro de una llamada a la API.
        """
        if not cls.GOOGLE_API_KEY:
            raise EnvironmentError(
                "GOOGLE_API_KEY no está definida. "
                "Copia .env.example como .env y completa tu API key."
            )
        if not cls.RAW_DATA_DIR.exists():
            raise FileNotFoundError(
                f"El directorio de datos crudos no existe: {cls.RAW_DATA_DIR}"
            )


settings = Settings()
