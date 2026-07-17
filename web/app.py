"""
app.py
------
Interfaz de chat de Alura Agente, construida con Streamlit. Consume la API
FastAPI (Fase 2.4) vía HTTP -- este servicio NO importa LangChain ni FAISS
directamente; es un cliente delgado, lo cual mantiene la imagen liviana y
desacopla completamente el frontend del motor RAG.

Diseño de UX en dos pestañas:
- "Preguntar": experiencia simple para el empleado promedio. Sin filtros,
  sin configuración -- solo escribe y pregunta. Busca en TODO el corpus.
- "Avanzado": filtros opcionales (categoría, cantidad de fragmentos) para
  quien quiera afinar la búsqueda. Los ajustes aquí aplican a la siguiente
  pregunta que se haga en la pestaña "Preguntar" -- ambas comparten la
  misma conversación (st.session_state.messages).
"""

import os

import requests
import streamlit as st

# En Docker Compose, "api" resuelve al contenedor de la API vía la red
# interna del proyecto. Para correr Streamlit fuera de Docker (desarrollo
# local sin contenedores), define API_URL=http://localhost:8000 en tu entorno.
API_URL = os.getenv("API_URL", "http://api:8000")
REQUEST_TIMEOUT_SECONDS = 60

st.set_page_config(page_title="Alura Agente", page_icon="💬", layout="centered")


# ---------------------------------------------------------------------------
# Utilidades de comunicación con la API
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def obtener_categorias() -> list[str]:
    """Consulta las categorías disponibles en el corpus. Cacheado 5 min."""
    try:
        respuesta = requests.get(f"{API_URL}/categories", timeout=10)
        respuesta.raise_for_status()
        return respuesta.json().get("categories", [])
    except requests.exceptions.RequestException:
        return []


def preguntar_al_agente(pregunta: str, categoria: str | None, k: int) -> dict:
    """
    Envía la pregunta a POST /query. Retorna un dict con "answer" y "sources",
    o un mensaje de error legible si la API no responde o rechaza la consulta.
    """
    payload: dict = {"question": pregunta, "k": k}
    if categoria and categoria != "Todas":
        payload["categoria"] = categoria

    try:
        respuesta = requests.post(
            f"{API_URL}/query", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
        )
        respuesta.raise_for_status()
        return respuesta.json()

    except requests.exceptions.Timeout:
        return {"answer": "⚠️ La API tardó demasiado en responder. Intenta de nuevo.", "sources": []}
    except requests.exceptions.ConnectionError:
        return {
            "answer": f"⚠️ No pude conectar con la API en `{API_URL}`. "
            "Verifica que el servicio esté corriendo.",
            "sources": [],
        }
    except requests.exceptions.HTTPError:
        detalle = "Error desconocido"
        try:
            detalle = respuesta.json().get("detail", detalle)
        except ValueError:
            pass
        return {"answer": f"⚠️ {detalle}", "sources": []}


def renderizar_historial() -> None:
    """Dibuja los mensajes ya existentes en la conversación (compartida entre pestañas)."""
    for mensaje in st.session_state.messages:
        with st.chat_message(mensaje["role"]):
            st.markdown(mensaje["content"])
            if mensaje.get("sources"):
                with st.expander("📄 Fuentes consultadas"):
                    for fuente in mensaje["sources"]:
                        st.markdown(
                            f"- **{fuente['source']}** _(categoría: {fuente['categoria']})_"
                        )


# ---------------------------------------------------------------------------
# Estado compartido (persiste entre ambas pestañas)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "categoria_filtro" not in st.session_state:
    st.session_state.categoria_filtro = "Todas"
if "k_fragmentos" not in st.session_state:
    st.session_state.k_fragmentos = 4

st.title("💬 Alura Agente")
st.caption("Asistente virtual de documentos internos — NeoBank Digital")

tab_preguntar, tab_avanzado = st.tabs(["💬 Preguntar", "⚙️ Avanzado"])


# ---------------------------------------------------------------------------
# Pestaña principal: experiencia simple para el empleado
# ---------------------------------------------------------------------------
with tab_preguntar:
    renderizar_historial()

    pregunta_usuario = st.chat_input("Escribe tu pregunta sobre los documentos internos...")

    if pregunta_usuario:
        st.session_state.messages.append({"role": "user", "content": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Consultando documentos internos..."):
                resultado = preguntar_al_agente(
                    pregunta_usuario,
                    st.session_state.categoria_filtro,
                    st.session_state.k_fragmentos,
                )

            st.markdown(resultado["answer"])

            fuentes = resultado.get("sources", [])
            if fuentes:
                with st.expander("📄 Fuentes consultadas"):
                    for fuente in fuentes:
                        st.markdown(
                            f"- **{fuente['source']}** _(categoría: {fuente['categoria']})_"
                        )

        st.session_state.messages.append(
            {"role": "assistant", "content": resultado["answer"], "sources": fuentes}
        )


# ---------------------------------------------------------------------------
# Pestaña avanzada: filtros opcionales, no requeridos para usar el chat
# ---------------------------------------------------------------------------
with tab_avanzado:
    st.markdown(
        "Estos ajustes aplican a tu **próxima** pregunta en la pestaña "
        "**💬 Preguntar**. No son necesarios para usar el asistente."
    )

    categorias_disponibles = obtener_categorias()

    st.session_state.categoria_filtro = st.selectbox(
        "Categoría",
        options=["Todas"] + categorias_disponibles,
        index=(["Todas"] + categorias_disponibles).index(st.session_state.categoria_filtro)
        if st.session_state.categoria_filtro in (["Todas"] + categorias_disponibles)
        else 0,
        help="Restringe la búsqueda a una categoría específica del corpus.",
    )

    st.session_state.k_fragmentos = st.slider(
        "Fragmentos a recuperar",
        min_value=1,
        max_value=10,
        value=st.session_state.k_fragmentos,
        help="Cuántos fragmentos de contexto se le pasan al modelo por consulta.",
    )

    st.divider()

    if st.button("🗑️ Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

    if not categorias_disponibles:
        st.warning(
            "No se pudieron cargar las categorías. "
            "¿Está la API corriendo y el índice construido?"
        )
