from pathlib import Path
import os
import uuid

import streamlit as st
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Configuración general
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="PokéAtlas",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Local: .env
load_dotenv(ROOT / ".env")

# Streamlit Community Cloud: Secrets
if not os.getenv("GEMINI_API_KEY"):
    try:
        secret_key = st.secrets["GEMINI_API_KEY"]
        if secret_key:
            os.environ["GEMINI_API_KEY"] = secret_key
    except Exception:
        pass

from src.pokeatlas_agent import (  # noqa: E402
    PokeAtlasEngine,
    LLM_MODEL,
    EMBEDDING_MODEL,
)


# -------------------------------------------------------------------
# Estilo
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(227, 53, 13, 0.08), transparent 24rem),
                radial-gradient(circle at 85% 10%, rgba(37, 99, 235, 0.06), transparent 22rem);
        }

        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .poke-hero {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.1rem;
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(8px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.05);
        }

        @media (prefers-color-scheme: dark) {
            .poke-hero {
                background: rgba(30, 30, 30, 0.65);
            }
        }

        .poke-title {
            font-size: 2.25rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }

        .poke-subtitle {
            opacity: 0.72;
            margin-top: 0.45rem;
            margin-bottom: 0;
        }

        .poke-badge {
            display: inline-block;
            margin-right: .4rem;
            margin-top: .65rem;
            padding: .25rem .65rem;
            border-radius: 999px;
            font-size: .82rem;
            border: 1px solid rgba(128,128,128,.25);
        }

        [data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: .2rem .35rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128,128,128,.16);
        }

        .small-muted {
            opacity: .68;
            font-size: .86rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Cargar el cerebro una sola vez por proceso
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    return PokeAtlasEngine(ROOT)


try:
    engine = load_engine()
except Exception as exc:
    st.error("No he podido iniciar PokéAtlas.")
    st.exception(exc)
    st.info(
        "Comprueba que existe `chroma_db/` en la raíz del proyecto y que "
        "`GEMINI_API_KEY` está configurada en `.env`."
    )
    st.stop()


# -------------------------------------------------------------------
# Estado de sesión
# -------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy **PokéAtlas** 🎮. Puedes preguntarme por Pokémon, "
                "tipos, estadísticas base, movimientos y habilidades."
            ),
            "sources": [],
            "standalone_question": None,
            "quota_error": False,
        }
    ]


def new_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Conversación reiniciada. ¿Sobre qué Pokémon quieres hablar? 🎮"
            ),
            "sources": [],
            "standalone_question": None,
            "quota_error": False,
        }
    ]


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🎮 PokéAtlas")
    st.caption("Asistente RAG experto en datos Pokémon")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vectores", engine.vector_count)
    with col2:
        st.metric("Fuentes", "4")

    st.caption(f"LLM: `{LLM_MODEL}`")
    st.caption(f"Embeddings: `{EMBEDDING_MODEL}`")

    st.divider()

    st.subheader("Prueba rápida")

    example_prompts = {
        "🐉 Tipos de Garchomp": "¿Qué tipos tiene Garchomp?",
        "⚡ Hiperrayo": "¿Qué potencia y precisión tiene Hiperrayo?",
        "🛡️ Piel Tosca": "¿Qué efecto tiene la habilidad Piel Tosca?",
        "❄️ Tipo Dragón": "¿De qué tipos recibe daño doble el tipo Dragón?",
    }

    selected_prompt = None
    for label, value in example_prompts.items():
        if st.button(label, use_container_width=True):
            selected_prompt = value

    st.divider()

    if st.button(
        "🧹 Nueva conversación",
        use_container_width=True,
        type="secondary",
    ):
        new_conversation()
        st.rerun()

    st.caption(
        "PokéAtlas responde solo con información recuperada de su base "
        "documental. Si no dispone de un dato, debe reconocerlo."
    )


# -------------------------------------------------------------------
# Cabecera
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="poke-hero">
        <div class="poke-title">🎮 PokéAtlas</div>
        <p class="poke-subtitle">
            Pregunta en lenguaje natural. Resuelvo entidades exactas cuando puedo,
            uso búsqueda semántica como fallback y Gemini genera la respuesta.
        </p>
        <span class="poke-badge">Gemini</span>
        <span class="poke-badge">ChromaDB</span>
        <span class="poke-badge">LangGraph</span>
        <span class="poke-badge">Memoria</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Pintar historial
# -------------------------------------------------------------------
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🎮"

    with st.chat_message(message["role"], avatar=avatar):
        if message.get("quota_error"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])

        sources = message.get("sources", [])
        standalone = message.get("standalone_question")

        if message["role"] == "assistant" and (sources or standalone):
            with st.expander("🔎 Cómo se construyó esta respuesta"):
                if standalone:
                    st.caption("Consulta usada para retrieval")
                    st.code(standalone, language=None)

                if sources:
                    st.caption("Fragmentos recuperados")
                    for i, source in enumerate(sources, start=1):
                        mode = source.get("retrieval_mode", "semántico")
                        st.markdown(
                            f"**{i}.** `{source['category']}` · "
                            f"**{source['name']}**  \n"
                            f"<span class='small-muted'>"
                            f"{source['source']} · retrieval: {mode}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )


# -------------------------------------------------------------------
# Entrada de chat
# -------------------------------------------------------------------
typed_prompt = st.chat_input(
    "Pregunta por un Pokémon, movimiento, habilidad o tipo..."
)

prompt = typed_prompt or selected_prompt

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎮"):
        result = {}
        activity = st.empty()
        activity.caption("🔎 Recuperando contexto y preparando la respuesta...")

        try:
            streamed_answer = st.write_stream(
                engine.ask_stream(
                    question=prompt,
                    thread_id=st.session_state.thread_id,
                    result_holder=result,
                )
            )
        except Exception as exc:
            activity.empty()
            st.error(
                "Ha ocurrido un error inesperado al consultar PokéAtlas."
            )
            st.exception(exc)
            st.stop()

        activity.empty()

        # st.write_stream devuelve el texto concatenado cuando todos los
        # fragmentos emitidos son strings. Lo usamos como respaldo.
        if not result.get("answer") and isinstance(streamed_answer, str):
            result["answer"] = streamed_answer

        result.setdefault("sources", [])
        result.setdefault("standalone_question", prompt)
        result.setdefault("quota_error", False)

        if result["quota_error"]:
            st.warning(result["answer"])

        if result["sources"] or result["standalone_question"]:
            with st.expander("🔎 Cómo se construyó esta respuesta"):
                st.caption("Consulta usada para retrieval")
                st.code(
                    result["standalone_question"],
                    language=None,
                )

                if result["sources"]:
                    st.caption("Fragmentos recuperados")
                    for i, source in enumerate(result["sources"], start=1):
                        mode = source.get("retrieval_mode", "semántico")
                        st.markdown(
                            f"**{i}.** `{source['category']}` · "
                            f"**{source['name']}**  \n"
                            f"<span class='small-muted'>"
                            f"{source['source']} · retrieval: {mode}"
                            f"</span>",
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "standalone_question": result["standalone_question"],
            "quota_error": result["quota_error"],
        }
    )
