# PokéAtlas — Interfaz Streamlit

Esta carpeta contiene la interfaz web opcional de PokéAtlas.

## Dónde colocar los archivos

Copia los archivos en la raíz de tu proyecto con esta estructura:

```text
PokeAtlas/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
├── .streamlit/
│   └── config.toml
├── src/
│   ├── __init__.py
│   └── pokeatlas_agent.py
├── chroma_db/
├── data/
├── notebooks/
└── docs/
```

## Ejecución local

Desde una terminal situada en la raíz de `PokeAtlas/`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit abrirá la aplicación en el navegador.

## API key

En local se reutiliza el `.env` existente:

```text
GEMINI_API_KEY=tu_clave_real
```

No subas `.env` a GitHub.

La aplicación también admite `st.secrets`, lo que facilitará el despliegue
posterior en Streamlit Community Cloud.

## Qué reutiliza

La web no reconstruye la base vectorial. Abre la colección existente de
`chroma_db/` y reutiliza:

- Gemini Embeddings
- ChromaDB
- LangGraph
- memoria por `thread_id`
- el System Prompt validado
- Gemini 3.5 Flash-Lite como LLM

## Nota sobre despliegue

Para una ejecución local basta con conservar `chroma_db/` en tu ordenador.
Para desplegar en Streamlit Community Cloud tendremos que decidir cómo incluir
la base Chroma en el repositorio y configurar la API key mediante Secrets.
Haremos ese paso después de validar la web local.
