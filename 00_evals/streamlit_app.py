"""
Public-safe RAGAS eval demo, standalone from the 38_rag_selfhosted_stack
project — deployable to Streamlit Community Cloud so anyone can try
RAGAS scoring with their own free Groq key, no dependency on any of
this repo's other projects or credentials at all.

Based on 38_rag_selfhosted_stack/evals/eval_viewer.py, but:

1. RAG Metrics only — no Guardrails tab. That tab calls
   app.guardrails.bedrock.check_input() directly via boto3, using
   38_rag_selfhosted_stack's own AWS credentials. On a public site,
   every anonymous visitor's click would bill Bedrock calls to that
   AWS account — dropped entirely rather than risk that.

2. No live /query call — RAG Metrics' "Run RAGAS eval" scores directly
   against a pre-filled dataset (bundled golden_dataset_prefilled.json,
   or a visitor's own uploaded JSON of the same shape), which already
   has actual_response/actual_contexts filled in (as if a real backend
   had already answered each question). This avoids needing a publicly
   reachable backend URL, which would otherwise let anonymous visitors
   trigger real queries against a live backend.

This file has NO import of app.config, app.guardrails, or boto3 at
all — the only credential it ever touches is whatever Groq key a
visitor pastes into the sidebar themselves.

Scores faithfulness/context_precision/context_recall only, not
answer_relevancy — that metric is the only one of the four that needs
a separate embeddings model (Groq has no embeddings API), which would
mean downloading and running a local sentence-transformers model.
Dropping it keeps this app to a single Groq LLM call path, no local
model, lighter dependencies.

Eval results live in st.session_state, not a file on disk — this app
serves multiple concurrent, unrelated visitors on one shared process
(Streamlit Community Cloud), so writing results to a shared path would
let one visitor's run clobber another's, or a fresh visitor land on a
stranger's leftover results.

Run:
    streamlit run 37_StreamlitAppsPublic/00_evals/streamlit_app.py

Deploy: push this repo to GitHub, then on share.streamlit.io point a
new app at this file (37_StreamlitAppsPublic/00_evals/streamlit_app.py)
as the entrypoint, with this same folder's requirements.txt.
"""
import json
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

# ragas's own evaluate() path (LangchainLLMWrapper, and importing metrics
# from ragas.metrics rather than ragas.metrics.collections) is flagged
# deprecated, but the suggested replacements in ragas.metrics.collections
# don't implement single_turn_ascore/single_turn_score — the methods
# evaluate() actually calls — so they're not a drop-in swap yet.
# Matching by module name is unreliable since ragas attributes these warnings
# to the caller's stack frame (this file), not its own — so filter by the
# warning's own text instead.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*(LangchainLLMWrapper|ragas\.metrics\.collections).*")

BUNDLED_DATASET_PATH = Path(__file__).parent / "golden_dataset_prefilled.json"

REQUIRED_SAMPLE_KEYS = ["question", "reference", "actual_response", "actual_contexts"]

# Groq's OpenAI-compatible endpoint — any model in Groq's catalog works
# here (https://console.groq.com/docs/models). This list is a curated
# snapshot, not fetched live from Groq (no API key is available at
# import time to call their /models endpoint) — it may drift out of
# date as Groq adds/retires models. "Custom" covers that gap.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_OPTIONS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
    "Custom (type below)",
]
DEFAULT_GROQ_MODEL = GROQ_MODEL_OPTIONS[0]

st.set_page_config(page_title="RAGAS Eval Demo", layout="wide")
st.title("RAGAS Eval Demo")
st.caption(
    "Try RAGAS scoring with your own free Groq API key — scores a "
    "pre-filled dataset (question/answer/context already populated, "
    "as if a real RAG backend had already answered each one), no "
    "backend of ours required. Use the bundled sample dataset or "
    "upload your own."
)

_ARCHITECTURE_DIAGRAM = """\
flowchart TD
    A[Visitor opens app] --> B{Pick dataset}
    B -->|Bundled sample| C[golden_dataset_prefilled.json]
    B -->|Upload own JSON| D[Uploaded file]
    D --> E{Validate shape}
    E -->|Invalid| F[Show error, stop]
    E -->|Valid| G[samples list]
    C --> G
    H[Paste Groq API key in sidebar] --> I[Judge LLM via Groq's\\nOpenAI-compatible endpoint]
    I --> K[ragas.evaluate]
    G --> K
    K --> L[faithfulness /\\ncontext_precision / context_recall]
    L --> M[(st.session_state,\\nper visitor session)]
    M --> N[Aggregate scores + per-sample table]
"""

with st.expander("How this app works (diagram)"):
    if hasattr(st, "mermaid_chart"):
        st.mermaid_chart(_ARCHITECTURE_DIAGRAM)
    else:
        # st.mermaid_chart needs a recent Streamlit (requirements.txt only
        # pins a floor version) — fall back to a plain code block so older
        # installs still show something useful instead of crashing.
        st.code(_ARCHITECTURE_DIAGRAM, language="mermaid")
    st.caption(
        "No step here touches this repo's other projects, AWS, or a live "
        "backend — the only external call is to Groq's API, using "
        "whichever key you paste in yourself."
    )

with st.sidebar:
    st.header("Judge model (RAGAS)")
    st.caption(
        "RAGAS's faithfulness/context_precision/context_recall metrics "
        "need an LLM to act as judge. Paste a free Groq API key "
        "(console.groq.com/keys)."
    )
    groq_api_key = st.text_input("Groq API key", type="password", key="groq_api_key")
    groq_model_choice = st.selectbox("Groq model", GROQ_MODEL_OPTIONS, index=0, key="groq_model_choice")
    if groq_model_choice == "Custom (type below)":
        groq_model = st.text_input("Custom model ID", value=DEFAULT_GROQ_MODEL, key="groq_model_custom")
    else:
        groq_model = groq_model_choice
    st.caption(
        "Hitting a 429/rate-limit or daily-token-limit error? Smaller "
        "models (e.g. `llama-3.1-8b-instant`) generally get more "
        "generous free-tier limits than `llama-3.3-70b-versatile` — "
        "try switching here rather than waiting out the reset."
    )
    if not groq_api_key:
        st.info("Enter a Groq API key above to enable \"Run RAGAS eval\".")

    st.caption(
        "RAGAS fires this many concurrent judge-LLM calls by default "
        "(16) — enough to trip Groq's free-tier rate limit. Lower this "
        "if you see 429/rate-limit errors; raise it on a paid Groq plan."
    )
    max_workers = st.slider("Max concurrent judge calls", min_value=1, max_value=16, value=4, key="max_workers")


def _validate_samples(data: dict) -> list[dict]:
    """Raises ValueError with a message meant to be shown directly to
    the uploading visitor, since this is the only validation their
    file gets before being fed to RAGAS."""
    if "rag_samples" not in data:
        raise ValueError("JSON must have a top-level \"rag_samples\" key.")
    samples = data["rag_samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("\"rag_samples\" must be a non-empty list.")
    for i, sample in enumerate(samples):
        missing = [k for k in REQUIRED_SAMPLE_KEYS if k not in sample]
        if missing:
            raise ValueError(f"Sample {i} is missing required field(s): {', '.join(missing)}.")
        if not isinstance(sample["actual_contexts"], list):
            raise ValueError(f"Sample {i}'s \"actual_contexts\" must be a list of strings.")
    return samples


def _build_judge_llm():
    """Wraps a Groq-backed ChatOpenAI (OpenAI-compatible endpoint) in
    RAGAS's LangchainLLMWrapper, so evaluate()'s llm= override routes
    every LLM-based metric's judging through Groq instead of defaulting
    to OpenAI (which would need an OPENAI_API_KEY nobody visiting this
    page has any reason to have)."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key, model=groq_model)
    return LangchainLLMWrapper(chat)


st.subheader("Dataset")
dataset_choice = st.radio(
    "Score against:",
    ["Bundled sample dataset", "Upload my own JSON"],
    key="dataset_choice",
    horizontal=True,
)

samples = None
if dataset_choice == "Bundled sample dataset":
    golden = json.loads(BUNDLED_DATASET_PATH.read_text())
    samples = golden["rag_samples"]
else:
    st.caption(
        "Upload a JSON file with a top-level `rag_samples` list. Each "
        "sample needs `question`, `reference` (ground truth answer), "
        "`actual_response` (what your RAG system answered), and "
        "`actual_contexts` (list of retrieved context strings) — see "
        "the bundled `golden_dataset_prefilled.json` in this folder for "
        "an example shape."
    )
    uploaded = st.file_uploader("Golden dataset JSON", type="json", key="uploaded_dataset")
    if uploaded is not None:
        try:
            samples = _validate_samples(json.loads(uploaded.getvalue()))
            st.success(f"Loaded {len(samples)} sample(s) from {uploaded.name}.")
        except (ValueError, json.JSONDecodeError) as e:
            st.error(f"Couldn't use this file: {e}")

if samples is not None:
    preview_cols = [c for c in ["id", "domain", "question", "reference", "actual_response"] if c in samples[0]]
    st.dataframe(pd.DataFrame(samples)[preview_cols or list(samples[0].keys())], width="stretch")

if st.button("Run RAGAS eval", key="run_ragas", disabled=not groq_api_key or samples is None):
    with st.spinner("Scoring with RAGAS (Groq judge) — may take a minute..."):
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall, faithfulness
            from ragas.run_config import RunConfig

            judge_llm = _build_judge_llm()

            dataset = Dataset.from_dict(
                {
                    "question": [s["question"] for s in samples],
                    "answer": [s["actual_response"] for s in samples],
                    "contexts": [s["actual_contexts"] for s in samples],
                    "ground_truth": [s["reference"] for s in samples],
                }
            )
            # No answer_relevancy here — it's the only one of these four
            # metrics that needs an embeddings model (to compare a judge-
            # generated hypothetical question against the real one via
            # cosine similarity), which meant downloading and running a
            # local sentence-transformers model. Dropping it keeps this
            # app to just an LLM judge call, no local model at all.
            result = evaluate(
                dataset,
                metrics=[faithfulness, context_precision, context_recall],
                llm=judge_llm,
                run_config=RunConfig(max_workers=max_workers),
            )
            st.session_state["ragas_results"] = result.to_pandas().to_dict(orient="records")
            st.success("RAGAS eval complete.")
        except Exception as e:
            st.error(f"RAGAS eval failed: {e}")

if "ragas_results" in st.session_state:
    df = pd.DataFrame(st.session_state["ragas_results"])
    metric_cols = [c for c in ["faithfulness", "context_precision", "context_recall"] if c in df.columns]

    st.subheader("Aggregate scores")
    cols = st.columns(len(metric_cols)) if metric_cols else []
    for col, metric in zip(cols, metric_cols):
        col.metric(metric.replace("_", " ").title(), f"{df[metric].mean():.3f}")

    if metric_cols:
        st.subheader("Score distribution by metric")
        st.bar_chart(df[metric_cols].mean())

    st.subheader("Per-sample results")
    display_cols = [c for c in ["user_input", "question", "answer", *metric_cols] if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, width="stretch")
else:
    st.info("No results yet — pick a dataset and click \"Run RAGAS eval\" above.")
