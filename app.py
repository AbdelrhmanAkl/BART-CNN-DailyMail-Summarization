import html
import time
from pathlib import Path

import streamlit as st
import torch
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.models.bart.tokenization_bart import BartTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent

LOCAL_MODEL_PATH = PROJECT_ROOT / "model"
HF_MODEL_ID = "AbdelrahmanAkl/bart-cnn-dailymail-summarization"

MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128

NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3

SAMPLE_ARTICLE = (
    "The city council on Tuesday approved a sweeping plan to expand the "
    "metropolitan bus network, adding twelve new routes and extending service "
    "hours until midnight on weekdays. The proposal, which has been debated for "
    "more than two years, passed by a vote of seven to two after a lengthy public "
    "hearing. Supporters argued that better public transport would reduce traffic "
    "congestion, cut carbon emissions and give lower-income residents easier "
    "access to jobs and healthcare. Opponents raised concerns about the cost, "
    "estimated at 85 million dollars over five years, and questioned whether "
    "ridership would grow enough to justify the spending. The mayor said the "
    "project would be funded through a combination of federal grants, a modest "
    "increase in parking fees and savings from the retirement of older, less "
    "efficient vehicles. Construction of new bus lanes is expected to begin next "
    "spring, with the first new routes entering service by the end of next year. "
    "City officials also promised to publish quarterly reports on ridership, "
    "costs and on-time performance so residents can track whether the investment "
    "is delivering results. Transit advocates welcomed the decision, calling it "
    "the most significant improvement to local public transport in a generation."
)


st.set_page_config(
    page_title="BART Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fa;
    }

    .main .block-container {
        max-width: 1150px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    .hero {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
    }

    .badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 6px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .hero-text {
        color: #6b7280;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 8px;
        max-width: 800px;
    }

    .tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #475569;
        padding: 5px 10px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 10px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-top: 16px;
        margin-bottom: 4px;
    }

    .section-text {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 10px;
    }

    div[data-testid="stTextArea"] textarea {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        color: #111827;
        font-size: 15px;
        line-height: 1.6;
        padding: 14px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #64748b;
        box-shadow: 0 0 0 1px #64748b;
    }

    div.stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 10px;
        font-size: 15px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"] {
        background-color: #111827;
        border: 1px solid #111827;
        color: #ffffff;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1f2937;
        border-color: #1f2937;
        color: #ffffff;
    }

    div.stButton > button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #d1d5db;
        color: #111827;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #111827;
        color: #111827;
    }

    .status {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
        color: #4b5563;
        font-size: 13px;
    }

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px;
        margin-top: 10px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.03);
    }

    .result-label {
        color: #6b7280;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .result-text {
        color: #1f2937;
        font-size: 16px;
        line-height: 1.8;
    }

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if (
        LOCAL_MODEL_PATH.exists()
        and (LOCAL_MODEL_PATH / "model.safetensors").exists()
    ):
        model_source = LOCAL_MODEL_PATH
        source_label = "Local model"
    else:
        model_source = HF_MODEL_ID
        source_label = "Hugging Face"

    tokenizer = BartTokenizer.from_pretrained(model_source)
    model = BartForConditionalGeneration.from_pretrained(model_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device, source_label


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown(
        """
        <h3 style="margin-bottom:5px;">BART Summarizer</h3>
        <p style="color:#6b7280;font-size:13px;">
        Abstractive text summarization using a fine-tuned BART Transformer.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Model")

    st.write("**Architecture:** BART")
    st.write("**Base Model:** facebook/bart-base")
    st.write("**Dataset:** CNN/DailyMail")
    st.write("**Task:** Abstractive Summarization")

    st.divider()

    st.markdown("### Generation Settings")

    num_beams = st.slider(
        "Beam Search",
        min_value=1,
        max_value=8,
        value=NUM_BEAMS,
        help="Higher values explore more candidates: better quality, slower.",
    )

    max_output = st.slider(
        "Max Output Tokens",
        min_value=32,
        max_value=256,
        value=MAX_OUTPUT_LENGTH,
        step=16,
        help="Upper limit for the summary length.",
    )

    length_penalty = st.slider(
        "Length Penalty",
        min_value=0.5,
        max_value=3.0,
        value=LENGTH_PENALTY,
        step=0.1,
        help="Above 1.0 favors longer summaries, below 1.0 favors shorter ones.",
    )

    no_repeat = st.slider(
        "No Repeat N-gram",
        min_value=0,
        max_value=5,
        value=NO_REPEAT_NGRAM_SIZE,
        help="Blocks repeating the same n-gram. 0 disables it.",
    )

    st.caption(f"Max input: {MAX_INPUT_LENGTH} tokens (longer text is truncated)")


# ------------------------------------------------------------------- Hero
st.markdown(
    """
    <div class="hero">
        <div class="badge">NLP · TRANSFORMER · SUMMARIZATION</div>
        <div class="hero-title">BART Text Summarization</div>
        <div class="hero-text">
            Transform long-form articles into concise summaries
            using a fine-tuned BART Transformer model trained on
            the CNN/DailyMail dataset.
        </div>
        <div>
            <span class="tag">BART</span>
            <span class="tag">CNN/DailyMail</span>
            <span class="tag">Transformer</span>
            <span class="tag">Abstractive Summarization</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    with st.spinner("Loading model..."):
        tokenizer, model, device, source_label = load_model()
except Exception as error:
    st.error(f"Failed to load the model: {error}")
    st.stop()


device_label = "GPU" if device.type == "cuda" else "CPU"

st.markdown(
    f"""
    <div class="status">
        Model ready · Running on {device_label} · Source: {source_label}
    </div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ Input
st.markdown(
    '<div class="section-title">Article Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-text">
        Paste an English article below, or try a sample.
    </div>
    """,
    unsafe_allow_html=True,
)

if "article" not in st.session_state:
    st.session_state["article"] = ""


def load_sample():
    st.session_state["article"] = SAMPLE_ARTICLE


def clear_text():
    st.session_state["article"] = ""
    st.session_state.pop("result", None)


btn_a, btn_b, _ = st.columns([1, 1, 3])
with btn_a:
    st.button("Try a sample", on_click=load_sample, key="sample_btn")
with btn_b:
    st.button("Clear", on_click=clear_text, key="clear_btn")


article = st.text_area(
    "Article",
    key="article",
    height=240,
    placeholder="Paste your article here...",
    label_visibility="collapsed",
)

if article.strip():
    st.caption(
        f"{len(article.split()):,} words entered · "
        f"Maximum input length: {MAX_INPUT_LENGTH} tokens"
    )


# --------------------------------------------------------------- Generate
if st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True,
):
    if not article.strip():
        st.warning("Please enter some text before generating a summary.")
    else:
        spinner_msg = "Generating summary..."
        if device.type == "cpu":
            spinner_msg += " (this may take a few seconds on CPU)"

        with st.spinner(spinner_msg):
            start = time.time()

            inputs = tokenizer(
                article,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
            )
            n_input_tokens = int(inputs["input_ids"].shape[1])

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat,
                    max_length=max_output,
                    early_stopping=True,
                )

            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = time.time() - start

        st.session_state["result"] = {
            "summary": summary,
            "article": article,
            "elapsed": elapsed,
            "truncated": n_input_tokens >= MAX_INPUT_LENGTH,
        }


# ---------------------------------------------------------------- Results
result = st.session_state.get("result")

if result:
    summary = result["summary"]
    source_text = result["article"]

    st.markdown(
        '<div class="section-title">Generated Summary</div>',
        unsafe_allow_html=True,
    )

    if result["truncated"]:
        st.info(
            f"The article exceeded {MAX_INPUT_LENGTH} tokens, "
            "so only the first part was summarized."
        )

    tab_summary, tab_compare = st.tabs(["Summary", "Compare with original"])

    with tab_summary:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">AI Generated Summary</div>
                <div class="result-text">{html.escape(summary)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Copy summary"):
            st.code(summary, language=None, wrap_lines=True)

    with tab_compare:
        left, right = st.columns(2)
        with left:
            st.markdown("**Original**")
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-text" style="font-size:14px;">
                        {html.escape(source_text)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("**Summary**")
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-text" style="font-size:14px;">
                        {html.escape(summary)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    input_words = len(source_text.split())
    output_words = len(summary.split())
    compression = (
        (1 - output_words / input_words) * 100 if input_words > 0 else 0
    )

    st.markdown(
        '<div class="section-title">Summary Statistics</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Input Words", f"{input_words:,}")
    col2.metric("Summary Words", f"{output_words:,}")
    col3.metric("Compression", f"{compression:.1f}%")
    col4.metric("Time", f"{result['elapsed']:.1f}s")


st.markdown(
    """
    <div class="footer">
        BART CNN/DailyMail Text Summarization
        <br>
        Fine-tuned Transformer-based abstractive summarization
    </div>
    """,
    unsafe_allow_html=True,
)
