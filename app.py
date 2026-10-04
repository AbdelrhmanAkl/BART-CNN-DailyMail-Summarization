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
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, button, textarea, input {
        font-family: 'Inter', sans-serif !important;
    }

    /* ---------- Page background ---------- */
    .stApp {
        background:
            radial-gradient(900px 420px at 10% -10%, rgba(99,102,241,0.14), transparent 60%),
            radial-gradient(800px 400px at 95% 0%, rgba(168,85,247,0.12), transparent 60%),
            #fbfbff;
    }

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    /* ---------- Hero ---------- */
    .hero {
        text-align: center;
        margin: 8px auto 26px auto;
        max-width: 760px;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.18);
        color: #4338ca;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .02em;
    }

    .dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 4px rgba(34,197,94,0.18);
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.1;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        margin: 18px 0 12px 0;
    }

    .grad {
        background: linear-gradient(90deg, #4f46e5, #9333ea);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-text {
        color: #64748b;
        font-size: 17px;
        line-height: 1.7;
    }

    .tags { margin-top: 16px; }
    .tag {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #475569;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px;
    }

    /* ---------- Cards (bordered containers) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e8eaf2 !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 30px rgba(79,70,229,0.06);
        padding: 8px 10px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        margin: 4px 0 2px 0;
    }

    .card-sub {
        font-size: 13px;
        color: #94a3b8;
        margin-bottom: 10px;
    }

    /* ---------- Text area ---------- */
    div[data-testid="stTextArea"] textarea {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        color: #0f172a;
        font-size: 15px;
        line-height: 1.7;
        padding: 14px 16px;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
        background: #ffffff;
    }

    /* ---------- Buttons ---------- */
    div.stButton > button {
        border-radius: 12px;
        height: 44px;
        font-weight: 600;
        font-size: 14px;
        transition: all .15s ease;
    }

    div.stButton > button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        color: #334155;
    }

    div.stButton > button[kind="secondary"]:hover {
        border-color: #6366f1;
        color: #4f46e5;
        background: #f5f5ff;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        border: none;
        color: #ffffff;
        height: 48px;
        font-size: 15px;
        box-shadow: 0 8px 20px rgba(99,102,241,0.30);
    }

    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
        color: #ffffff;
    }

    /* ---------- Expander ---------- */
    div[data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e8eaf2;
        border-radius: 16px;
        margin-bottom: 18px;
    }

    /* ---------- Result ---------- */
    .result-box {
        background: linear-gradient(180deg, #f8f7ff, #ffffff);
        border: 1px solid #e0e0ff;
        border-radius: 16px;
        padding: 20px 22px;
        color: #1e293b;
        font-size: 16px;
        line-height: 1.85;
    }

    .empty-box {
        border: 2px dashed #e2e8f0;
        border-radius: 16px;
        padding: 60px 20px;
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        line-height: 1.8;
    }

    .empty-icon { font-size: 34px; margin-bottom: 6px; }

    .chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
    .chip {
        flex: 1 1 110px;
        background: #ffffff;
        border: 1px solid #e8eaf2;
        border-radius: 14px;
        padding: 12px 14px;
    }
    .chip-label {
        font-size: 11px; font-weight: 600; color: #94a3b8;
        text-transform: uppercase; letter-spacing: .05em;
    }
    .chip-value {
        font-size: 22px; font-weight: 700; color: #4338ca; margin-top: 2px;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 36px;
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


# -------------------------------------------------------------- Load model
try:
    with st.spinner("Loading model..."):
        tokenizer, model, device, source_label = load_model()
except Exception as error:
    st.error(f"Failed to load the model: {error}")
    st.stop()

device_label = "GPU" if device.type == "cuda" else "CPU"


# -------------------------------------------------------------------- Hero
st.markdown(
    f"""
    <div class="hero">
        <span class="pill"><span class="dot"></span>
            Model ready · {device_label} · {html.escape(source_label)}
        </span>
        <div class="hero-title">
            Turn long articles into <span class="grad">clear summaries</span>
        </div>
        <div class="hero-text">
            A BART Transformer fine-tuned on CNN/DailyMail that rewrites
            long-form English text into short, readable abstractive summaries.
        </div>
        <div class="tags">
            <span class="tag">BART-base</span>
            <span class="tag">CNN/DailyMail</span>
            <span class="tag">Abstractive</span>
            <span class="tag">Beam search</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- Settings
with st.expander("⚙️  Generation settings", expanded=False):
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        num_beams = st.slider(
            "Beam search", 1, 8, NUM_BEAMS,
            help="More beams explore more candidates: better quality, slower.",
        )
    with s2:
        max_output = st.slider(
            "Max output tokens", 32, 256, MAX_OUTPUT_LENGTH, step=16,
            help="Upper limit for the summary length.",
        )
    with s3:
        length_penalty = st.slider(
            "Length penalty", 0.5, 3.0, LENGTH_PENALTY, step=0.1,
            help="Above 1.0 favors longer summaries, below 1.0 shorter ones.",
        )
    with s4:
        no_repeat = st.slider(
            "No-repeat n-gram", 0, 5, NO_REPEAT_NGRAM_SIZE,
            help="Blocks repeating the same n-gram. 0 disables it.",
        )
    st.caption(f"Input is truncated after {MAX_INPUT_LENGTH} tokens.")


# --------------------------------------------------------- State/callbacks
if "article" not in st.session_state:
    st.session_state["article"] = ""


def load_sample():
    st.session_state["article"] = SAMPLE_ARTICLE


def clear_all():
    st.session_state["article"] = ""
    st.session_state.pop("result", None)


# ------------------------------------------------------------------ Layout
left, right = st.columns(2, gap="large")

# ------------------------------ Left: input
with left:
    with st.container(border=True):
        st.markdown('<div class="card-title">Your article</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-sub">Paste English text, or start with a sample.</div>',
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns(2)
        with b1:
            st.button("✨ Try a sample", on_click=load_sample, use_container_width=True)
        with b2:
            st.button("Clear", on_click=clear_all, use_container_width=True)

        article = st.text_area(
            "Article",
            key="article",
            height=320,
            placeholder="Paste your article here...",
            label_visibility="collapsed",
        )

        st.caption(f"{len(article.split()):,} words")

        generate = st.button(
            "Generate summary",
            type="primary",
            use_container_width=True,
        )

# ------------------------------ Generation
if generate:
    if not article.strip():
        st.warning("Please enter some text before generating a summary.")
    else:
        msg = "Generating summary..."
        if device.type == "cpu":
            msg += " (a few seconds on CPU)"

        with st.spinner(msg):
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

# ------------------------------ Right: result
with right:
    with st.container(border=True):
        st.markdown('<div class="card-title">Summary</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-sub">Generated by the fine-tuned BART model.</div>',
            unsafe_allow_html=True,
        )

        result = st.session_state.get("result")

        if not result:
            st.markdown(
                """
                <div class="empty-box">
                    <div class="empty-icon">📝</div>
                    Your summary will appear here.<br>
                    Add an article and press <b>Generate summary</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            summary = result["summary"]
            src = result["article"]

            if result["truncated"]:
                st.info(
                    f"The article exceeded {MAX_INPUT_LENGTH} tokens, "
                    "so only the first part was summarized."
                )

            st.markdown(
                f'<div class="result-box">{html.escape(summary)}</div>',
                unsafe_allow_html=True,
            )

            in_words = len(src.split())
            out_words = len(summary.split())
            compression = (1 - out_words / in_words) * 100 if in_words else 0

            st.markdown(
                f"""
                <div class="chips">
                    <div class="chip"><div class="chip-label">Input</div>
                        <div class="chip-value">{in_words:,}</div></div>
                    <div class="chip"><div class="chip-label">Summary</div>
                        <div class="chip-value">{out_words:,}</div></div>
                    <div class="chip"><div class="chip-label">Compression</div>
                        <div class="chip-value">{compression:.0f}%</div></div>
                    <div class="chip"><div class="chip-label">Time</div>
                        <div class="chip-value">{result['elapsed']:.1f}s</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Copy summary"):
                st.code(summary, language=None)


st.markdown(
    """
    <div class="footer">
        BART · CNN/DailyMail · Abstractive text summarization
    </div>
    """,
    unsafe_allow_html=True,
)
