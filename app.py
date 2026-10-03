from pathlib import Path
import html

import streamlit as st
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = PROJECT_ROOT / "model"
HF_MODEL_ID = "AbdelrahmanAkl/bart-cnn-dailymail-summarization"

MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128

NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3


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
        padding-top: 2rem;
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
        padding: 30px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.04);
    }

    .badge {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 6px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .hero-text {
        color: #6b7280;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 10px;
        max-width: 800px;
    }

    .tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        color: #475569;
        padding: 6px 10px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 12px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 6px;
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
        height: 48px;
        border-radius: 10px;
        background-color: #111827;
        border: 1px solid #111827;
        color: #ffffff;
        font-size: 15px;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #1f2937;
        border-color: #1f2937;
        color: #ffffff;
    }

    .status {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 20px;
        color: #4b5563;
        font-size: 13px;
    }

    .result-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 22px;
        margin-top: 10px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.03);
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

    model = BartForConditionalGeneration.from_pretrained(
        model_source
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device, source_label


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
    st.write("**Dataset:** CNN/DailyMail")
    st.write("**Task:** Text Summarization")

    st.markdown("### Generation")

    st.write(f"**Beam Search:** {NUM_BEAMS}")
    st.write(f"**Length Penalty:** {LENGTH_PENALTY}")
    st.write(f"**No Repeat N-gram:** {NO_REPEAT_NGRAM_SIZE}")
    st.write(f"**Max Input:** {MAX_INPUT_LENGTH} tokens")
    st.write(f"**Max Output:** {MAX_OUTPUT_LENGTH} tokens")


st.markdown(
    """
    <div class="hero">
        <div class="badge">NLP · TRANSFORMER · SUMMARIZATION</div>
        <div class="hero-title">BART Text Summarization</div>
        <div class="hero-text">
            Transform long-form articles into concise summaries using
            a fine-tuned BART Transformer model trained on the
            CNN/DailyMail dataset.
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
    tokenizer, model, device, source_label = load_model()
except Exception as error:
    st.error(f"Failed to load the model: {error}")
    st.stop()


device_label = "GPU" if device.type == "cuda" else "CPU"

st.markdown(
    f"""
    <div class="status">
        ● Model ready · Running on {device_label} · Source: {source_label}
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="section-title">Article Input</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-text">
        Paste an article or long-form English text below.
    </div>
    """,
    unsafe_allow_html=True,
)


article = st.text_area(
    "Article",
    height=300,
    placeholder="Paste your article here...",
    label_visibility="collapsed",
)


if article.strip():
    word_count = len(article.split())

    st.caption(
        f"{word_count:,} words entered · "
        f"Maximum input length: {MAX_INPUT_LENGTH} tokens"
    )


if st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True,
):
    if not article.strip():
        st.warning("Please enter some text before generating a summary.")
        st.stop()

    with st.spinner("Generating summary..."):
        inputs = tokenizer(
            article,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                max_length=MAX_OUTPUT_LENGTH,
                early_stopping=True,
            )

        summary = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

    safe_summary = html.escape(summary)

    st.markdown(
        '<div class="section-title">Generated Summary</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">AI Generated Summary</div>
            <div class="result-text">
                {safe_summary}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_words = len(article.split())
    output_words = len(summary.split())

    if input_words > 0:
        compression = (
            1 - (output_words / input_words)
        ) * 100
    else:
        compression = 0

    st.markdown(
        '<div class="section-title">Summary Statistics</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Input Words",
            f"{input_words:,}",
        )

    with col2:
        st.metric(
            "Summary Words",
            f"{output_words:,}",
        )

    with col3:
        st.metric(
            "Compression",
            f"{compression:.1f}%",
        )


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
