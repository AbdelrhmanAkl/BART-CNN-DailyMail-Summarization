import streamlit as st
import torch
from pathlib import Path
from transformers import BartForConditionalGeneration, BartTokenizer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LOCAL_MODEL_PATH = PROJECT_ROOT / "model"
HF_MODEL_ID = "AbdelrahmanAkl/bart-cnn-dailymail-summarization"

MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128

NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="BART Text Summarization",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #666;
            margin-bottom: 2rem;
        }

        .summary-box {
            padding: 1.25rem;
            border-radius: 0.75rem;
            border: 1px solid #ddd;
            background-color: #fafafa;
            line-height: 1.7;
        }

        .metric-label {
            font-size: 0.85rem;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

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


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">BART Text Summarization</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Abstractive text summarization powered by a fine-tuned "
    "BART model trained on CNN/DailyMail."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Model Information")

    st.write("**Architecture:** BART")
    st.write("**Base Model:** facebook/bart-base")
    st.write("**Dataset:** CNN/DailyMail")
    st.write("**Task:** Abstractive Summarization")

    st.divider()

    st.header("Generation Settings")

    st.write(f"**Beam Search:** {NUM_BEAMS}")
    st.write(f"**Length Penalty:** {LENGTH_PENALTY}")
    st.write(
        f"**No Repeat N-gram:** {NO_REPEAT_NGRAM_SIZE}"
    )
    st.write(
        f"**Max Input Tokens:** {MAX_INPUT_LENGTH}"
    )
    st.write(
        f"**Max Output Tokens:** {MAX_OUTPUT_LENGTH}"
    )


# ============================================================
# Load Model
# ============================================================

try:
    tokenizer, model, device, source_label = load_model()

except Exception as error:
    st.error(f"Failed to load the model: {error}")
    st.stop()


# ============================================================
# Device Status
# ============================================================

if device.type == "cuda":
    st.success(
        f"Model loaded successfully on GPU ({source_label})."
    )
else:
    st.info(
        f"Model loaded successfully on CPU ({source_label})."
    )


# ============================================================
# Input
# ============================================================

st.subheader("Input Article")

article = st.text_area(
    "Enter article text",
    height=320,
    placeholder=(
        "Paste an article or long-form text here..."
    ),
    label_visibility="collapsed",
)


# ============================================================
# Summarization
# ============================================================

if st.button(
    "Generate Summary",
    type="primary",
    use_container_width=True,
):

    if not article.strip():
        st.warning(
            "Please enter some text before generating a summary."
        )
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


    # ========================================================
    # Results
    # ========================================================

    st.subheader("Generated Summary")

    st.markdown(
        f'<div class="summary-box">{summary}</div>',
        unsafe_allow_html=True,
    )


    # ========================================================
    # Statistics
    # ========================================================

    input_words = len(article.split())
    output_words = len(summary.split())

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
        if input_words > 0:
            compression = (
                1 - output_words / input_words
            ) * 100

            st.metric(
                "Compression",
                f"{compression:.1f}%",
            )
        else:
            st.metric(
                "Compression",
                "N/A",
            )