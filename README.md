# BART CNN/DailyMail Text Summarization

> An end-to-end abstractive text summarization system built with a fine-tuned BART Transformer on the CNN/DailyMail dataset, evaluated against the original pretrained BART baseline and deployed as an interactive Streamlit application.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://bart-cnn-dailymail-summarization.streamlit.app/)

[![Model](https://img.shields.io/badge/Model-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/AbdelrahmanAkl/bart-cnn-dailymail-summarization)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/AbdelrhmanAkl/BART-CNN-DailyMail-Summarization)

---

## Project Overview

This project implements an **abstractive text summarization pipeline** using Facebook's BART Transformer architecture.

The model was fine-tuned on a curated subset of the **CNN/DailyMail 3.0.0** dataset and evaluated on a held-out test set using standard ROUGE metrics.

The project covers the complete machine learning workflow:

* Dataset exploration and preprocessing
* Transformer tokenization
* BART fine-tuning
* Validation and checkpoint selection
* Baseline comparison
* Generation strategy optimization
* Quantitative ROUGE evaluation
* Model packaging and integrity validation
* Hugging Face model hosting
* Streamlit deployment
* Local and cloud inference testing

---

## Live Demo

### Try the Application

**[Launch the Streamlit Demo](https://bart-cnn-dailymail-summarization.streamlit.app/)**

The application allows users to:

1. Enter an article or long-form text
2. Generate an abstractive summary
3. View the generated summary
4. Inspect input and output word counts
5. View the resulting compression ratio

The deployed application loads the fine-tuned model directly from Hugging Face when the local model directory is unavailable.

---

## Model

| Component             | Configuration                  |
| --------------------- | ------------------------------ |
| Architecture          | BART                           |
| Base Model            | `facebook/bart-base`           |
| Parameters            | 139,420,416                    |
| Task                  | Abstractive Text Summarization |
| Dataset               | CNN/DailyMail 3.0.0            |
| Input Length          | 1024 tokens                    |
| Target Length         | 128 tokens                     |
| Training Epochs       | 2                              |
| Learning Rate         | `5e-5`                         |
| Train Batch Size      | 2                              |
| Gradient Accumulation | 8                              |
| Effective Batch Size  | 16                             |
| Weight Decay          | 0.01                           |
| Training Framework    | Hugging Face `Seq2SeqTrainer`  |
| Precision             | FP16                           |
| Random Seed           | 42                             |

---

## Dataset

The project uses the **CNN/DailyMail 3.0.0** dataset.

The complete dataset contains:

* Training: 287,113 examples
* Validation: 13,368 examples
* Test: 11,490 examples

For this portfolio project, a controlled subset was used:

| Split      | Examples |
| ---------- | -------: |
| Train      |    8,000 |
| Validation |    1,000 |
| Test       |    1,000 |

The subset was shuffled using a fixed seed of `42` to maintain reproducibility.

### Preprocessing

Articles were tokenized with a maximum length of **1024 BART tokens**, while reference summaries were limited to **128 tokens**.

Long articles exceeding the input limit were truncated during tokenization.

---

## Training

The model was fine-tuned for two epochs using:

* Batch size: 2
* Gradient accumulation: 8
* Effective batch size: 16
* Learning rate: `5e-5`
* Weight decay: `0.01`
* Warmup steps: 100
* FP16 training
* Gradient checkpointing
* Evaluation every 500 steps
* Best checkpoint selected using validation loss

### Best Checkpoint

The selected checkpoint was:

```text
checkpoint-1000
```

Validation loss:

```text
1.757423
```

The checkpoint was independently reloaded and validated before packaging.

---

## Baseline Comparison

The evaluation was designed to separate the effect of **model fine-tuning** from the effect of **generation optimization**.

### 1. Fine-tuning Impact

The original pretrained `facebook/bart-base` was compared with the fine-tuned model using the original generation configuration.

| Metric     | Base BART | Original Fine-tuned | Absolute Change |
| ---------- | --------: | ------------------: | --------------: |
| ROUGE-1    |  0.392200 |            0.406423 |       +0.014223 |
| ROUGE-2    |  0.176266 |            0.180911 |       +0.004645 |
| ROUGE-L    |  0.245521 |            0.276005 |       +0.030484 |
| ROUGE-Lsum |  0.319560 |            0.374562 |       +0.055002 |

These results show the change associated with **fine-tuning the pretrained BART model** under the original generation setup.

### 2. Generation Optimization Impact

After fine-tuning, generation parameters were optimized using the validation set.

The final optimized configuration produced the following test results:

| Metric     | Original Fine-tuned | Final Optimized | Absolute Change |
| ---------- | ------------------: | --------------: | --------------: |
| ROUGE-1    |            0.406423 |        0.407043 |       +0.000620 |
| ROUGE-2    |            0.180911 |        0.181450 |       +0.000539 |
| ROUGE-L    |            0.276005 |        0.276308 |       +0.000303 |
| ROUGE-Lsum |            0.374562 |        0.374985 |       +0.000423 |

This second comparison isolates the effect of the selected generation configuration from the effect of model fine-tuning.

### Interpretation

The final fine-tuned model achieved higher ROUGE scores than the original pretrained baseline under the same evaluation framework.

The largest fine-tuning-related improvement was observed in **ROUGE-Lsum**.

ROUGE measures lexical overlap with reference summaries and should not be interpreted as a direct measure of factual accuracy, coherence, or overall summary quality.

---

## Generation Optimization

Generation parameters were tuned on the **validation set**, keeping the test set isolated for final evaluation.

The evaluated configurations included:

* Beam search with 4 beams
* Different length penalties
* `no_repeat_ngram_size=3`
* Combined generation constraints

The selected configuration was:

```text
num_beams = 4
length_penalty = 1.0
no_repeat_ngram_size = 3
max_length = 128
early_stopping = True
```

The validation experiment showed that preventing repeated 3-grams produced the strongest validation ROUGE-Lsum among the tested configurations.

The final test evaluation was then performed using this configuration.

---

## Final Evaluation

### Final Test Results

```text
ROUGE-1     0.407043
ROUGE-2     0.181450
ROUGE-L     0.276308
ROUGE-Lsum  0.374985
```

Evaluation was performed on:

```text
1,000 held-out CNN/DailyMail test examples
```

using the final generation configuration.

---

## Example

### Input

```text
Artificial intelligence is transforming the way organizations operate across industries. Companies are increasingly adopting machine learning and natural language processing systems to automate repetitive tasks, analyze large volumes of information, and support employees in making faster decisions. In healthcare, AI systems can help doctors analyze medical images and identify patterns that may require further investigation.
```

### Generated Summary

```text
Artificial intelligence is transforming the way organizations operate across industries. Companies are increasingly adopting machine learning and natural language processing systems to automate repetitive tasks, analyze large volumes of information, and support employees in making faster decisions. In healthcare, AI systems can help doctors analyze medical images and identify patterns that may require further investigation.
```

This example demonstrates the deployed inference pipeline. The generated output is not intended to represent benchmark-level quality on its own; quantitative performance is reported using the full 1,000-example test evaluation above.

---

## Deployment

The application is deployed using:

* **Streamlit Community Cloud**
* **Hugging Face Hub**

### Deployment Architecture

```text
User
  │
  ▼
Streamlit Application
  │
  ├── Local model available?
  │       │
  │       ├── Yes → Load local model
  │       │
  │       └── No
  │
  ▼
Hugging Face Hub
  │
  ▼
Fine-tuned BART
  │
  ▼
Generated Summary
```

This architecture allows the same application to run locally with the packaged model while also supporting lightweight cloud deployment through the hosted Hugging Face model.

---

## Hugging Face Model

The fine-tuned model is hosted on Hugging Face:

**[AbdelrahmanAkl/bart-cnn-dailymail-summarization](https://huggingface.co/AbdelrahmanAkl/bart-cnn-dailymail-summarization)**

The repository contains:

* Fine-tuned BART weights
* Model configuration
* Tokenizer configuration
* Generation configuration

---

## Local Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AbdelrhmanAkl/BART-CNN-DailyMail-Summarization.git

cd BART-CNN-DailyMail-Summarization
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available locally through the Streamlit URL shown in the terminal.

---

## Project Structure

```text
BART-CNN-DailyMail-Summarization/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── evaluation/
│   ├── metrics.json
│   └── generation_config.json
│
├── notebooks/
│   └── BART_CNN_DailyMail_Text_Summarization.ipynb
│
└── assets/
```

The fine-tuned model weights are hosted separately on Hugging Face rather than committed directly to GitHub.

---

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* BART
* Hugging Face Datasets
* ROUGE
* Pandas
* Streamlit
* Hugging Face Hub
* Jupyter / Google Colab

---

## Engineering Highlights

This project demonstrates practical experience with:

* Transformer-based NLP
* Sequence-to-sequence modeling
* Abstractive summarization
* Fine-tuning pretrained language models
* GPU-based training
* Mixed-precision training
* Gradient accumulation
* Gradient checkpointing
* Checkpoint management
* Evaluation methodology
* Baseline benchmarking
* Generation optimization
* Model serialization
* Hugging Face model hosting
* Streamlit application development
* Cloud deployment

---

## Limitations

* The training subset contains 8,000 examples rather than the full CNN/DailyMail training set.
* Articles longer than 1024 tokens are truncated.
* ROUGE evaluates lexical overlap and does not fully measure factual consistency, coherence, or hallucination.
* The model was trained and evaluated primarily in English.
* CPU inference can be slower for long inputs.
* The current application is designed as a portfolio demonstration rather than a production-scale summarization service.

---

## Future Improvements

Potential extensions include:

* Fine-tuning on the full CNN/DailyMail training set
* Evaluating larger BART variants
* Factual consistency evaluation
* Hallucination detection
* Semantic evaluation using BERTScore
* Longer-context summarization
* Batch summarization for multiple documents
* PDF and document upload support
* API deployment
* Production inference optimization
* Quantization for lower-resource deployment

---

## Resources

* **Live Demo:** https://bart-cnn-dailymail-summarization.streamlit.app/
* **GitHub:** https://github.com/AbdelrhmanAkl/BART-CNN-DailyMail-Summarization
* **Hugging Face Model:** https://huggingface.co/AbdelrahmanAkl/bart-cnn-dailymail-summarization

---

## Author

### Eng. Abdelrahman Akl

AI Engineer | NLP | LLMs | Agentic AI | Deep Learning

* **GitHub:** https://github.com/AbdelrhmanAkl
* **LinkedIn:** https://www.linkedin.com/in/abdelrahmanakl/

---

## License

This repository is intended for educational and portfolio purposes.
