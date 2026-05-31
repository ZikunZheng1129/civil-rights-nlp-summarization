# Civil Rights Lawsuit Summarization & Classification

**Northwestern University — MLDS 414 Text Analytics — Final Project (Option 2)**

This project builds an NLP pipeline for analyzing civil rights lawsuit documents from the Multi LexSum dataset. The system cleans legal text, generates summaries at different levels of detail, and trains machine learning models to classify cases. An interactive Streamlit app is also included for testing new legal text.

---

## Project Objective

Main components of the project:

1. Loads civil rights lawsuit data from the `allenai/multi_lexsum` dataset
2. Cleans and normalizes legal text using rule based preprocessing
3. Generates **long**, **short**, and **tiny** summaries
4. Trains text classification models on the cases
5. Provides an interactive interface for analyzing new legal text

---

## Dataset

**Multi-LexSum** is a legal NLP dataset containing civil rights lawsuits paired with expert written summaries at multiple levels of detail.

| Property | Value |
|---|---|
| Source | `allenai/multi_lexsum`, version `v20230518` |
| Project dataset size | 4,539 cases in the loaded HuggingFace `v20230518` split |
| Original resource size | 9,280 expert-authored summaries |
| Splits | train / validation / test |
| Targets | `class_action_sought` (binary), `case_type` (multiclass) |
| HuggingFace | [allenai/multi_lexsum](https://huggingface.co/datasets/allenai/multi_lexsum) |

All notebooks were executed before submission, and generated outputs are included in `outputs/`.

---

## Repository Structure

```
civil-rights-nlp-summarization/
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_text_cleaning_and_key_findings.ipynb
│   ├── 03_summarization_analysis.ipynb
│   └── 04_classification_modeling.ipynb
├── app/
│   └── streamlit_app.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── samples/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('stopwords')"
```

Python 3.9+ recommended.

---

## How to Run

### Notebooks

```bash
jupyter notebook
```

Run the notebooks from the project root directory, in order, if reproducing the analysis from scratch. Notebook 02 writes the cleaned dataset `outputs/tables/multilexsum_cleaned.parquet`, which Notebooks 03 and 04 load — this file is large and not committed, so run Notebook 02 first.

| Notebook | Description |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load data, inspect features, EDA plots |
| `02_text_cleaning_and_key_findings.ipynb` | Clean text, TF-IDF, n-grams, sentiment |
| `03_summarization_analysis.ipynb` | Generate summaries, compute ROUGE |
| `04_classification_modeling.ipynb` | Train classifiers, evaluate, save models |

### Interactive Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The saved classifiers are included in `outputs/models/`.

---

## Modeling Summary

### Summarization

- Model: sshleifer/distilbart-cnn-12-6
- Method: hierarchical chunking for long legal documents
- Outputs: long, short, and tiny summaries
- Evaluation: ROUGE based comparison against reference summaries

### Classification

| Task | Model |
|---|---|
| `class_action_sought` | TF IDF + LinearSVC |
| `case_type` (top 6 classes) | TF IDF + LinearSVC |

---

## Evaluation Summary

| Task | Metric | Score |
|---|---|---|
| Summarization (short) | ROUGE-1 | 0.3302 |
| Summarization (short) | ROUGE-L | 0.1922 |
| class_action_sought, TF-IDF + LinearSVC | Test Accuracy | 0.9482 |
| class_action_sought, TF-IDF + LinearSVC | Weighted F1 | 0.9481 |
| class_action_sought, TF-IDF + LinearSVC | Macro F1 | 0.9400 |
| case_type (top 6), TF-IDF + LinearSVC | Test Accuracy | 0.9630 |
| case_type (top 6), TF-IDF + LinearSVC | Weighted F1 | 0.9630 |
| case_type (top 6), TF-IDF + LinearSVC | Macro F1 | 0.9361 |


### Results

The strongest classification model was TF-IDF + LinearSVC.

- class_action_sought:
  - Accuracy: 94.8%
  - Macro F1: 0.94

- case_type (top 6 classes):
  - Accuracy: 96.3%
  - Macro F1: 0.936

---  

## NLP Techniques

| Technique | Where |
|---|---|
| Regular expressions | Notebook 02, Streamlit app |
| Text normalization & tokenization | Notebook 02, Streamlit app |
| TF-IDF vectorization | Notebook 02, Notebook 04 |
| N-gram analysis | Notebook 02 |
| Sentiment analysis | Notebook 02 |
| TF-IDF cosine similarity | Notebook 03 |
| Transformer-based summarization | Notebook 03 |
| Text classification | Notebook 04, Streamlit app |
| ROUGE evaluation | Notebook 03 |

