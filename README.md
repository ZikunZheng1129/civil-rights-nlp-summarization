# Civil Rights Lawsuit Summarization & Classification

**Northwestern University — MLDS 414 Text Analytics — Final Project (Option 2)**

An end-to-end NLP pipeline that loads, cleans, summarizes, and classifies civil rights lawsuits from the Multi-LexSum dataset, with an interactive Streamlit demo.

---

## Project Goal

Build a reproducible NLP system that:

1. Loads the `allenai/multi_lexsum` legal case dataset
2. Cleans legal text using domain-aware regex normalization
3. Generates **long**, **short**, and **tiny** summaries using a transformer model with hierarchical chunking
4. Trains two text classifiers on the cases
5. Provides an interactive tool for analyzing new case text

---

## Dataset

**Multi-LexSum** — civil rights lawsuits with three levels of human-written summaries

| Property | Value |
|---|---|
| Source | `allenai/multi_lexsum`, version `v20230518` |
| Project dataset size | 4,539 cases in the loaded HuggingFace `v20230518` split |
| Original resource size | 9,280 expert-authored summaries |
| Splits | train / validation / test |
| Targets | `class_action_sought` (binary), `case_type` (multiclass) |
| HuggingFace | [allenai/multi_lexsum](https://huggingface.co/datasets/allenai/multi_lexsum) |

Raw data is loaded at runtime via HuggingFace — see `data/README.md`.

The notebooks have been run and their output artifacts are included locally under
`outputs/`.

---

## Repository Structure

```
final_project/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb          # Load data, EDA, distributions
│   ├── 02_text_cleaning_and_key_findings.ipynb # Cleaning, TF-IDF, n-grams, sentiment
│   ├── 03_summarization_analysis.ipynb         # Summarization pipeline + evaluation
│   └── 04_classification_modeling.ipynb        # Two classifiers + confusion matrices
├── app/
│   └── streamlit_app.py   # Interactive demo
├── outputs/
│   ├── figures/           # Saved plots
│   ├── tables/            # Parquet DataFrames
│   ├── models/            # Saved classifiers (joblib)
│   └── samples/           # Generated summaries CSV
└── data/
    └── README.md          # Data loading instructions
```

---

## Setup

```bash
# Clone / navigate to the repo
cd final_project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (needed for stopwords in Notebook 02)
python3 -c "import nltk; nltk.download('stopwords')"
```

Python 3.9+ recommended.

---

## How to Run

### Notebooks

```bash
jupyter lab
# or
jupyter notebook
```

The notebooks have already been run for this project. Open them to inspect the
saved code, markdown, and outputs. If regenerating the analysis from scratch,
run them sequentially:

| Notebook | Description |
|---|---|
| `01_data_loading_and_eda.ipynb` | Load data, inspect features, EDA plots |
| `02_text_cleaning_and_key_findings.ipynb` | Clean text, TF-IDF, n-grams, word cloud, sentiment |
| `03_summarization_analysis.ipynb` | Generate summaries, compute ROUGE |
| `04_classification_modeling.ipynb` | Train classifiers, evaluate, save models |

### Interactive Streamlit App

The saved classifiers are included in `outputs/models/`. To launch the app:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Modeling Summary

### Summarization

- **Model:** `sshleifer/distilbart-cnn-12-6` (DistilBART fine-tuned on CNN/DailyMail)
- **Strategy:** Hierarchical chunking (768-token chunks, 80-token overlap) → chunk summaries → final pass
- **Outputs:** Long, short, and tiny generated summaries compared with dataset references
- **Notebook 3 demo:** 5 selected cases with all three reference summaries
- **Streamlit app:** lightweight extractive demo summaries for local interactivity; it does not rerun DistilBART

### Classification

| Model | Algorithm | Feature | Target |
|---|---|---|---|
| Model 1 | TF-IDF + LinearSVC | Cleaned case text | `class_action_sought` |
| Model 2 | TF-IDF + LinearSVC | Cleaned case text | `case_type` (top 6) |

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

---

## NLP Techniques Demonstrated

| Technique | Where |
|---|---|
| Regular expressions | Notebook 02, Streamlit app |
| Text normalization / tokenization | Notebook 02, Streamlit app |
| TF-IDF vectorization | Notebook 02, Notebook 04 |
| N-gram analysis | Notebook 02 |
| Sentiment analysis (TextBlob) | Notebook 02 |
| TF-IDF cosine similarity | Notebook 03 |
| Transformer-based summarization | Notebook 03 |
| Text classification | Notebook 04, Streamlit app |
| ROUGE evaluation | Notebook 03 |

---

## Limitations

1. **Summarization model:** `distilbart-cnn-12-6` was trained on news, not legal text — expect hallucinations on domain-specific terminology.
2. **Chunking:** Cross-chunk dependencies may be lost; information from early document sections may not appear in summaries.
3. **Classification:** Bag-of-words TF-IDF ignores word order; transformer-based encoders (Legal-BERT) would improve accuracy.
4. **Data:** Rare case types are excluded from Model 2 due to insufficient training examples.
5. **TextBlob sentiment:** Not calibrated for formal legal language — for exploratory demonstration only.

---

## Future Improvements

- Replace DistilBART with a legal-domain summarizer (e.g., LexRank, LED-base-16384)
- Fine-tune a classification model using Legal-BERT or a similar legal LM
- Add active learning to handle rare case types
- Implement retrieval-augmented generation (RAG) for case Q&A
- Add a proper evaluation harness with multiple reference summaries

---

## Educational Disclaimer

> This project is created for academic purposes as part of MLDS 414 Text Analytics at Northwestern University. The models, summaries, and predictions produced by this system are for educational NLP demonstration only and are **not legal advice**. Do not rely on any output of this system for legal decisions.

---

## Citation

```bibtex
@inproceedings{shen-etal-2022-multi,
    title = "Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities",
    author = "Shen, Zejiang and Lo, Kyle and Yu, Lauren and Dahlberg, Nathan and Bailey, Margo Schlanger and Downey, Doug",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2022",
}
```
