from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import joblib
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ACTION_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "class_action_model.joblib"
CASE_TYPE_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "case_type_model.joblib"

DISCLAIMER = "This tool is for educational NLP demonstration only and is not legal advice."

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "will", "with",
    "court", "case", "civil", "action", "plaintiff", "defendant", "document",
    "filed", "page", "states", "united",
}


def clean_for_summary(text: str | None) -> str:
    """
    Light legal-text cleaning, matching Notebook 2's summarization cleaner.
    """
    if text is None:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"Document\s+\d+(-\d+)?\s+Filed\s+\d{1,2}/\d{1,2}/\d{2,4}",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"Case\s+\d+:\d{2}-cv-[\w\-]+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_for_modeling(text: str | None) -> str:
    """
    TF-IDF modeling cleaner from Notebook 2.
    """
    cleaned = clean_for_summary(text).lower()
    cleaned = re.sub(r"\bno\.\s*\d+:\d{2}-cv-[\w\-]+\b", " docket_id ", cleaned)
    cleaned = re.sub(r"\bcase\s+\d+:\d{2}-cv-[\w\-]+\b", " docket_id ", cleaned)
    cleaned = re.sub(r"u\.s\.c\.", "usc", cleaned)
    cleaned = re.sub(r"§", " section ", cleaned)
    cleaned = re.sub(r"[_]{2,}", " ", cleaned)
    cleaned = re.sub(r"[-]{3,}", " ", cleaned)
    cleaned = re.sub(r"\.{3,}", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def split_sentences(text: str) -> list[str]:
    text = clean_for_summary(text)
    if not text:
        return []

    protected = {
        "U.S.": "US_ABBREV",
        "D.C.": "DC_ABBREV",
        "Inc.": "INC_ABBREV",
        "No.": "NO_ABBREV",
        "v.": "V_ABBREV",
    }
    for original, replacement in protected.items():
        text = text.replace(original, replacement)

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for sentence in raw_sentences:
        for original, replacement in protected.items():
            sentence = sentence.replace(replacement, original)
        sentence = sentence.strip()
        if len(sentence.split()) >= 6:
            sentences.append(sentence)
    return sentences


def sentence_scores(sentences: list[str]) -> dict[int, float]:
    tokens = []
    for sentence in sentences:
        tokens.extend(
            token
            for token in re.findall(r"[A-Za-z][A-Za-z0-9']+", sentence.lower())
            if token not in STOPWORDS and len(token) > 2
        )

    frequencies = Counter(tokens)
    if not frequencies:
        return {index: 0.0 for index in range(len(sentences))}

    return {
        index: sum(frequencies[token] for token in re.findall(r"[A-Za-z][A-Za-z0-9']+", sentence.lower()))
        / max(len(sentence.split()), 1)
        for index, sentence in enumerate(sentences)
    }


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def extractive_summary(text: str, sentence_count: int, max_words: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    scores = sentence_scores(sentences)
    selected_indexes = sorted(
        sorted(scores, key=scores.get, reverse=True)[:sentence_count]
    )
    summary = " ".join(sentences[index] for index in selected_indexes)
    return truncate_words(summary, max_words)


def predict_label(model, cleaned_text: str) -> str | None:
    if model is None or not cleaned_text:
        return None
    return str(model.predict([cleaned_text])[0])


st.set_page_config(page_title="Civil Rights Lawsuit NLP Demo", layout="wide")

st.title("Civil Rights Lawsuit NLP Demo")
st.caption(DISCLAIMER)

class_action_model = load_model(CLASS_ACTION_MODEL_PATH)
case_type_model = load_model(CASE_TYPE_MODEL_PATH)

missing_models = []
if class_action_model is None:
    missing_models.append(str(CLASS_ACTION_MODEL_PATH.relative_to(PROJECT_ROOT)))
if case_type_model is None:
    missing_models.append(str(CASE_TYPE_MODEL_PATH.relative_to(PROJECT_ROOT)))

if missing_models:
    st.warning(
        "Some saved model files are missing. Predictions will be unavailable until these files exist: "
        + ", ".join(missing_models)
    )

case_text = st.text_area(
    "Paste legal case text",
    height=300,
    placeholder="Paste a real or fictional civil-rights complaint, order, or case description here.",
)

analyze = st.button("Analyze case text", type="primary")

if analyze:
    if not case_text.strip():
        st.error("Please paste case text before analyzing.")
        st.stop()

    summary_text = clean_for_summary(case_text)
    modeling_text = clean_for_modeling(case_text)

    st.subheader("Model Predictions")
    col1, col2 = st.columns(2)

    class_action_prediction = predict_label(class_action_model, modeling_text)
    case_type_prediction = predict_label(case_type_model, modeling_text)

    with col1:
        st.metric("class_action_sought", class_action_prediction or "Model unavailable")
    with col2:
        st.metric("case_type", case_type_prediction or "Model unavailable")

    st.subheader("Demo Extractive Summaries")
    st.caption("These summaries use sentence selection/truncation for a lightweight local demo. They are not transformer-generated.")

    long_summary = extractive_summary(summary_text, sentence_count=8, max_words=250)
    short_summary = extractive_summary(summary_text, sentence_count=3, max_words=100)
    tiny_summary = extractive_summary(summary_text, sentence_count=1, max_words=35)

    st.markdown("**Long demo summary**")
    st.write(long_summary or "No summary could be generated from the pasted text.")

    st.markdown("**Short demo summary**")
    st.write(short_summary or "No summary could be generated from the pasted text.")

    st.markdown("**Tiny demo summary**")
    st.write(tiny_summary or "No summary could be generated from the pasted text.")

    with st.expander("Cleaned text preview"):
        st.markdown("**Summary/display cleaning**")
        st.text(summary_text[:3000])
        st.markdown("**Modeling cleaning**")
        st.text(modeling_text[:3000])

st.divider()
st.caption(DISCLAIMER)
