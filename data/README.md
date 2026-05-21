# data/

Raw dataset files are **not committed to this repository**.

The Multi-LexSum dataset is loaded at runtime via the HuggingFace `datasets` library
and cached locally by HuggingFace's default cache mechanism.

The original Multi-LexSum resource contains 9,280 expert-authored summaries. This
project loads the HuggingFace `v20230518` split, which contains 4,539 civil
rights cases.

## Load the dataset

```python
from datasets import load_dataset

multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20230518")
```

This requires `datasets` to be installed (`pip install datasets`).  
The first run downloads ~2 GB; subsequent runs use the local cache.

## Processed outputs

Small processed outputs are committed to the repository under `outputs/`:

- `outputs/tables/*.csv` — EDA, cleaning, summarization, and modeling result tables
- `outputs/samples/sample_summaries.csv` — generated summaries from Notebook 03
- `outputs/figures/` — all plots
- `outputs/models/*.joblib` — trained classifiers used by the Streamlit app

The cleaned dataset `outputs/tables/multilexsum_cleaned.parquet` is produced by
Notebook 02 and consumed by Notebooks 03–04. It is large (~2 GB) and is **not
committed** — regenerate it by running Notebook 02. Notebook 01 does not save a
parquet file; it writes lightweight CSV summaries instead.

## Dataset citation

```
@inproceedings{shen-etal-2022-multi,
    title = "Multi-LexSum: Real-World Summaries of Civil Rights Lawsuits at Multiple Granularities",
    author = "Shen, Zejiang and Lo, Kyle and Yu, Lauren and Dahlberg, Nathan and Bailey, Margo Schlanger and Downey, Doug",
    booktitle = "Advances in Neural Information Processing Systems",
    year = "2022",
}
```
