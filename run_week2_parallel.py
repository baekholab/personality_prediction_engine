"""
run_week2_parallel.py

Week 2: Feature Engineering (Parallelized)
Memory-optimized to avoid OOM kills by chunking, reducing parallelism, and sampling.

This version also:
  • Computes vocabulary metrics on TRUE words only (no URLs, emojis, punctuation)
  • Computes MTLD on the cleaned word stream (more faithful lexical diversity)
  • Fits BERTopic on a SAMPLE for efficiency, then TRANSFORMS the FULL CORPUS
  • Keeps GPU embedding batches small and predictable to avoid memory spikes
  • Prints lightweight step timing so you can follow progress in logs
"""

# ──────────────── Standard library imports ────────────────
import os                                  # for creating directories and checking file paths
from pathlib import Path                   # object-oriented filesystem path handling
import re                                  # regular expressions, used in style feature extraction
import time                                # for simple timing
from contextlib import contextmanager       # to build a tiny timing context manager

# ──────────────── Third-party imports ────────────────
import pandas as pd                         # DataFrame operations and fast I/O
import numpy as np                          # numerical computations and array handling

import nltk                                 # Natural Language Toolkit for NLP utilities
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER sentiment analyzer

import spacy                                # spaCy for fast, industrial-strength NLP
from spacy.cli import download as spacy_download             # helper to install spaCy models on the fly

from joblib import Parallel, delayed        # easy parallel loops across multiple CPU cores

import torch                                # to detect GPU availability for BERTopic embeddings

from lexicalrichness import LexicalRichness # computes MTLD lexical diversity metric
import emoji                                # utilities to extract and count emojis in text

from sentence_transformers import SentenceTransformer  # transformer-based embeddings (GPU-capable)
from bertopic import BERTopic               # topic modeling framework built on transformers + clustering
from umap import UMAP                       # dimensionality reduction for clustering
from hdbscan import HDBSCAN                 # density-based clustering algorithm

from sklearn.decomposition import PCA       # principal component analysis for dimensionality reduction
from tqdm import tqdm                       # progress bars for pandas operations

tqdm.pandas()  # enable df['col'].progress_map() with a progress bar


# ──────────────── Logging helper ────────────────
@contextmanager
def timed(label: str):
    """
    Tiny context manager to print [START]/[END] messages with elapsed seconds.
    Usage:
        with timed("Step name"):
            ... do work ...
    This makes your logs readable while adding near-zero overhead.
    """
    t0 = time.time()
    print(f"[START] {label}", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[END]   {label} in {dt:.1f}s", flush=True)


# ──────────────── Utility functions ────────────────

def ensure_nltk_resources():
    """
    Ensure required NLTK data is available:
      - 'punkt' for tokenization
      - 'vader_lexicon' for VADER sentiment
    """
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Check if the spaCy model is installed; download if missing.
    """
    try:
        spacy.load(model_name)  # attempt to load
    except OSError:
        print(f"Model '{model_name}' not found—downloading now...")
        spacy_download(model_name)  # install the model


# ──────────────── Feature-engineering functions ────────────────

def compute_vocab_features(df: pd.DataFrame, text_col: str = 'text', n_process: int = 2) -> pd.DataFrame:
    """
    Vocabulary & lexical diversity on TRUE words only.

    Steps (and why):
      1) Tokenize in parallel with spaCy using a lightweight pipeline.  → Fast + memory-friendly.
      2) Keep ONLY alphabetic tokens (tok.is_alpha).                    → Drops URLs/emojis/numbers/punct so counts reflect real words.
      3) Compute lexical metrics: word_count, unique_word_count, TTR, avg_word_length.
      4) Compute MTLD on the CLEANED word stream (joined tokens).       → Length-stable lexical diversity, unaffected by links/punct.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # tokenizer-only pipeline

    texts = df[text_col].tolist()
    docs = list(nlp.pipe(texts, n_process=n_process, batch_size=500))

    # Clean, lowercased alphabetic tokens only
    df['tokens'] = [[tok.text.lower() for tok in doc if tok.is_alpha] for doc in docs]

    # Basic lexical metrics
    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))
    df['ttr'] = df.apply(
        lambda row: row['unique_word_count'] / row['word_count'] if row['word_count'] > 0 else 0,
        axis=1
    )
    df['avg_word_length'] = df['tokens'].map(lambda toks: np.mean([len(w) for w in toks]) if toks else 0)

    # MTLD on the cleaned tokens (joined back into a string)
    def mtld_metric_from_tokens(tok_list) -> float:
        if not tok_list:
            return 0.0
        try:
            return LexicalRichness(" ".join(tok_list)).mtld()
        except ZeroDivisionError:
            return 0.0

    df['mtld'] = Parallel(n_jobs=n_process)(
        delayed(mtld_metric_from_tokens)(toks) for toks in df['tokens']
    )

    df.drop(columns=['tokens'], inplace=True)  # drop temp to save memory
    return df


def compute_sentiment_features(df: pd.DataFrame, text_col: str = 'text', n_jobs: int = 2) -> pd.DataFrame:
    """
    Sentiment per comment + user-level volatility.

    Steps:
      1) Run VADER polarity in parallel → sent_neg/neu/pos/comp.
      2) If a user column exists, compute mean/std of compound per user (emotional “volatility”).
    """
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].tolist()

    scores = Parallel(n_jobs=n_jobs)(delayed(analyzer.polarity_scores)(txt) for txt in texts)
    sent_df = pd.DataFrame(scores, index=df.index).rename(columns={
        'neg': 'sent_neg', 'neu': 'sent_neu', 'pos': 'sent_pos', 'compound': 'sent_comp'
    })
    df = pd.concat([df, sent_df], axis=1)

    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean', 'std']).rename(columns={
            'mean': 'sent_comp_mean', 'std': 'sent_comp_std'
        })
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Paralinguistic style cues (cheap to compute):
      - emoji_count: number of emojis in raw text
      - all_caps_count: how often user SHOUTS (ALL CAPS tokens)
      - excl_count / quest_count: punctuation intensity
      - all_caps_ratio = all_caps_count / word_count (from vocab step)
    We compute on RAW text so emoji/punct remain style signals, separate from lexical diversity.
    """
    df['emoji_count'] = df[text_col].map(lambda txt: len(emoji.emoji_list(txt)))
    df['all_caps_count'] = df[text_col].map(lambda txt: len(re.findall(r"\b[A-Z]{2,}\b", txt)))
    df['excl_count'] = df[text_col].map(lambda txt: txt.count('!'))
    df['quest_count'] = df[text_col].map(lambda txt: txt.count('?'))

    df['all_caps_ratio'] = df.apply(
        lambda row: row['all_caps_count'] / row['word_count'] if row['word_count'] > 0 else 0,
        axis=1
    )
    return df


def compute_topic_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    sample_size: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Topic modeling with BERTopic (efficient + full coverage):
      • FIT on a SAMPLE (limits UMAP/HDBSCAN memory)
      • TRANSFORM the FULL CORPUS to assign a topic to every comment
    """
    # 1) Sample for fitting the cluster structure
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    sample_texts = sample[text_col].tolist()

    # 2) Configure UMAP & HDBSCAN with modest parallelism (memory-safe)
    umap_model = UMAP(n_neighbors=15, n_components=5, n_jobs=2)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=2)

    # 3) Use GPU for BERTopic’s internal embeddings if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    tm = BERTopic(
        embedding_model=embedding_model,   # GPU for embeddings if available
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        low_memory=True,                   # chunk internal steps to lower RAM/VRAM
        calculate_probabilities=False,     # skip heavy prob matrix
        verbose=False
    )

    # 4) Fit on the sample (learn topic space)
    _topics_sample, _ = tm.fit_transform(sample_texts)
    print(f"[BERTopic] fit on {len(sample_texts)} docs (device={device})", flush=True)  # <— simple progress print

    # 5) Assign topics to EVERY comment using the learned space (lightweight)
    all_texts = df[text_col].tolist()
    topics_all, _ = tm.transform(all_texts)

    # 6) Store topic ids
    df['topic'] = pd.Series(topics_all, index=df.index).astype(int)
    return df


def compute_embedding_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Dense semantic features (GPU-friendly + memory-safe):
      1) Load a SentenceTransformer model on GPU
      2) Encode texts in SMALL BATCHES to avoid VRAM spikes
      3) Reduce to 34 dimensions with PCA
    """
    model = SentenceTransformer(model_name, device='cuda')  # GPU model (falls back to CPU if no CUDA)
    texts = df[text_col].tolist()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        embs = model.encode(chunk, show_progress_bar=False, batch_size=batch_size)
        all_embs.append(embs)
    embs = np.vstack(all_embs)  # shape ≈ (N_docs, embedding_dim)

    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate comment-level MTLD to per-user mean/std.
    Useful for user-level modeling or stabilizing noisy per-comment signals.
    """
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean', 'std']).rename(columns={
            'mean': 'mtld_mean', 'std': 'mtld_std'
        })
        df = df.merge(um, on=user_col, how='left')
    return df


# ──────────────── Main execution ────────────────

def main():
    """
    Orchestrates the entire Week 2 pipeline:
      1) Resource setup (NLTK + spaCy model)
      2) Load cleaned Week 1 data
      3) Compute features in sequence (vocab → sentiment → style → topics → embeddings)
      4) Aggregate user-level MTLD
      5) Save final feature table for Week 3 modeling
    """
    with timed("Resource setup (NLTK + spaCy)"):
        ensure_nltk_resources()
        ensure_spacy_model("en_core_web_sm")

    # Load data
    root = Path(__file__).parent
    with timed("Load Week 1 parquet"):
        df = pd.read_parquet(root / 'data' / 'pandora_cleaned.parquet')

    # Feature engineering (each step timed)
    with timed("Vocab features"):
        df = compute_vocab_features(df, n_process=4)

    with timed("Sentiment"):
        df = compute_sentiment_features(df, n_jobs=4)

    with timed("Style"):
        df = compute_style_features(df)

    with timed("BERTopic (fit sample + transform all)"):
        df = compute_topic_features(df, sample_size=10000)

    with timed("Embeddings + PCA"):
        df = compute_embedding_features(df, batch_size=32)

    with timed("User MTLD aggregation"):
        df = aggregate_user_mtld(df)

    # Save
    with timed("Write features parquet"):
        out_path = root / 'data' / 'pandora_user_features.parquet'
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved features to {out_path} with {df.shape[1]} columns", flush=True)


if __name__ == '__main__':
    main()
