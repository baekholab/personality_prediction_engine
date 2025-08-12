"""
run_week2_parallel.py

Week 2: Feature Engineering (Parallelized)
Memory-optimized to avoid OOM kills by chunking, reducing parallelism, and sampling.

Upgrades in this version:
  • Computes vocabulary metrics on TRUE words only (no URLs, emojis, punctuation)
  • Computes MTLD on the cleaned word stream (more faithful lexical diversity)
  • Encodes embeddings ONCE on GPU and reuses** them for BERTopic + PCA
  • Fits BERTopic on a **SAMPLE** (UMAP/HDBSCAN), then **TRANSFORMS the FULL CORPUS**
  • Keeps GPU embedding batches small and predictable to avoid memory spikes
  • Prints lightweight step timing so you can follow progress in logs
  • CLI flags for tunable runs + run manifest JSON for traceability
  • Persist artifacts (embeddings .npy, PCA .pkl, BERTopic model)
"""

# ──────────────── Standard library imports ────────────────
import os                                  # for creating directories and checking file paths
from pathlib import Path                   # object-oriented filesystem path handling
import re                                  # regular expressions, used in style feature extraction
import time                                # for simple timing
from contextlib import contextmanager       # to build a tiny timing context manager
import contextlib                          # for nullcontext
import random                               # global seeding
import json                                 # (NEW) write a manifest
import platform                             # (NEW) version/device info for manifest
import argparse                             # (NEW) command-line flags
import gc                                   # (NEW) manual garbage collection after big arrays are freed

# ──────────────── Third-party imports ────────────────
import pandas as pd                         # DataFrame operations and fast I/O
import numpy as np                          # numerical computations and array handling

import nltk                                 # Natural Language Toolkit for NLP utilities
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER sentiment analyzer

import spacy                                # spaCy for fast, industrial-strength NLP
from spacy.cli import download as spacy_download             # helper to install spaCy models on the fly

from joblib import Parallel, delayed        # easy parallel loops across multiple CPU cores
import joblib                               # (NEW) persist sklearn PCA model

import torch                                # CUDA detection / mixed precision

from lexicalrichness import LexicalRichness # computes MTLD lexical diversity metric
import emoji                                # utilities to extract and count emojis in text

from sentence_transformers import SentenceTransformer  # transformer-based embeddings (GPU-capable)
import sentence_transformers as st          # (NEW) correct package version for manifest
from bertopic import BERTopic               # topic modeling framework built on transformers + clustering
from umap import UMAP                       # dimensionality reduction for clustering
from hdbscan import HDBSCAN                 # density-based clustering algorithm

import sklearn                              # (NEW) include version in manifest
from sklearn.decomposition import PCA       # principal component analysis for dimensionality reduction
from tqdm import tqdm                       # progress bars for pandas operations

tqdm.pandas()  # enable df['col'].progress_map() with a progress bar


# ──────────────── Global seeds for reproducibility ────────────────
SEED = 707
random.seed(SEED)
np.random.seed(SEED)
# (For inference we don't need torch.manual_seed, but add if you ever train.)


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


# ──────────────── CLI flags ────────────────
def parse_args():
    """
    Command-line knobs so you can tune runs without editing code.
    Examples:
      python run_week2_parallel.py --sample-size 12000 --batch-size 48 --n-process 4 --n-jobs 4 --seed 42
    """
    p = argparse.ArgumentParser(description="Week 2: parallel feature engineering")
    p.add_argument("--sample-size", type=int, default=10000, help="Docs to fit BERTopic on (transform runs on full corpus).")
    p.add_argument("--batch-size", type=int, default=32, help="SBERT embedding batch size (GPU).")
    p.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2",  # (NEW) choose the SBERT model via CLI
                   help="SentenceTransformer model to use for embeddings.")
    p.add_argument("--n-process", type=int, default=4, help="spaCy tokenization workers for vocab features.")
    p.add_argument("--n-jobs", type=int, default=4, help="Joblib parallelism for VADER + MTLD.")
    p.add_argument("--seed", type=int, default=SEED, help="Global seed for reproducibility.")
    p.add_argument("--in-path", type=str, default=None, help="Override input parquet path (defaults to data/pandora_cleaned.parquet).")
    p.add_argument("--out-path", type=str, default=None, help="Override output parquet path (defaults to data/pandora_user_features.parquet).")
    p.add_argument("--log-dir", type=str, default=None, help="Directory to write manifest and logs (defaults to ./logs).")
    p.add_argument("--artifacts-dir", type=str, default=None, help="Directory to save embeddings/PCA/BERTopic (defaults to ./models & ./data).")
    return p.parse_args()


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
    # Optional: add sentence splitter if you later want sentence-level features
    # if "sentencizer" not in nlp.pipe_names:
    #     nlp.add_pipe("sentencizer")

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


# ──────────────── Encode-once & reuse helpers ────────────────

def encode_embeddings(
    df: pd.DataFrame,
    text_col: str = 'text',
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    mixed_precision: bool = True
) -> np.ndarray:
    """
    Encode all texts exactly once on GPU (if available).
    Returns a single (N_docs, D) float32 NumPy array.
    Optionally uses mixed precision (fp16) for speed / lower VRAM.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    texts = df[text_col].tolist()

    embs_list = []
    # Prepare autocast/no-op contexts
    autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if (device == 'cuda' and mixed_precision) else contextlib.nullcontext())
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            with autocast_ctx:
                arr = model.encode(
                    chunk,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
            # Ensure float32 for downstream PCA stability
            embs_list.append(arr.astype(np.float32, copy=False))
    embs = np.vstack(embs_list)
    return embs


def assign_topics_with_precomputed_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_col: str = 'text',
    sample_size: int = 10000,
    seed: int = SEED
) -> tuple[pd.DataFrame, BERTopic]:
    """
    Fit BERTopic on a sampled subset using the provided embeddings, then
    transform the full corpus (also with provided embeddings). Returns:
      - a DataFrame with 'topic' column
      - the fitted BERTopic model (so we can persist it for Week 3)
    This avoids re-encoding texts inside BERTopic and keeps results reproducible.
    """
    n = len(df)
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    sample_texts = df.iloc[sample_idx][text_col].tolist()
    sample_embs = embeddings[sample_idx]

    # CPU clustering config; add random_state to UMAP for determinism
    umap_model = UMAP(n_neighbors=15, n_components=5, n_jobs=2, random_state=seed)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=2)

    # Important: Pass embedding_model=None since we supply embeddings explicitly
    tm = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        low_memory=True,
        calculate_probabilities=False,
        verbose=False,
    )

    # Fit on the sampled texts+embeddings
    _topics_sample, _ = tm.fit_transform(sample_texts, embeddings=sample_embs)
    print(f"[BERTopic] fit on {len(sample_texts)} docs (precomputed embs)", flush=True)

    # Transform the full corpus with precomputed embeddings
    all_texts = df[text_col].tolist()
    topics_all, _ = tm.transform(all_texts, embeddings=embeddings)

    df = df.copy()
    df['topic'] = pd.Series(topics_all, index=df.index).astype(int)
    return df, tm


def add_pca_from_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_components: int = 34,
    seed: int = SEED
) -> tuple[pd.DataFrame, PCA]:
    """
    Run PCA on the already-computed embeddings and append pca_0..pca_{k-1}.
    Return the fitted PCA object so we can persist it for Week 3.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(embeddings)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1), pca


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
      3) Compute features in sequence (vocab → sentiment → style → embeddings once → BERTopic → PCA)
      4) Aggregate user-level MTLD
      5) Save final feature table + artifacts + manifest for Week 3 modeling/reuse
    """
    args = parse_args()  # (NEW) read CLI flags

    # Apply global seed (and allow CLI override)
    global SEED
    SEED = int(args.seed)
    random.seed(SEED)
    np.random.seed(SEED)

    with timed("Resource setup (NLTK + spaCy)"):
        ensure_nltk_resources()
        ensure_spacy_model("en_core_web_sm")
        print(f"[GPU] Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    # Resolve paths
    root = Path(__file__).parent
    in_path = Path(args.in_path) if args.in_path else (root / 'data' / 'pandora_cleaned.parquet')
    out_path = Path(args.out_path) if args.out_path else (root / 'data' / 'pandora_user_features.parquet')
    log_dir = Path(args.log_dir) if args.log_dir else (root / 'logs')
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (root / 'models')
    data_dir = root / 'data'
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with timed("Load Week 1 parquet"):
        if not in_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {in_path}")
        df = pd.read_parquet(in_path)

    # Hygiene: ensure text is string and drop empty rows to avoid edge cases
    text_col = 'text'
    df[text_col] = df[text_col].astype(str).fillna('')
    df = df[df[text_col].str.strip().ne('')].copy()
    #  Guard against an all-empty dataset after cleaning
    if df.empty:
        raise ValueError("No non-empty texts found after cleaning; aborting Week 2.")

    # Feature engineering (each step timed)
    with timed("Vocab features"):
        df = compute_vocab_features(df, n_process=args.n_process)

    with timed("Sentiment"):
        df = compute_sentiment_features(df, n_jobs=args.n_jobs)

    with timed("Style"):
        df = compute_style_features(df)

    # Encode once on GPU and reuse
    with timed("Embeddings (encode once on GPU)"):
        embs = encode_embeddings(
            df, text_col=text_col, batch_size=args.batch_size, mixed_precision=True,
            model_name=args.model_name  # (NEW) model is now a CLI knob
        )

    # Persist embeddings for Week 3 reuse/auditing
    emb_path = data_dir / 'embeddings_fp32.npy'
    np.save(emb_path, embs)  # saves as float32
    print(f"[Artifacts] Saved embeddings to {emb_path}", flush=True)

    with timed("BERTopic (fit sample + transform all) [precomputed embs]"):
        df, tm = assign_topics_with_precomputed_embeddings(
            df, embs, text_col=text_col, sample_size=args.sample_size, seed=SEED
        )

    with timed("PCA from precomputed embeddings"):
        df, pca = add_pca_from_embeddings(df, embs, n_components=34, seed=SEED)

    # We no longer need the in-memory embedding matrix after PCA & topics
    del embs
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # hint to free VRAM sooner
    except Exception:
        pass  # best-effort only

    with timed("User MTLD aggregation"):
        df = aggregate_user_mtld(df)

    # Save features parquet
    with timed("Write features parquet"):
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved features to {out_path} with {df.shape[1]} columns", flush=True)

    # Persist PCA + BERTopic for Week 3 (fast reload, reproducibility)
    pca_path = artifacts_dir / 'week2_pca.pkl'
    joblib.dump(pca, pca_path)
    print(f"[Artifacts] Saved PCA to {pca_path}", flush=True)

    bertopic_dir = artifacts_dir / 'bertopic'
    tm.save(str(bertopic_dir))  # BERTopic handles directories
    print(f"[Artifacts] Saved BERTopic model to {bertopic_dir}", flush=True)

    # Also save a human-readable topic summary for sanity checks
    try:
        topic_info_path = bertopic_dir / "topic_info.csv"
        tm.get_topic_info().to_csv(topic_info_path, index=False)
        print(f"[Artifacts] Saved topic info to {topic_info_path}", flush=True)
    except Exception as e:
        print(f"[Warn] Could not write topic_info.csv: {e}", flush=True)

    # Write a manifest capturing params, versions, device, and shapes
    manifest = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "embeddings_path": str(emb_path),
        "pca_path": str(pca_path),
        "bertopic_dir": str(bertopic_dir),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "params": {
            "sample_size": args.sample_size,
            "batch_size": args.batch_size,
            "model_name": args.model_name,    
            "n_process": args.n_process,
            "n_jobs": args.n_jobs,
            "seed": SEED
        },
        "env": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "spacy": spacy.__version__,
            "nltk": nltk.__version__,
            "sentence_transformers": st.__version__,  
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        }
    }
    manifest_path = log_dir / 'week2_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Manifest] wrote {manifest_path}", flush=True)


if __name__ == '__main__':
    main()
