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

# ──────────────── Standard library imports (built into Python) ────────────────
# These are generic tools we use: files & folders, timing, random numbers, etc.
import os                                  # create folders, check if files exist
from pathlib import Path                   # handle paths like objects (cleaner than strings)
import re                                  # find patterns in text (e.g., ALL CAPS words)
import time                                # measure how long steps take
from contextlib import contextmanager       # easy way to make timed(...) helper
import contextlib                          # gives us nullcontext() for "do nothing" blocks
import random                               # set a seed so results are reproducible
import json                                 # save a small "manifest" file with run details
import platform                             # record Python version / OS info in the manifest
import argparse                             # read command-line knobs (flags)
import gc                                   # ask Python to free memory sooner when possible

# ──────────────── Third-party imports (installed via pip/conda) ────────────────
import pandas as pd                         # table data (DataFrame) loading/saving
import numpy as np                          # fast math on arrays

import nltk                                 # classic NLP package
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # simple sentiment tool

import spacy                                # modern fast NLP (we’ll use it mostly to tokenize)
from spacy.cli import download as spacy_download             # auto-download spacy model

from joblib import Parallel, delayed        # run simple loops in parallel across CPU cores
import joblib                               # save/load sklearn models like PCA

import torch                                # detect/use GPU and mixed precision

from lexicalrichness import LexicalRichness # computes MTLD (lexical diversity)
import emoji                                # count emojis

from sentence_transformers import SentenceTransformer  # text embeddings (can use GPU)
import sentence_transformers as st          # we import the package itself to read its version
from bertopic import BERTopic               # topic model built on embeddings + clustering
from umap import UMAP                       # dimension reduction before clustering
from hdbscan import HDBSCAN                 # clustering algorithm used by BERTopic

import sklearn                              # machine learning toolkit (we log its version)
from sklearn.decomposition import PCA       # reduce embeddings to ~34 features
from tqdm import tqdm                       # pretty progress bars

# enable progress bars for pandas .progress_map() if we ever call it
tqdm.pandas()


# ──────────────── Make runs reproducible ────────────────
# A fixed "seed" means random choices will be the same each run (useful for debugging).
SEED = 707
random.seed(SEED)
np.random.seed(SEED)
# (If you ever TRAIN a torch model, you’d also set torch.manual_seed here. Not needed for inference.)


# ──────────────── A tiny timing tool so logs are readable ────────────────
@contextmanager
def timed(label: str):
    """
    Use like:
      with timed("Step name"):
          ...do work...
    It prints [START]/[END] with the elapsed seconds so you can see progress.
    """
    t0 = time.time()
    print(f"[START] {label}", flush=True)  # flush=True makes logs appear immediately
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[END]   {label} in {dt:.1f}s", flush=True)


# ──────────────── Command-line knobs (flags) ────────────────
# These let you change settings without editing the code, e.g.:
# python run_week2_parallel.py --sample-size 5000 --batch-size 16
def parse_args():
    p = argparse.ArgumentParser(description="Week 2: parallel feature engineering")
    p.add_argument("--sample-size", type=int, default=10000,
                   help="How many docs to FIT BERTopic on (we still TRANSFORM all docs).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for GPU embeddings (lower=safer on memory).")
    p.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2",
                   help="SentenceTransformer model to use for embeddings.")
    p.add_argument("--n-process", type=int, default=4,
                   help="Number of spaCy worker processes for tokenization.")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="CPU jobs for parallel sentiment + MTLD.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Global seed for reproducibility.")
    p.add_argument("--in-path", type=str, default=None,
                   help="Input parquet path (default: data/pandora_cleaned.parquet).")
    p.add_argument("--out-path", type=str, default=None,
                   help="Output parquet path (default: data/pandora_user_features.parquet).")
    p.add_argument("--log-dir", type=str, default=None,
                   help="Where to write the manifest (default: ./logs).")
    p.add_argument("--artifacts-dir", type=str, default=None,
                   help="Where to save PCA/BERTopic (default: ./models).")
    return p.parse_args()


# ──────────────── Small helpers to make sure dependencies are ready ────────────────
def ensure_nltk_resources():
    # Downloads small data files NLTK needs (quietly, does nothing if already present)
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def ensure_spacy_model(model_name: str = "en_core_web_sm"):
    # Try to load the spaCy English model; if missing, download it automatically.
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found—downloading now...")
        spacy_download(model_name)


# ──────────────── Feature engineering functions ────────────────
# Each one takes a DataFrame and adds columns with new features.


def compute_vocab_features(
    df: pd.DataFrame,
    text_col: str = 'text',
    n_process: int = 2,
    pipe_batch_size: int = 2000,   # How many rows spaCy sees per mini-batch (bigger=faster, smaller=safer)
    mtld_jobs: int | None = None   # How many CPU workers for MTLD (None → use n_process)
) -> pd.DataFrame:
    """
    Vocabulary features that are safe on memory and more accurate:
      • We tokenize with spaCy USING A STREAM (nlp.pipe), so we don't build a giant list in RAM.
      • We keep ONLY "real words" (alphabetic tokens): drops URLs, numbers, emojis, punctuation.
      • We compute: word_count, unique_word_count, TTR, avg_word_length.
      • We compute MTLD on the CLEANED words joined back to a string.
    """
    # Load a light spaCy pipeline (just a tokenizer, no parser/NER → faster and less RAM).
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    # Convert to list of strings. (astype(str) is defensive: if some entries are numbers/None.)
    texts = df[text_col].astype(str).tolist()

    # We'll fill these lists and then attach them as columns (cheap and simple).
    word_counts, uniq_counts, avg_lens, cleaned_stream = [], [], [], []

    # KEY IDEA: nlp.pipe(texts) yields Doc objects one batch at a time (streaming).
    # n_process lets spaCy use multiple CPU processes; batch_size trades speed vs RAM.
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=pipe_batch_size):
        # Keep only alphabetic tokens → "true words"
        toks = [t.text.lower() for t in doc if t.is_alpha]

        # Basic metrics for this row
        wc = len(toks)                                   # total words
        uc = len(set(toks))                              # unique words
        al = float(np.mean([len(w) for w in toks])) if toks else 0.0  # average length

        # Collect
        word_counts.append(wc)
        uniq_counts.append(uc)
        avg_lens.append(al)

        # Join back into a simple space-separated string for MTLD
        cleaned_stream.append(" ".join(toks))

    # Attach the basic metrics to the DataFrame
    df['word_count'] = pd.Series(word_counts, index=df.index)
    df['unique_word_count'] = pd.Series(uniq_counts, index=df.index)
    df['ttr'] = df.apply(
        lambda r: (r['unique_word_count'] / r['word_count']) if r['word_count'] > 0 else 0.0,
        axis=1
    )
    df['avg_word_length'] = pd.Series(avg_lens, index=df.index)

    # Compute MTLD (lexical diversity) on the CLEAN strings.
    # We parallelize this with joblib to use multiple CPU cores.
    def _mtld(txt: str) -> float:
        if not txt:
            return 0.0
        try:
            return LexicalRichness(txt).mtld()
        except ZeroDivisionError:
            # Happens on extremely short texts; we just return 0.0
            return 0.0

    if mtld_jobs is None:
        mtld_jobs = n_process

    df['mtld'] = Parallel(n_jobs=mtld_jobs)(
        delayed(_mtld)(s) for s in cleaned_stream
    )

    # Free the temporary list to release RAM (not strictly necessary but good hygiene).
    del cleaned_stream
    return df


def compute_sentiment_features(df: pd.DataFrame, text_col: str = 'text', n_jobs: int = 2) -> pd.DataFrame:
    """
    Adds per-row sentiment scores (negative/neutral/positive/compound).
    If a user column exists (e.g., 'user_id'), also adds per-user mean/std of compound score.
    """
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].tolist()

    # Run VADER on each text in parallel across CPU cores
    scores = Parallel(n_jobs=n_jobs)(delayed(analyzer.polarity_scores)(txt) for txt in texts)

    # Put the scores next to the original DataFrame
    sent_df = pd.DataFrame(scores, index=df.index).rename(columns={
        'neg': 'sent_neg', 'neu': 'sent_neu', 'pos': 'sent_pos', 'compound': 'sent_comp'
    })
    df = pd.concat([df, sent_df], axis=1)

    # If we have a user column, compute user-level mean/std (volatility)
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean', 'std']).rename(columns={
            'mean': 'sent_comp_mean', 'std': 'sent_comp_std'
        })
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Adds style features that capture tone/formatting rather than content:
      • emoji_count: how many emojis used
      • all_caps_count: "SHOUTING" tokens like THIS
      • excl_count / quest_count: how many '!' and '?'
      • all_caps_ratio: all_caps_count divided by word_count (normalizes by length)
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


# ──────────────── Embeddings (encode once and reuse) ────────────────
# We compute SentenceTransformer embeddings ONCE on GPU, and reuse them for:
#   1) BERTopic (topic IDs) and 2) PCA (dense numeric features).
def encode_embeddings(
    df: pd.DataFrame,
    text_col: str = 'text',
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 32,
    mixed_precision: bool = True
) -> np.ndarray:
    # If we have a GPU, use it; otherwise this runs on CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    texts = df[text_col].tolist()

    embs_list = []
    # Mixed precision on GPU (float16) can be faster and use less VRAM.
    # On CPU or if disabled, we use a no-op context (nullcontext).
    autocast_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                    if (device == 'cuda' and mixed_precision) else contextlib.nullcontext())
    with torch.inference_mode():  # tells PyTorch "we are not training"
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
            # Convert to float32 for stability in downstream PCA
            embs_list.append(arr.astype(np.float32, copy=False))

    # Stack all mini-batches into a single (N_docs, D) array
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
    BERTopic normally re-embeds texts; we skip that by giving it embeddings directly.
    We FIT (learn topics) on a SAMPLE to save memory, then TRANSFORM all docs to get topic IDs.
    Returns the updated DataFrame and the fitted BERTopic model (so we can save it).
    """
    n = len(df)
    sample_size = min(sample_size, n)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False)

    sample_texts = df.iloc[sample_idx][text_col].tolist()
    sample_embs = embeddings[sample_idx]

    # UMAP reduces dimension; HDBSCAN clusters. We keep CPU-friendly settings.
    umap_model = UMAP(n_neighbors=15, n_components=5, n_jobs=2, random_state=seed)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=2)

    # embedding_model=None tells BERTopic "do NOT embed for me; I'll pass embeddings".
    tm = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        low_memory=True,
        calculate_probabilities=False,
        verbose=False,
    )

    # Learn topics on the sample
    _topics_sample, _ = tm.fit_transform(sample_texts, embeddings=sample_embs)
    print(f"[BERTopic] fit on {len(sample_texts)} docs (precomputed embs)", flush=True)

    # Assign a topic to EVERY document using the full embeddings
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
    PCA turns the high-dimensional embedding vector into ~34 compact numeric features
    (pca_0, pca_1, ..., pca_33). These are handy for Week 3 modeling.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(embeddings)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1), pca


def aggregate_user_mtld(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some users write multiple comments. We add per-user MTLD mean/std to
    stabilize variability at the comment level. If no user column exists, we skip it.
    """
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean', 'std']).rename(columns={
            'mean': 'mtld_mean', 'std': 'mtld_std'
        })
        df = df.merge(um, on=user_col, how='left')
    return df


# ──────────────── Main: orchestrates the whole pipeline ────────────────
def main():
    # Read knobs from the command line
    args = parse_args()

    # Make random steps reproducible based on the chosen seed
    global SEED
    SEED = int(args.seed)
    random.seed(SEED)
    np.random.seed(SEED)

    # Make sure NLP resources exist; print which device we’ll use
    with timed("Resource setup (NLTK + spaCy)"):
        ensure_nltk_resources()
        ensure_spacy_model("en_core_web_sm")
        print(f"[GPU] Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    # Build and create directories we’ll use
    root = Path(__file__).parent
    in_path = Path(args.in_path) if args.in_path else (root / 'data' / 'pandora_cleaned.parquet')
    out_path = Path(args.out_path) if args.out_path else (root / 'data' / 'pandora_user_features.parquet')
    log_dir = Path(args.log_dir) if args.log_dir else (root / 'logs')
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (root / 'models')
    data_dir = root / 'data'
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned data from Week 1
    with timed("Load Week 1 parquet"):
        if not in_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {in_path}")
        df = pd.read_parquet(in_path)

    # Make sure the text column is string, drop empty rows to avoid edge cases
    text_col = 'text'
    df[text_col] = df[text_col].astype(str).fillna('')
    df = df[df[text_col].str.strip().ne('')].copy()
    if df.empty:
        # If everything is empty, there's nothing to process — fail fast.
        raise ValueError("No non-empty texts found after cleaning; aborting Week 2.")

    # 1) Vocabulary (tokenize → counts → MTLD). This is the most CPU-heavy step.
    with timed("Vocab features"):
        df = compute_vocab_features(df, n_process=args.n_process)

    # 2) Sentiment (VADER) — also CPU, but cheap
    with timed("Sentiment"):
        df = compute_sentiment_features(df, n_jobs=args.n_jobs)

    # 3) Style (emoji, ALL CAPS, !, ?) — cheap string counts
    with timed("Style"):
        df = compute_style_features(df)

    # 4) Embeddings ONCE (GPU if available). We reuse these for topics + PCA.
    with timed("Embeddings (encode once on GPU)"):
        embs = encode_embeddings(
            df, text_col=text_col, batch_size=args.batch_size, mixed_precision=True,
            model_name=args.model_name
        )

    # Save embeddings to disk so Week 3 can reuse them without recomputing
    emb_path = data_dir / 'embeddings_fp32.npy'
    np.save(emb_path, embs)
    print(f"[Artifacts] Saved embeddings to {emb_path}", flush=True)

    # 5) Topics (fit on a sample for efficiency, transform all with precomputed embs)
    with timed("BERTopic (fit sample + transform all) [precomputed embs]"):
        df, tm = assign_topics_with_precomputed_embeddings(
            df, embs, text_col=text_col, sample_size=args.sample_size, seed=SEED
        )

    # 6) PCA (turn each embedding into ~34 numeric features)
    with timed("PCA from precomputed embeddings"):
        df, pca = add_pca_from_embeddings(df, embs, n_components=34, seed=SEED)

    # We don’t need the big embeddings array in RAM anymore — drop it to free memory.
    del embs
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # also nudge the GPU to free VRAM
    except Exception:
        pass  # if this fails, it’s fine — best effort

    # 7) User-level MTLD aggregation (optional, if user id exists)
    with timed("User MTLD aggregation"):
        df = aggregate_user_mtld(df)

    # 8) Save the final feature table for Week 3 modeling
    with timed("Write features parquet"):
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved features to {out_path} with {df.shape[1]} columns", flush=True)

    # 9) Save the PCA and the BERTopic model so Week 3 can reload them quickly
    pca_path = artifacts_dir / 'week2_pca.pkl'
    joblib.dump(pca, pca_path)
    print(f"[Artifacts] Saved PCA to {pca_path}", flush=True)

    bertopic_dir = artifacts_dir / 'bertopic'
    tm.save(str(bertopic_dir))  # BERTopic knows how to save itself into a folder
    print(f"[Artifacts] Saved BERTopic model to {bertopic_dir}", flush=True)

    # Optional: save a CSV summary of topics to eyeball in a spreadsheet
    try:
        topic_info_path = bertopic_dir / "topic_info.csv"
        tm.get_topic_info().to_csv(topic_info_path, index=False)
        print(f"[Artifacts] Saved topic info to {topic_info_path}", flush=True)
    except Exception as e:
        print(f"[Warn] Could not write topic_info.csv: {e}", flush=True)

    # 10) Write a small JSON "manifest" with parameters and version info.
    #     This makes runs traceable and reproducible.
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


# Standard Python entry point: only run main() if we executed this file directly
if __name__ == '__main__':
    main()
