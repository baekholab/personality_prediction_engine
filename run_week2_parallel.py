"""
Week 2 — Extended Feature Engineering (Annotated)

This version keeps the SAME five core buckets as the core script:
  1) Vocabulary richness
  2) Sentiment swings
  3) Style markers
  4) Topics (BERTopic: fit on a sample, transform all)
  5) Dense meaning (embeddings -> PCA)

…plus TWO practical extensions for scale & reproducibility:
  A) Embedding persistence options (save none / fp16 / fp32 / sharded fp16)
  B) Run manifest: record paths, params, and library versions

Why this matters:
- Reusing embeddings speeds iteration (e.g., retune UMAP/HDBSCAN without re-encoding).
- Sharding fp16 keeps files small and avoids single very-large .npy files.
- Manifest makes experiments traceable across machines and teammates.
"""

# ===================== Standard & third-party imports =====================
# (Everything here is used below; comments say what each library contributes.)
import os, re, json, random, time, contextlib, gc
from pathlib import Path
from contextlib import contextmanager
import platform  # for recording OS/Python in the manifest

import numpy as np              # fast numeric arrays
import pandas as pd             # tabular data (DataFrame)

import nltk                     # classic NLP utilities
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # simple, fast sentiment

import spacy                    # modern tokenizer; we stream with nlp.pipe
from spacy.cli import download as spacy_download

from joblib import Parallel, delayed  # CPU parallelism for VADER/MTLD
import joblib                          # save/load sklearn objects (PCA)

import torch                    # detect GPU; mixed-precision for embedding
from sentence_transformers import SentenceTransformer  # embeddings on CPU/GPU
import sentence_transformers as st  # to record package version in manifest

from lexicalrichness import LexicalRichness  # MTLD (lexical diversity)
import emoji                                  # emoji counting

from bertopic import BERTopic                 # topic modeling over embeddings
from umap import UMAP                         # dimensionality reduction before clustering
from hdbscan import HDBSCAN                   # clustering algorithm used by BERTopic

import sklearn
from sklearn.decomposition import PCA         # turn embeddings into ~34 numeric features
import argparse                               # user-friendly CLI flags

# =========================== Global configuration ==========================
SEED = 707                                # fixed seed for reproducibility
random.seed(SEED); np.random.seed(SEED)   # seed Python & NumPy

# Precompile regex once (faster than compiling inside loops)
ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")

# -------------------------- Tiny timing helper ----------------------------
@contextmanager
def timed(label: str):
    """Print START/END with elapsed seconds around a code block.
    Usage:
        with timed("Vocab features"):
            ... do work ...
    """
    t0 = time.time()
    print(f"[START] {label}", flush=True)
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[END]   {label} in {dt:.1f}s", flush=True)

# --------------------- Ensure small NLP resources exist --------------------
def ensure_nltk_spacy():
    """Download tiny NLTK data and the spaCy English model if missing."""
    nltk.download("punkt", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("[Setup] Downloading spaCy model en_core_web_sm...")
        spacy_download("en_core_web_sm")

# =========================== Feature bucket 1 ==============================
# Vocabulary richness: word_count, unique_word_count, TTR, avg_word_length, MTLD

def compute_vocab_features(
    df: pd.DataFrame,
    text_col: str = "text",
    n_process: int = 4,
    pipe_batch_size: int = 2000,
    mtld_jobs: int | None = None
) -> pd.DataFrame:
    """Tokenize text in a RAM-safe stream and compute lexical metrics.

    Tips:
    - We disable heavy spaCy components to keep it fast and light.
    - We keep only alphabetic tokens (true words) to avoid URLs/numbers.
    - We compute MTLD over the cleaned word stream.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", "tagger"])

    texts = df[text_col].astype(str).tolist()
    word_counts, uniq_counts, avg_lens, cleaned_stream = [], [], [], []

    # Stream with nlp.pipe to avoid building giant lists in RAM.
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=pipe_batch_size):
        toks = [t.text.lower() for t in doc if t.is_alpha]
        wc = len(toks)
        uc = len(set(toks))
        al = float(np.mean([len(w) for w in toks])) if toks else 0.0
        word_counts.append(wc); uniq_counts.append(uc); avg_lens.append(al)
        cleaned_stream.append(" ".join(toks))  # MTLD expects a word stream

    df["word_count"] = pd.Series(word_counts, index=df.index)
    df["unique_word_count"] = pd.Series(uniq_counts, index=df.index)
    df["ttr"] = np.where(df["word_count"] > 0, df["unique_word_count"] / df["word_count"], 0.0)
    df["avg_word_length"] = pd.Series(avg_lens, index=df.index)

    def _mtld(txt: str) -> float:
        if not txt:
            return 0.0
        try:
            return LexicalRichness(txt).mtld()
        except ZeroDivisionError:
            return 0.0

    if mtld_jobs is None:
        mtld_jobs = n_process
    df["mtld"] = Parallel(n_jobs=mtld_jobs)(delayed(_mtld)(s) for s in cleaned_stream)
    return df

# =========================== Feature bucket 2 ==============================
# Sentiment swings: VADER per row; optional per-user mean/std (volatility)

def compute_sentiment_features(df: pd.DataFrame, text_col: str = "text", n_jobs: int = 4) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].tolist()

    scores = Parallel(n_jobs=n_jobs)(delayed(analyzer.polarity_scores)(t) for t in texts)
    sent_df = pd.DataFrame(scores, index=df.index).rename(columns={
        "neg": "sent_neg", "neu": "sent_neu", "pos": "sent_pos", "compound": "sent_comp"
    })
    df = pd.concat([df, sent_df], axis=1)

    user_col = next((c for c in ["user_id", "author", "username"] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)["sent_comp"].agg(["mean", "std"]).rename(columns={
            "mean": "sent_comp_mean", "std": "sent_comp_std"
        })
        df = df.merge(agg, on=user_col, how="left")
    return df

# =========================== Feature bucket 3 ==============================
# Style markers: emojis, ALL CAPS, exclamation/question marks, normalized

def compute_style_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df["emoji_count"]    = df[text_col].map(lambda s: len(emoji.emoji_list(s)))
    df["all_caps_count"] = df[text_col].map(lambda s: len(ALL_CAPS_RE.findall(s)))
    df["excl_count"]     = df[text_col].map(lambda s: s.count("!"))
    df["quest_count"]    = df[text_col].map(lambda s: s.count("?"))
    df["all_caps_ratio"] = np.where(df["word_count"] > 0, df["all_caps_count"] / df["word_count"], 0.0)
    return df

# =========================== Feature bucket 4 ==============================
# Dense meaning: encode SentenceTransformer embeddings ONCE (GPU if available)

def encode_embeddings(
    df: pd.DataFrame,
    text_col: str = "text",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    mixed_precision: bool = True
) -> np.ndarray:
    """Encode each text into a dense vector.
    - Batches the work for stable memory/VRAM
    - Mixed precision on GPU saves VRAM; falls back to no-op on CPU
    - Returns a single (N, D) float32 matrix for PCA/BERTopic
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    texts = df[text_col].tolist()
    embs_list = []

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if (device == "cuda" and mixed_precision) else contextlib.nullcontext())

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
            embs_list.append(arr.astype(np.float32, copy=False))

    return np.vstack(embs_list)

# =========================== Feature bucket 5a =============================
# Topics: BERTopic with precomputed embeddings (fit on sample, transform all)

def assign_topics_with_precomputed_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    text_col: str = "text",
    sample_size: int = 10000,
    seed: int = SEED
) -> tuple[pd.DataFrame, BERTopic]:
    n = len(df)
    sample_size = min(sample_size, n)

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    sample_texts = df.iloc[sample_idx][text_col].tolist()
    sample_embs  = embeddings[sample_idx]

    # UMAP reduces dimensionality; HDBSCAN clusters points into topics.
    umap_model    = UMAP(n_neighbors=15, n_components=5, n_jobs=2, random_state=seed)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=2)

    # embedding_model=None → "do not re-embed"; use our precomputed matrix
    tm = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        low_memory=True,
        calculate_probabilities=False,
        verbose=False,
    )

    # Learn topics on the sample, then transform ALL docs to assign IDs
    tm.fit(sample_texts, embeddings=sample_embs)
    topics_all, _ = tm.transform(df[text_col].tolist(), embeddings=embeddings)

    df = df.copy()
    df["topic"] = pd.Series(topics_all, index=df.index).astype(int)
    return df, tm

# =========================== Feature bucket 5b =============================
# Dense meaning (compact): PCA over embeddings → pca_0..pca_33

def add_pca_from_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_components: int = 34,
    seed: int = SEED
) -> tuple[pd.DataFrame, PCA]:
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(embeddings)
    pc_cols = [f"pca_{i}" for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1), pca

# ================================ CLI =====================================

def parse_args():
    """Command-line knobs (additions highlighted):
    --save-embeddings: how to write embeddings to disk, if at all
        none    → do not save (lowest disk use, but you must re-encode later)
        fp16    → single .npy in half precision (2 bytes/float)
        fp32    → single .npy in full precision (4 bytes/float)
        sharded → multiple fp16 shards, each smaller & easier to move
    --shard-rows: number of rows per shard when using sharded mode
    """
    p = argparse.ArgumentParser(description="Week 2: extended feature engineering (annotated)")
    p.add_argument("--in-path", type=str, default=None, help="Input parquet path (Week 1 output)")
    p.add_argument("--out-path", type=str, default=None, help="Where to write Week 2 features")
    p.add_argument("--log-dir", type=str, default=None, help="Where to write the manifest JSON")
    p.add_argument("--artifacts-dir", type=str, default=None, help="Where to save PCA/BERTopic")

    p.add_argument("--sample-size", type=int, default=10000, help="Docs to FIT BERTopic on")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings encode")
    p.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    p.add_argument("--n-process", type=int, default=4, help="spaCy tokenizer worker processes")
    p.add_argument("--n-jobs", type=int, default=4, help="CPU jobs for sentiment + MTLD")
    p.add_argument("--seed", type=int, default=SEED, help="Global random seed")

    # (NEW) Embedding persistence knobs to avoid disk OOMs / speed iteration
    p.add_argument("--save-embeddings", choices=["none", "fp16", "fp32", "sharded"], default="none",
                   help="How to persist embeddings to disk.")
    p.add_argument("--shard-rows", type=int, default=250_000,
                   help="Rows per shard when --save-embeddings=sharded.")
    return p.parse_args()

# =============================== Orchestration =============================

def main():
    args = parse_args()

    # Keep randomness reproducible for this run
    global SEED
    SEED = int(args.seed)
    random.seed(SEED); np.random.seed(SEED)

    # Ensure small resources exist; print which device we'll use
    with timed("Resource setup (NLTK + spaCy)"):
        ensure_nltk_spacy()
        print(f"[GPU] Using: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    # Build common paths (relative to this file)
    root = Path(__file__).parent
    in_path  = Path(args.in_path)  if args.in_path  else (root / "data" / "pandora_cleaned.parquet")
    out_path = Path(args.out_path) if args.out_path else (root / "data" / "pandora_user_features.parquet")
    log_dir  = Path(args.log_dir)  if args.log_dir  else (root / "logs")
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else (root / "models")
    data_dir = root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load Week 1 output and do defensive cleanup
    with timed("Load Week 1 parquet"):
        if not in_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {in_path}")
        df = pd.read_parquet(in_path)
        text_col = "text"
        df[text_col] = df[text_col].astype(str).fillna("")
        df = df[df[text_col].str.strip().ne("")].copy()
        if df.empty:
            raise ValueError("No non-empty texts found after cleaning; aborting Week 2.")

    # 1) Vocabulary
    with timed("Vocab features"):
        df = compute_vocab_features(df, text_col=text_col, n_process=args.n_process)

    # 2) Sentiment
    with timed("Sentiment"):
        df = compute_sentiment_features(df, text_col=text_col, n_jobs=args.n_jobs)

    # 3) Style
    with timed("Style"):
        df = compute_style_features(df, text_col=text_col)

    # 4) Embeddings (encode once on GPU if available)
    with timed("Embeddings (encode once)"):
        embs = encode_embeddings(df, text_col=text_col, model_name=args.model_name, batch_size=args.batch_size, mixed_precision=True)

    # -------- Extension A: Embedding persistence (optional) --------
    # Decide if/how to save embeddings to disk. This is optional; skip if disk is tight.
    emb_artifacts = None  # will be str path, list of shard paths, or None

    if args.save_embeddings == "fp16":
        emb_path = data_dir / "embeddings_fp16.npy"
        np.save(emb_path, embs.astype(np.float16))
        emb_artifacts = str(emb_path)
        print(f"[Artifacts] Saved fp16 embeddings to {emb_path}", flush=True)

    elif args.save_embeddings == "fp32":
        emb_path = data_dir / "embeddings_fp32.npy"
        np.save(emb_path, embs)
        emb_artifacts = str(emb_path)
        print(f"[Artifacts] Saved fp32 embeddings to {emb_path}", flush=True)

    elif args.save_embeddings == "sharded":
        shards = []
        n = embs.shape[0]
        for start in range(0, n, args.shard_rows):
            end = min(start + args.shard_rows, n)
            shard_path = data_dir / f"embeddings_fp16_part_{start:09d}.npy"
            np.save(shard_path, embs[start:end].astype(np.float16))
            shards.append(str(shard_path))
        emb_artifacts = shards
        print(f"[Artifacts] Saved {len(shards)} fp16 shards to {data_dir}", flush=True)

    else:  # "none"
        print("[Artifacts] Skipping embedding save (--save-embeddings=none)", flush=True)

    # 5a) Topics
    with timed("BERTopic (fit sample + transform all)"):
        df, tm = assign_topics_with_precomputed_embeddings(df, embs, text_col=text_col, sample_size=args.sample_size, seed=SEED)

    # 5b) PCA
    with timed("PCA from embeddings"):
        df, pca = add_pca_from_embeddings(df, embs, n_components=34, seed=SEED)

    # Free big arrays from RAM/VRAM
    del embs
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # Save the final feature table
    with timed("Write features parquet"):
        os.makedirs(out_path.parent, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"[SAVE] Features -> {out_path}")

    # Save PCA reducer and BERTopic model for fast reload in Week 3
    pca_path = artifacts_dir / "week2_pca.pkl"
    joblib.dump(pca, pca_path)
    bertopic_dir = artifacts_dir / "bertopic"
    tm.save(str(bertopic_dir))
    print(f"[Artifacts] Saved PCA to {pca_path}")
    print(f"[Artifacts] Saved BERTopic model to {bertopic_dir}")

    # Optional: export topic info CSV for quick inspection in a spreadsheet
    try:
        topic_info_path = bertopic_dir / "topic_info.csv"
        tm.get_topic_info().to_csv(topic_info_path, index=False)
        print(f"[Artifacts] Saved topic info to {topic_info_path}")
    except Exception as e:
        print(f"[Warn] Could not write topic_info.csv: {e}")

    # -------- Extension B: Manifest with parameters & versions --------
    manifest = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "embeddings_path": emb_artifacts,  # can be str, list[str], or None
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
            "seed": SEED,
            "save_embeddings": args.save_embeddings,
            "shard_rows": args.shard_rows,
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
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
        },
    }
    manifest_path = log_dir / "week2_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[Manifest] wrote {manifest_path}")


if __name__ == "__main__":
    main()
