"""
run_week2_parallel.py

Week 2: Feature Engineering (Parallelized)
This version uses parallel CPU cores and GPU offload where available, with memory-optimized settings.
"""

# ──────────────── Standard library imports ────────────────
import os                    # for creating directories, handling file paths
from pathlib import Path     # Path objects for filesystem paths
import re                    # regular expressions for style feature extraction

# ──────────────── Third-party imports ────────────────
import pandas as pd          # DataFrame operations & I/O
import numpy as np           # numerical computations on arrays

import nltk                  # Natural Language Toolkit
from nltk.sentiment.vader import SentimentIntensityAnalyzer  
                             # VADER sentiment analyzer

import spacy                 # spaCy NLP library
from spacy.cli import download as spacy_download  
                             # helper to download spaCy models

from joblib import Parallel, delayed  
                             # easy parallel loops across CPU cores

from lexicalrichness import LexicalRichness  
                             # computes MTLD lexical diversity metric

import emoji                 # extract & count emojis in text

from sentence_transformers import SentenceTransformer  
                             # transformer embeddings
from bertopic import BERTopic  
                             # topic modeling framework

from umap import UMAP        # dimensionality reduction for clustering
from hdbscan import HDBSCAN  # density-based clusterer

from sklearn.decomposition import PCA  
                             # principal component analysis

from tqdm import tqdm        # progress bars
tqdm.pandas()                # integrate tqdm with pandas apply/progress_map


# ──────────────── Utility functions ────────────────

def ensure_nltk_resources():
    """
    Download any missing NLTK data quietly:
      - 'punkt' tokenizer for splitting text into tokens
      - 'vader_lexicon' needed by VADER sentiment
    """
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def ensure_spacy_model(model_name="en_core_web_sm"):
    """
    Check if a spaCy model is installed by attempting to load.
    If not found, download it on the fly.
    """
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found—downloading now...")
        spacy_download(model_name)


# ──────────────── Feature-engineering functions ────────────────

def compute_vocab_features(df, text_col='text', n_process=2):
    """
    Parallel spaCy tokenization & lexical features with reduced processes to save memory.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts = df[text_col].tolist()
    docs = list(nlp.pipe(texts, n_process=n_process, batch_size=500))
    df['tokens'] = [[tok.text.lower() for tok in doc] for doc in docs]

    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))
    df['ttr'] = df.apply(
        lambda row: row['unique_word_count'] / row['word_count'] if row['word_count'] > 0 else 0,
        axis=1
    )
    df['avg_word_length'] = df['tokens'].map(
        lambda toks: np.mean([len(w) for w in toks]) if toks else 0
    )

    def mtld_metric(txt):
        try:
            return LexicalRichness(txt).mtld()
        except ZeroDivisionError:
            return 0.0

    df['mtld'] = Parallel(n_jobs=n_process)(
        delayed(mtld_metric)(txt) for txt in df[text_col]
    )

    df.drop(columns=['tokens'], inplace=True)
    return df


def compute_sentiment_features(df, text_col='text', n_jobs=2):
    """
    Parallel VADER sentiment scoring with reduced jobs.
    """
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].tolist()

    scores = Parallel(n_jobs=n_jobs)(
        delayed(analyzer.polarity_scores)(txt) for txt in texts
    )
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


def compute_style_features(df, text_col='text'):
    """
    Lightweight stylistic features (no parallelism needed).
    """
    df['emoji_count'] = df[text_col].map(lambda txt: len(emoji.emoji_list(txt)))
    df['all_caps_count'] = df[text_col].map(
        lambda txt: len(re.findall(r"\b[A-Z]{2,}\b", txt))
    )
    df['excl_count'] = df[text_col].map(lambda txt: txt.count('!'))
    df['quest_count'] = df[text_col].map(lambda txt: txt.count('?'))
    df['all_caps_ratio'] = df.apply(
        lambda row: row['all_caps_count'] / row['word_count'] if row['word_count'] > 0 else 0,
        axis=1
    )
    return df


def compute_topic_features(df, text_col='text', sample_size=5000, seed=42):
    """
    Topic modeling with reduced sample size and parallel UMAP/HDBSCAN.
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    texts = sample[text_col].tolist()

    umap_model = UMAP(n_neighbors=15, n_components=5, n_jobs=2)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=2)
    tm = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)

    topics, _ = tm.fit_transform(texts)
    sample = sample.assign(topic=topics)
    df = df.merge(sample[['topic']], left_index=True, right_index=True, how='left')
    df['topic'] = df['topic'].fillna(-1).astype(int)
    return df


def compute_embedding_features(df, text_col='text', model_name='all-MiniLM-L6-v2', batch_size=16):
    """
    Chunked GPU embeddings + PCA to limit peak memory usage.
    """
    model = SentenceTransformer(model_name, device='cuda')
    texts = df[text_col].tolist()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        embs = model.encode(chunk, show_progress_bar=False, batch_size=batch_size)
        all_embs.append(embs)
    embs = np.vstack(all_embs)

    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df):
    """
    Aggregate comment-level MTLD to user-level mean & std.
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
    ensure_nltk_resources()
    ensure_spacy_model("en_core_web_sm")

    root = Path(__file__).parent
    df = pd.read_parquet(root / 'data' / 'pandora_cleaned.parquet')

    df = compute_vocab_features(df, n_process=2)
    df = compute_sentiment_features(df, n_jobs=2)
    df = compute_style_features(df)
    df = compute_topic_features(df, sample_size=5000)
    df = compute_embedding_features(df, batch_size=16)
    df = aggregate_user_mtld(df)

    out_path = root / 'data' / 'pandora_user_features.parquet'
    os.makedirs(out_path.parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path} with {df.shape[1]} columns")

if __name__ == '__main__':
    main()
