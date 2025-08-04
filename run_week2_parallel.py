"""
run_week2_parallel.py

Week 2: Feature Engineering (Parallelized)
This version uses parallel CPU cores and GPU offload where available.
"""

import os  # filesystem operations (making directories, handling paths)
from pathlib import Path  # object-oriented filesystem paths
import re  # regex operations for style feature extraction

import pandas as pd  # DataFrame operations and I/O
import numpy as np  # numerical computations

import nltk  # natural language toolkit for downloads and tokenizers
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER sentiment analysis

# Parallel processing libraries
import spacy  # fast NLP pipeline with multi-process support
from joblib import Parallel, delayed  # easy parallel loops for Python

# Lexical richness metric
from lexicalrichness import LexicalRichness  # computes MTLD diversity metric
# Unicode emoji support
import emoji  # extract and count emojis

# Topic modeling and transformer embeddings
from sentence_transformers import SentenceTransformer  # transformer-based embeddings
from bertopic import BERTopic  # topic modeling framework
# UMAP + HDBSCAN for efficient, parallelizable clustering
from umap import UMAP
from hdbscan import HDBSCAN

# Dimensionality reduction for embeddings
from sklearn.decomposition import PCA

# Optional progress bars for pandas operations
from tqdm import tqdm

tqdm.pandas()  # enable df['col'].progress_map() with tqdm


def ensure_nltk_resources():
    """
    Download NLTK resources needed for this script if missing:
      - 'punkt' tokenizer data for splitting text into tokens
      - 'vader_lexicon' for sentiment scoring via VADER
    This runs silently (quiet=True) to avoid cluttering output.
    """
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def compute_vocab_features(df, text_col='text', n_process=4):
    """
    Tokenize and compute lexical diversity features in parallel.

    Returns df with new columns: word_count, unique_word_count, ttr,
    avg_word_length, mtld.
    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts = df[text_col].tolist()
    docs = list(nlp.pipe(texts, n_process=n_process, batch_size=1000))
    df['tokens'] = [[tok.text.lower() for tok in doc] for doc in docs]

    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))
    df['ttr'] = df.apply(
        lambda r: r['unique_word_count'] / r['word_count'] if r['word_count'] > 0 else 0,
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


def compute_sentiment_features(df, text_col='text', n_jobs=4):
    """
    Compute VADER sentiment scores per comment and user-level volatility.
    Adds sent_neg, sent_neu, sent_pos, sent_comp, sent_comp_mean, sent_comp_std.
    """
    analyzer = SentimentIntensityAnalyzer()
    texts = df[text_col].tolist()
    scores = Parallel(n_jobs=n_jobs)(
        delayed(analyzer.polarity_scores)(txt) for txt in texts
    )
    sent_df = pd.DataFrame(scores, index=df.index).rename(columns={
        'neg': 'sent_neg',
        'neu': 'sent_neu',
        'pos': 'sent_pos',
        'compound': 'sent_comp'
    })
    df = pd.concat([df, sent_df], axis=1)

    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean', 'std']).rename(columns={
            'mean': 'sent_comp_mean',
            'std': 'sent_comp_std'
        })
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df, text_col='text'):
    """
    Extract stylistic features: emoji_count, all_caps_count,
    excl_count, quest_count, all_caps_ratio.
    """
    df['emoji_count'] = df[text_col].map(lambda txt: len(emoji.emoji_list(txt)))
    df['all_caps_count'] = df[text_col].map(
        lambda txt: len(re.findall(r"\b[A-Z]{2,}\b", txt))
    )
    df['excl_count'] = df[text_col].map(lambda txt: txt.count('!'))
    df['quest_count'] = df[text_col].map(lambda txt: txt.count('?'))
    df['all_caps_ratio'] = df.apply(
        lambda r: r['all_caps_count'] / r['word_count'] if r['word_count'] > 0 else 0,
        axis=1
    )
    return df


def compute_topic_features(df, text_col='text', sample_size=10000, seed=42):
    """
    Perform BERTopic modeling on a sample and assign topic labels to all.
    """
    sample = df.sample(n=min(sample_size, len(df)), random_state=seed)
    texts = sample[text_col].tolist()

    umap_model = UMAP(n_neighbors=15, n_components=5, n_jobs=4)
    hdbscan_model = HDBSCAN(core_dist_n_jobs=4)
    tm = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)

    topics, _ = tm.fit_transform(texts)
    sample = sample.assign(topic=topics)
    df = df.merge(sample[['topic']], left_index=True, right_index=True, how='left')
    df['topic'] = df['topic'].fillna(-1).astype(int)
    return df


def compute_embedding_features(df, text_col='text', model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Generate transformer embeddings on GPU and reduce to 34 PCA components.
    """
    model = SentenceTransformer(model_name, device='cuda')
    embs = model.encode(df[text_col].tolist(), show_progress_bar=True, batch_size=batch_size)
    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df):
    """
    Aggregate comment-level MTLD to user-level stats: mtld_mean, mtld_std.
    """
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean', 'std']).rename(columns={
            'mean': 'mtld_mean',
            'std': 'mtld_std'
        })
        df = df.merge(um, on=user_col, how='left')
    return df


def main():
    """
    1) Download NLTK data
    2) Load cleaned data
    3) Run all feature functions
    4) Write out a Parquet of features
    """
    ensure_nltk_resources()
    root = Path(__file__).parent
    df = pd.read_parquet(root / 'data' / 'pandora_cleaned.parquet')

    df = compute_vocab_features(df, n_process=4)
    df = compute_sentiment_features(df, n_jobs=4)
    df = compute_style_features(df)
    df = compute_topic_features(df)
    df = compute_embedding_features(df)
    df = aggregate_user_mtld(df)

    out_path = root / 'data' / 'pandora_user_features.parquet'
    os.makedirs(out_path.parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path} with {df.shape[1]} columns")


if __name__ == '__main__':
    main()
