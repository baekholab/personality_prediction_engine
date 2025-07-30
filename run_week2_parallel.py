```python
#!/usr/bin/env python
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

    Parameters:
      df (pd.DataFrame): Input DataFrame containing a text column.
      text_col (str): Name of the column with raw text.
      n_process (int): Number of CPU processes for spaCy tokenization.

    Returns:
      pd.DataFrame: Original df with additional columns:
        - word_count: number of tokens per comment
        - unique_word_count: distinct tokens per comment
        - ttr: type-token ratio (unique/total)
        - avg_word_length: mean length of tokens
        - mtld: Measure of Textual Lexical Diversity
    """
    # Load spaCy model optimized for tokenization only
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts = df[text_col].tolist()  # extract all comment texts
    # Parallel processing: divide texts across n_process workers
    docs = list(nlp.pipe(texts, n_process=n_process, batch_size=1000))
    # Extract lowercase token lists from spaCy Doc objects
    df['tokens'] = [[tok.text.lower() for tok in doc] for doc in docs]

    # Basic lexical features
    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))
    # Type-Token Ratio for lexical diversity
    df['ttr'] = df.apply(
        lambda r: r['unique_word_count'] / r['word_count'] if r['word_count'] > 0 else 0,
        axis=1
    )
    # Average token length in each comment
    df['avg_word_length'] = df['tokens'].map(
        lambda toks: np.mean([len(w) for w in toks]) if toks else 0
    )

    # MTLD calculation can be expensive; parallelize with Joblib
    def mtld_metric(txt):
        try:
            return LexicalRichness(txt).mtld()
        except ZeroDivisionError:
            return 0.0
    df['mtld'] = Parallel(n_jobs=n_process)(
        delayed(mtld_metric)(txt) for txt in df[text_col]
    )

    # Drop intermediate token column to save memory
    df.drop(columns=['tokens'], inplace=True)
    return df


def compute_sentiment_features(df, text_col='text', n_jobs=4):
    """
    Compute VADER sentiment scores per comment and user-level volatility.

    Parameters:
      df (pd.DataFrame): Input DataFrame containing a text column.
      text_col (str): Name of the column with raw text.
      n_jobs (int): Number of parallel jobs for sentiment scoring.

    Returns:
      pd.DataFrame: Original df with added columns:
        - sent_neg, sent_neu, sent_pos, sent_comp: VADER scores per comment
        - sent_comp_mean, sent_comp_std: mean and std of compound score per user
    """
    analyzer = SentimentIntensityAnalyzer()  # initialize VADER
    texts = df[text_col].tolist()
    # Parallelize sentiment analysis across n_jobs CPU cores
    scores = Parallel(n_jobs=n_jobs)(
        delayed(analyzer.polarity_scores)(txt) for txt in texts
    )
    sent_df = pd.DataFrame(scores, index=df.index)
    sent_df.rename(columns={
        'neg': 'sent_neg',
        'neu': 'sent_neu',
        'pos': 'sent_pos',
        'compound': 'sent_comp'
    }, inplace=True)
    df = pd.concat([df, sent_df], axis=1)

    # Identify user ID column and compute volatility of sentiment
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean', 'std'])
        agg.rename(columns={'mean': 'sent_comp_mean', 'std': 'sent_comp_std'}, inplace=True)
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df, text_col='text'):
    """
    Extract stylistic features such as emoji usage, all-caps words, and punctuation.

    Parameters:
      df (pd.DataFrame): Input DataFrame.
      text_col (str): Column name containing comment text.

    Returns:
      pd.DataFrame: df with added columns:
        - emoji_count: count of emojis in text
        - all_caps_count: number of words in ALL CAPS
        - excl_count: count of exclamation marks '!'
        - quest_count: count of question marks '?'
        - all_caps_ratio: ratio of all_caps_count to total words
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
    Perform topic modeling using BERTopic on a sample and apply to all data.

    Steps:
      1. Randomly sample up to sample_size documents for faster fitting.
      2. Configure UMAP and HDBSCAN with n_jobs for parallel embeddings/clustering.
      3. Fit BERTopic on sample and assign cluster labels.
      4. Merge sample topics back into full DataFrame, filling missing with -1.

    Parameters:
      df (pd.DataFrame): Input DataFrame.
      text_col (str): Column name with comment text.
      sample_size (int): Max number of docs to sample for training.
      seed (int): Random seed for reproducibility.

    Returns:
      pd.DataFrame: df with a new 'topic' integer column per comment.
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
    Generate dense semantic embeddings and reduce dimensions.

    Steps:
      1. Load SentenceTransformer model onto GPU (
         device='cuda') for fast encoding.
      2. Encode all texts in batches to utilize GPU cores.
      3. Apply PCA on CPU to reduce embedding size to 34 components.
      4. Append PCA scores as new 'pca_*' columns.

    Parameters:
      df (pd.DataFrame): Input DataFrame.
      text_col (str): Column with comment text.
      model_name (str): Pretrained model identifier.
      batch_size (int): Number of docs per GPU batch.

    Returns:
      pd.DataFrame: df with appended PCA-reduced embedding features.
    """
    model = SentenceTransformer(model_name, device='cuda')
    embs = model.encode(
        df[text_col].tolist(), show_progress_bar=True, batch_size=batch_size
    )
    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df):
    """
    Aggregate comment-level MTLD to user-level statistics.

    Computes mean and standard deviation of MTLD per user to capture
    variability in lexical diversity across comments.

    Parameters:
      df (pd.DataFrame): DataFrame with 'mtld' and user ID columns.

    Returns:
      pd.DataFrame: df with new columns 'mtld_mean' and 'mtld_std'.
    """
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean', 'std'])
        um.rename(columns={'mean': 'mtld_mean', 'std': 'mtld_std'}, inplace=True)
        df = df.merge(um, on=user_col, how='left')
    return df


def main():
    """
    Main execution flow:
      1. Ensure necessary NLTK data is available.
      2. Load cleaned Week 1 data from Parquet.
      3. Sequentially apply each compute_* function, which handle
         their own parallelism and device placement.
      4. Save the fully-featured DataFrame to Parquet for modeling.
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
```
