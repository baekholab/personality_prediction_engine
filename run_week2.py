#!/usr/bin/env python
"""
run_week2.py

Week 2: Feature Engineering

Brief:
This script loads the cleaned Pandora Reddit dataset (Week 1 output),
computes a comprehensive set of text-based features (vocabulary richness,
advanced lexical diversity metrics, sentiment scores & volatility,
style cues, topic clusters, reduced embeddings) and writes out a
Parquet file ready for modeling.

Outputs:
- data/pandora_features.parquet
"""

import os
from pathlib import Path
import re

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from lexicalrichness import LexicalRichness
import emoji

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.decomposition import PCA

# Optional: for progress bars on pandas operations
from tqdm import tqdm

tqdm.pandas()  # enable `progress_map` / `progress_apply`


def ensure_nltk_resources():
    """
    Download required NLTK data if it's missing:
    - 'punkt' and 'punkt_tab' for tokenization
    - 'vader_lexicon' for sentiment analysis
    """
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def compute_vocab_features(df, text_col='text'):
    """
    Compute vocabulary-based and advanced lexical diversity features:
      - word_count: total tokens
      - unique_word_count: distinct tokens
      - ttr: type-token ratio
      - avg_word_length: mean token length
      - mtld: Measure of Textual Lexical Diversity
      - yule_i: Yule's I lexical burstiness
    """
    # Tokenize each document to a list of words
    df['tokens'] = df[text_col].progress_map(
        lambda txt: word_tokenize(txt.lower())
    )

    # Basic counts
    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))
    df['ttr'] = df.apply(
        lambda row: row['unique_word_count'] / row['word_count']
        if row['word_count'] > 0 else 0,
        axis=1
    )
    df['avg_word_length'] = df['tokens'].map(
        lambda toks: np.mean([len(w) for w in toks]) if toks else 0
    )

    # Advanced lexical diversity
    def lex_metrics(txt):
        lr = LexicalRichness(txt)
        return lr.mtld(), lr.yule_i()
    lex_df = df[text_col].progress_map(lex_metrics).apply(pd.Series)
    lex_df.columns = ['mtld','yule_i']
    df = pd.concat([df, lex_df], axis=1)

    # Drop helper tokens
    df.drop(columns=['tokens'], inplace=True)
    return df


def compute_sentiment_features(df, text_col='text'):
    """
    Compute VADER sentiment scores per comment,
    then sentiment volatility per user (mean & std. of compound).
    """
    analyzer = SentimentIntensityAnalyzer()
    # per-comment scores
    scores = df[text_col].progress_map(lambda txt: analyzer.polarity_scores(txt))
    sent_df = pd.DataFrame(list(scores), index=df.index)
    sent_df.rename(columns={
        'neg':'sent_neg','neu':'sent_neu',
        'pos':'sent_pos','compound':'sent_comp'
    }, inplace=True)
    df = pd.concat([df, sent_df], axis=1)

    # sentiment volatility (per user)
    user_col = 'user_id' if 'user_id' in df.columns else 'author' if 'author' in df.columns else None
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean','std']).rename(
            columns={'mean':'sent_comp_mean','std':'sent_comp_std'}
        )
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df, text_col='text'):
    """
    Extract style cues:
      - emoji_count
      - all_caps_ratio
      - exclamation_count
      - question_count
    """
    # emoji count
    df['emoji_count'] = df[text_col].progress_map(
        lambda txt: len(emoji.emoji_list(txt))
    )
    # all-caps ratio
    df['all_caps_count'] = df[text_col].progress_map(
        lambda txt: len(re.findall(r"\b[A-Z]{2,}\b", txt))
    )
    df['all_caps_ratio'] = df.apply(
        lambda row: row['all_caps_count'] / row['word_count']
        if row['word_count']>0 else 0, axis=1
    )
    # punctuation counts
    df['excl_count'] = df[text_col].progress_map(lambda txt: txt.count('!'))
    df['quest_count'] = df[text_col].progress_map(lambda txt: txt.count('?'))
    return df


def compute_topic_features(df, text_col='text', sample_size=10_000, seed=42):
    """
    Fit BERTopic and assign topic IDs.
    """
    sample = df.sample(n=min(sample_size,len(df)), random_state=seed)
    texts = sample[text_col].tolist()
    print(f"Fitting BERTopic on {len(texts)} docs...")
    tm = BERTopic(verbose=False)
    topics,_ = tm.fit_transform(texts)
    sample = sample.assign(topic=topics)
    df = df.merge(sample[['topic']], left_index=True, right_index=True, how='left')
    df['topic'] = df['topic'].fillna(-1).astype(int)
    return df


def compute_embedding_features(df, text_col='text', model_name='all-MiniLM-L6-v2'):
    """
    Encode texts and reduce embedding dimensions via PCA.
    """
    model = SentenceTransformer(model_name)
    print("Computing embeddings (batch mode)...")
    embs = model.encode(df[text_col].tolist(), show_progress_bar=True, batch_size=32)
    # full embedding columns
    cols = [f'emb_{i}' for i in range(embs.shape[1])]
    emb_df = pd.DataFrame(embs, index=df.index, columns=cols)

    # PCA reduction
    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)

    # concat and drop full embeddings
    df = pd.concat([df, pca_df], axis=1)
    return df


def main():
    ensure_nltk_resources()
    root = Path(__file__).parent
    in_path = root / 'data' / 'pandora_cleaned.parquet'
    out_path = root / 'data' / 'pandora_features.parquet'

    print(f"Loading data from {in_path}")
    df = pd.read_parquet(in_path)

    print("-> Vocabulary features")
    df = compute_vocab_features(df)

    print("-> Sentiment features & volatility")
    df = compute_sentiment_features(df)

    print("-> Style features")
    df = compute_style_features(df)

    print("-> Topic modeling")
    df = compute_topic_features(df)

    print("-> Embeddings + PCA")
    df = compute_embedding_features(df)

    os.makedirs(out_path.parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path} with {df.shape[1]} columns")


if __name__ == '__main__':
    main()
