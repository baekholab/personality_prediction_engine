#!/usr/bin/env python
"""
run_week2.py

Week 2: Feature Engineering

This script loads cleaned Reddit comments (Week 1 output) and
builds numeric features that capture linguistic, emotional,
stylistic, topical, and semantic characteristics of each comment.
We drop Yule's I and aggregate MTLD at the user level here.
The features are saved to a single Parquet file for modeling.

Features:
  - Vocabulary: counts & diversity metrics (MTLD only)
  - Sentiment: VADER scores and user-level volatility
  - Style: emojis, ALL-CAPS, punctuation counts
  - Topic: BERTopic cluster assignments
  - Semantic: MiniLM embeddings + PCA reduction
  - User-Level MTLD: mean & std of comment-level MTLD

Output:
  data/pandora_user_features.parquet
"""

import os                              # filesystem operations
from pathlib import Path              # path utilities
import re                             # regex for style cues

import pandas as pd                   # DataFrame & I/O
import numpy as np                    # numeric operations

import nltk                           # NLP toolkit
from nltk.tokenize import word_tokenize            # word-level tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # VADER sentiment

# Advanced lexical diversity metrics
from lexicalrichness import LexicalRichness
# Emoji processing
import emoji

# Sentence embeddings and topic modeling
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
# PCA for reducing embedding dimensions
from sklearn.decomposition import PCA

# Progress bars for pandas
from tqdm import tqdm

tqdm.pandas()  # enables df['col'].progress_map()


def ensure_nltk_resources():
    """
    Download required NLTK resources if missing:
      - 'punkt', 'punkt_tab' for tokenization
      - 'vader_lexicon' for sentiment
    """
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def compute_vocab_features(df, text_col='text'):
    """
    Compute basic and advanced vocabulary features:
      1. Tokenize text into lowercase words
      2. Count words and unique types
      3. Compute type-token ratio (TTR)
      4. Compute average token length
      5. Compute MTLD safely (Yule's I dropped)

    Adds: 'word_count','unique_word_count','ttr','avg_word_length','mtld'
    """
    # 1) Tokenize each comment
    df['tokens'] = df[text_col].progress_map(lambda txt: word_tokenize(txt.lower()))

    # 2) Basic counts
    df['word_count'] = df['tokens'].map(len)
    df['unique_word_count'] = df['tokens'].map(lambda toks: len(set(toks)))

    # 3) Type-Token Ratio
    df['ttr'] = df.apply(
        lambda row: row['unique_word_count']/row['word_count'] if row['word_count']>0 else 0,
        axis=1
    )

    # 4) Average word length
    df['avg_word_length'] = df['tokens'].map(
        lambda toks: np.mean([len(w) for w in toks]) if toks else 0
    )

    # 5) MTLD only (Yule's I removed)
    def lex_mtld(txt):
        """Return MTLD or 0.0 if text too short"""
        try:
            return LexicalRichness(txt).mtld()
        except ZeroDivisionError:
            return 0.0

    df['mtld'] = df[text_col].progress_map(lex_mtld)

    # Clean up helper tokens
    df.drop(columns=['tokens'], inplace=True)
    return df


def compute_sentiment_features(df, text_col='text'):
    """
    Compute VADER sentiment per comment and user-level volatility:
      Adds 'sent_neg','sent_neu','sent_pos','sent_comp',
      and merges 'sent_comp_mean','sent_comp_std' by user
    """
    analyzer = SentimentIntensityAnalyzer()
    # Per-comment sentiment
    scores = df[text_col].progress_map(lambda txt: analyzer.polarity_scores(txt))
    sent_df = pd.DataFrame(list(scores), index=df.index)
    sent_df.rename(columns={'neg':'sent_neg','neu':'sent_neu',
                             'pos':'sent_pos','compound':'sent_comp'}, inplace=True)
    df = pd.concat([df, sent_df], axis=1)

    # Determine user column
    user_col = next((c for c in ['user_id','author','username'] if c in df.columns), None)
    if user_col:
        agg = df.groupby(user_col)['sent_comp'].agg(['mean','std'])
        agg.rename(columns={'mean':'sent_comp_mean','std':'sent_comp_std'}, inplace=True)
        df = df.merge(agg, on=user_col, how='left')
    return df


def compute_style_features(df, text_col='text'):
    """Extract style cues: emojis, ALL-CAPS, punctuation counts"""
    df['emoji_count'] = df[text_col].progress_map(lambda txt: len(emoji.emoji_list(txt)))
    df['all_caps_count'] = df[text_col].progress_map(
        lambda txt: len(re.findall(r"\b[A-Z]{2,}\b", txt))
    )
    df['all_caps_ratio'] = df.apply(
        lambda row: row['all_caps_count']/row['word_count'] if row['word_count']>0 else 0,
        axis=1
    )
    df['excl_count'] = df[text_col].progress_map(lambda txt: txt.count('!'))
    df['quest_count'] = df[text_col].progress_map(lambda txt: txt.count('?'))
    return df


def compute_topic_features(df, text_col='text', sample_size=10000, seed=42):
    """Fit BERTopic on a sample and label all comments"""
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
    """Generate MiniLM embeddings and reduce via PCA"""
    model = SentenceTransformer(model_name)
    print("Computing embeddings (batch mode)...")
    embs = model.encode(df[text_col].tolist(), show_progress_bar=True, batch_size=32)
    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df):
    """Aggregate per-user MTLD statistics"""
    user_col = next((c for c in ['user_id','author','username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean','std'])
        um.rename(columns={'mean':'mtld_mean','std':'mtld_std'}, inplace=True)
        df = df.merge(um, on=user_col, how='left')
    return df


def main():
    ensure_nltk_resources()
    root = Path(__file__).parent
    in_path = root / 'data' / 'pandora_cleaned.parquet'
    out_path = root / 'data' / 'pandora_user_features.parquet'

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

    print("-> Aggregating user-level MTLD")
    df = aggregate_user_mtld(df)

    os.makedirs(out_path.parent, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved features to {out_path} with {df.shape[1]} columns")

if __name__ == '__main__':
    main()
