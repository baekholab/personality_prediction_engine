#!/usr/bin/env python
"""
run_pipeline.py

Unified Week 1–2 pipeline:
  1. Load raw CSV of Big-Five-labeled Reddit data
  2. Clean text, save pandora_cleaned.parquet
  3. Load cleaned data, extract features (vocab, sentiment, time, topics, embeddings)
  4. Save pandora_features.parquet
"""

import re
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import nltk
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

def setup():
    """Download any required models/data once."""
    nltk.download('vader_lexicon', quiet=True)
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")

def clean_data(raw_csv: Path, cleaned_parquet: Path):
    """Load raw CSV, clean text, save parquet."""
    print(f"1️⃣  Loading raw CSV from {raw_csv}")
    df = pd.read_csv(raw_csv)
    print("   Initial shape:", df.shape)

    def is_valid(text):
        return (
            isinstance(text, str)
            and len(text) >= 20
            and text.lower() not in ['[deleted]', '[removed]']
        )

    df = df[df['text'].apply(is_valid)].dropna(subset=['created_utc'])
    df = df.reset_index(drop=True)
    print("   After cleaning:", df.shape)

    df.to_parquet(cleaned_parquet, index=False)
    print(f"✅ Cleaned data saved to {cleaned_parquet}")

def feature_engineer(cleaned_parquet: Path, features_parquet: Path):
    """Load cleaned data, extract features, save parquet."""
    print(f"2️⃣  Loading cleaned data from {cleaned_parquet}")
    df = pd.read_parquet(cleaned_parquet)
    print("   Rows:", len(df))

    tqdm.pandas()

    # Vocabulary richness
    def vocab_feats(txt):
        words = re.findall(r'\w+', txt.lower())
        total = len(words)
        unique = len(set(words))
        ttr = unique / total if total else 0
        avg_len = np.mean([len(w) for w in words]) if words else 0
        return pd.Series([total, unique, ttr, avg_len])

    df[['word_count','unique_words','ttr','avg_word_len']] = (
        df['text'].progress_apply(vocab_feats)
    )

    # Sentiment (VADER)
    analyzer = SentimentIntensityAnalyzer()
    def vader_feats(txt):
        s = analyzer.polarity_scores(txt)
        return pd.Series([s['neg'], s['neu'], s['pos'], s['compound']])

    df[['vader_neg','vader_neu','vader_pos','vader_compound']] = (
        df['text'].progress_apply(vader_feats)
    )

    # Posting clock
    def time_feats(utc):
        dt = datetime.utcfromtimestamp(utc)
        return pd.Series([dt.hour, dt.weekday()])

    df[['post_hour','post_weekday']] = (
        df['created_utc'].progress_apply(time_feats)
    )

    # Topics & style (optional sampling)
    docs = df['text'].sample(n=10000, random_state=42).tolist()
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(docs)
    topic_df = pd.DataFrame({'text': docs, 'topic': topics})
    df = df.merge(topic_df, on='text', how='left')
    df['topic'] = df['topic'].fillna(-1).astype(int)

    # Dense meaning (MiniLM embeddings)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    df['embedding'] = df['text'].progress_apply(lambda t: embedder.encode(t))

    # Expand embeddings
    embs = pd.DataFrame(df['embedding'].tolist(), index=df.index)
    embs.columns = [f"emb_{i}" for i in range(embs.shape[1])]
    df = pd.concat([df, embs], axis=1).drop(columns=['embedding'])

    # Save
    df.to_parquet(features_parquet, index=False)
    print(f"✅ Features saved to {features_parquet} ({df.shape[1]} columns)")

def main():
    base = Path(__file__).parent
    raw_csv = base / "data" / "reddit_bigfive_sample.csv"
    clean_pq = base / "data" / "pandora_cleaned.parquet"
    feat_pq  = base / "data" / "pandora_features.parquet"

    setup()
    clean_data(raw_csv, clean_pq)
    feature_engineer(clean_pq, feat_pq)

if __name__ == "__main__":
    main()
