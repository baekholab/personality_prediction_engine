"""
run_week2_parallel.py

Week 2: Feature Engineering (Parallelized)
This version uses parallel CPU cores and GPU offload where available.
"""

import os
from pathlib import Path
import re

import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import spacy
from spacy.cli import download as spacy_download
from joblib import Parallel, delayed

from lexicalrichness import LexicalRichness
import emoji

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.decomposition import PCA
from tqdm import tqdm

tqdm.pandas()


def ensure_nltk_resources():
    """ Download punkt & vader_lexicon """
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)


def ensure_spacy_model(model_name="en_core_web_sm"):
    """
    Ensure the specified spaCy model is installed. If not, download it.
    """
    try:
        spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found—downloading now...")
        spacy_download(model_name)


def compute_vocab_features(df, text_col='text', n_process=4):
    """ Tokenize with spaCy and compute lexical diversity features. """
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
    model = SentenceTransformer(model_name, device='cuda')
    embs = model.encode(df[text_col].tolist(), show_progress_bar=True, batch_size=batch_size)
    pca = PCA(n_components=34, random_state=42)
    pcs = pca.fit_transform(embs)
    pc_cols = [f'pca_{i}' for i in range(pcs.shape[1])]
    pca_df = pd.DataFrame(pcs, index=df.index, columns=pc_cols)
    return pd.concat([df, pca_df], axis=1)


def aggregate_user_mtld(df):
    user_col = next((c for c in ['user_id', 'author', 'username'] if c in df.columns), None)
    if user_col:
        um = df.groupby(user_col)['mtld'].agg(['mean', 'std']).rename(columns={
            'mean': 'mtld_mean',
            'std': 'mtld_std'
        })
        df = df.merge(um, on=user_col, how='left')
    return df


def main():
    ensure_nltk_resources()
    ensure_spacy_model("en_core_web_sm")

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
