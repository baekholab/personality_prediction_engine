#!/usr/bin/env python
"""
run_week1.py

Week 1: Data Acquisition & Cleaning
  1. Download the four Parquet shards from Hugging Face
  2. Merge into one DataFrame
  3. Clean text (remove nulls, [deleted]/[removed], short <10 chars)
     AND drop rows missing any of the Big-Five labels: O, C, E, A, N
  4. Save as data/pandora_cleaned.parquet
"""

import urllib.request
from pathlib import Path
import pandas as pd

def download_shards():
    """
    Download the four parquet shards from the Hugging Face Pandora Big-5 dataset.
    Only downloads if a file doesn’t already exist locally.
    Returns paths to the downloaded shard files.
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)  # ensure data/ exists

    base = (
        "https://huggingface.co/datasets/jingjietan/pandora-big5/"
        "resolve/main/data/"
    )
    shards = [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
    ]

    for fname in shards:
        dest = data_dir / fname
        if not dest.exists():
            print(f"⬇️  Downloading {fname}")
            urllib.request.urlretrieve(base + fname, dest)

    print("✅ All shards downloaded")
    return [data_dir / f for f in shards]

def merge_shards(shard_paths):
    """
    Read each parquet shard, concatenate into one DataFrame,
    and save as data/pandora_raw.parquet.
    """
    print("🔗 Merging shards…")
    dfs = [pd.read_parquet(p) for p in shard_paths]
    full = pd.concat(dfs, ignore_index=True)

    raw_path = Path("data") / "pandora_raw.parquet"
    full.to_parquet(raw_path, index=False)
    print(f"✅ Merged raw data → {raw_path} ({len(full):,} rows)")
    return raw_path

def clean_data(raw_path):
    """
    Load the merged raw data, filter out:
      - comments shorter than 10 characters
      - “[deleted]” or “[removed]”
      - rows missing any Big-Five labels (O, C, E, A, N)
      - rows missing created_utc
    Save the cleaned DataFrame as data/pandora_cleaned.parquet.
    """
    print("🧹 Cleaning data…")
    df = pd.read_parquet(raw_path)

    # Keep only comments longer than 10 chars, not deleted/removed
    def valid(text):
        return (
            isinstance(text, str)
            and len(text) > 10
            and text.lower() not in ["[deleted]", "[removed]"]
        )

    # Apply mask and drop rows missing timestamp or any Big-Five label
    df = (
        df[df["text"].apply(valid)]
          .dropna(subset=["created_utc", "O", "C", "E", "A", "N"])
          .reset_index(drop=True)
    )

    clean_path = Path("data") / "pandora_cleaned.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"✅ Cleaned data → {clean_path} ({len(df):,} rows)")
    return clean_path

def main():
    """
    Execute Week 1 pipeline end-to-end:
      1. download_shards
      2. merge_shards
      3. clean_data
    """
    shards = download_shards()
    raw = merge_shards(shards)
    clean = clean_data(raw)

if __name__ == "__main__":
    main()
