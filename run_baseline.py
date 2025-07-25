# run_baseline.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# 1) load features
df = pd.read_parquet("data/pandora_features.parquet")
X = df.drop(columns=[c for c in df.columns if c.startswith("label_")])
y = df["label_extraversion"]  # repeat per trait or loop
# 2) train RF baseline
rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print("RF baseline R²:", scores.mean())
# 3) transformer‐only baseline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# ... load embeddings columns, PCA down, etc.
# print comparison
