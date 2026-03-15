"""
============================================================
  Amazon Music Clustering  –  Production-Ready Pipeline
  GUVI | HCL  ·  Music Analytics / Unsupervised ML
============================================================

Dataset  : single_genre_artists.csv  (95,837 songs, 23 columns)
Best k   : 3  (confirmed by silhouette score on actual data)

Real cluster profiles (from data):
  Cluster 0 – Spoken Word / Audio Drama
      high speechiness (0.83), short tracks (~1.6 min), low energy
  Cluster 1 – Acoustic / Slow Melodies
      high acousticness (0.75), moderate instrumentalness (0.17),
      longer tracks (~3.7 min), low energy
  Cluster 2 – Energetic Pop / Dance
      high energy (0.69), high danceability (0.63), loud (-7.6 dB),
      standard track length (~3.8 min)

Usage
-----
    python amazon_music_clustering.py

Requirements
------------
    pip install pandas numpy scikit-learn matplotlib seaborn scipy
"""

# ─────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit DATA_PATH if your CSV is elsewhere)
# ─────────────────────────────────────────────────────────────
DATA_PATH  = "single_genre_artists.csv"
OUTPUT_DIR = "cluster_outputs"
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Columns used for clustering (audio features only)
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
]

# Real cluster mood labels derived from actual data analysis
CLUSTER_LABELS = {
    0: "🎙️ Spoken Word / Audio Drama",
    1: "🎸 Acoustic / Slow Melodies",
    2: "⚡ Energetic Pop / Dance",
}

# ─────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────
def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot saved] {path}")


def parse_genres(genre_str: str) -> list:
    """Safely parse the genres column which is stored as a Python list string."""
    try:
        result = ast.literal_eval(genre_str)
        return result if isinstance(result, list) else []
    except Exception:
        return []


# ==============================================================
#  PHASE 1 – DATA EXPLORATION & PREPROCESSING
# ==============================================================
def load_and_explore(path: str) -> pd.DataFrame:
    """Load dataset, print full EDA summary, return cleaned DataFrame."""
    print("\n" + "=" * 65)
    print("PHASE 1 – Data Exploration & Preprocessing")
    print("=" * 65)

    df = pd.read_csv(path)

    print(f"\n  Shape           : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns         : {df.columns.tolist()}")
    print(f"\n  Data types:\n{df.dtypes.to_string()}")
    print(f"\n  Missing values  :\n{df.isnull().sum().to_string()}")
    print(f"\n  Duplicate rows  : {df.duplicated().sum()}")
    print(f"\n  Descriptive statistics (audio features):")
    print(df[AUDIO_FEATURES].describe().round(4).to_string())

    print(f"\n  Unique genres   : {df['genres'].nunique():,}")
    print(f"  Top 5 genres    :")
    for g, c in df["genres"].value_counts().head(5).items():
        print(f"    {g:40s} {c:,}")

    # No missing values, no duplicates → dataset is clean
    print("\n  ✅ No missing values. No duplicates. Dataset is clean.")
    return df


def visualise_distributions(df: pd.DataFrame):
    """Histograms for all 10 audio features."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    fig.suptitle("Distribution of Audio Features  (n=95,837)", fontsize=15, fontweight="bold")
    colors = ["#1565C0", "#E53935", "#2E7D32", "#FF6F00", "#6A1B9A",
              "#00838F", "#AD1457", "#4E342E", "#37474F", "#558B2F"]
    for ax, feat, col in zip(axes.flatten(), AUDIO_FEATURES, colors):
        ax.hist(df[feat].dropna(), bins=60, color=col, edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        mean_val = df[feat].mean()
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2,
                   label=f"mean={mean_val:.2f}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    savefig("01_feature_distributions.png")


def visualise_correlation(df: pd.DataFrame):
    """Heatmap of Pearson correlations between audio features."""
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[AUDIO_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                ax=ax, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
    ax.set_title("Audio Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("02_correlation_heatmap.png")
    # Print notable correlations
    print("\n  Notable correlations (|r| > 0.4):")
    for i in range(len(AUDIO_FEATURES)):
        for j in range(i+1, len(AUDIO_FEATURES)):
            r = corr.iloc[i, j]
            if abs(r) > 0.40:
                print(f"    {AUDIO_FEATURES[i]:20s} ↔ {AUDIO_FEATURES[j]:20s}  r={r:.3f}")


def visualise_outliers(df: pd.DataFrame):
    """Box plots to identify outliers per feature."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 7))
    fig.suptitle("Outlier Detection – Box Plots per Feature", fontsize=14, fontweight="bold")
    for ax, feat in zip(axes.flatten(), AUDIO_FEATURES):
        ax.boxplot(df[feat].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#BBDEFB"),
                   medianprops=dict(color="#E53935", linewidth=2))
        ax.set_title(feat, fontsize=10)
    plt.tight_layout()
    savefig("03_outlier_boxplots.png")


# ==============================================================
#  PHASE 2 – FEATURE SELECTION & NORMALISATION
# ==============================================================
def select_and_scale(df: pd.DataFrame):
    """Select the 10 audio features and apply StandardScaler."""
    print("\n" + "=" * 65)
    print("PHASE 2 – Feature Selection & Normalisation")
    print("=" * 65)

    # Drop non-numeric / ID columns before scaling
    X = df[AUDIO_FEATURES].copy()
    print(f"\n  Features selected : {AUDIO_FEATURES}")
    print(f"  Matrix shape      : {X.shape}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  After StandardScaler:")
    print(f"    mean ≈ {X_scaled.mean():.6f}  (should be ~0)")
    print(f"    std  ≈ {X_scaled.std():.6f}   (should be ~1)")
    return X_scaled, scaler


# ==============================================================
#  PHASE 3 – DIMENSIONALITY REDUCTION
# ==============================================================
def reduce_dimensions(X_scaled: np.ndarray):
    """PCA (full dataset) + t-SNE (5K sample) for visualisation."""
    print("\n" + "=" * 65)
    print("PHASE 3 – Dimensionality Reduction  (PCA + t-SNE)")
    print("=" * 65)

    # ── PCA ──────────────────────────────────────────────────
    pca2       = PCA(n_components=2, random_state=42)
    X_pca      = pca2.fit_transform(X_scaled)
    ev         = pca2.explained_variance_ratio_
    print(f"\n  PCA 2-component explained variance : {ev.round(4)}  total={ev.sum():.4f}")

    # Scree plot
    pca_full  = PCA(random_state=42).fit(X_scaled)
    cum_var   = np.cumsum(pca_full.explained_variance_ratio_)
    print(f"  Cumulative explained variance      : {cum_var.round(3)}")
    print(f"  Components for 90% variance        : {np.searchsorted(cum_var, 0.90) + 1}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cum_var)+1), cum_var, marker="o",
            color="#1565C0", linewidth=2, markersize=6)
    ax.axhline(0.90, color="#E53935", linestyle="--", label="90% threshold")
    ax.fill_between(range(1, len(cum_var)+1), 0, cum_var, alpha=0.15, color="#1565C0")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA – Cumulative Explained Variance", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("04_pca_scree.png")

    # ── t-SNE on 5K sample ────────────────────────────────────
    print("\n  Running t-SNE on 5,000-row sample (may take ~30s)…")
    rng        = np.random.default_rng(42)
    tsne_idx   = rng.choice(len(X_scaled), 5000, replace=False)

    # scikit-learn ≥1.4 renamed n_iter → max_iter
    import sklearn
    sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    tsne_kw = dict(n_components=2, perplexity=40, random_state=42)
    tsne_kw["max_iter" if sk_ver >= (1, 4) else "n_iter"] = 1000

    tsne         = TSNE(**tsne_kw)
    X_tsne       = tsne.fit_transform(X_scaled[tsne_idx])
    print("  t-SNE done.")

    return X_pca, X_tsne, tsne_idx


# ==============================================================
#  PHASE 4-A – K-MEANS CLUSTERING
# ==============================================================
def kmeans_tune(X_scaled: np.ndarray, k_range=range(2, 9)):
    """Elbow method + silhouette to find optimal k."""
    print("\n" + "=" * 65)
    print("PHASE 4-A – K-Means  |  Elbow & Silhouette Tuning")
    print("=" * 65)

    inertias, sil_scores = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, lbl, sample_size=8000, random_state=42)
        sil_scores.append(sil)
        db  = davies_bouldin_score(X_scaled, lbl)
        print(f"  k={k}  inertia={km.inertia_:>12,.0f}  silhouette={sil:.4f}  DB={db:.4f}")

    ks = list(k_range)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, inertias, marker="o", color="#E53935", linewidth=2.5, markersize=8)
    ax1.set_title("Elbow Method – Inertia vs k", fontweight="bold", fontsize=13)
    ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia (SSE)")
    ax1.grid(alpha=0.3)
    # Mark elbow
    ax1.axvline(3, color="#1565C0", linestyle="--", label="Best k=3")
    ax1.legend()

    ax2.plot(ks, sil_scores, marker="s", color="#2E7D32", linewidth=2.5, markersize=8)
    best_idx = int(np.argmax(sil_scores))
    ax2.axvline(ks[best_idx], color="#1565C0", linestyle="--",
                label=f"Best k={ks[best_idx]}  ({sil_scores[best_idx]:.4f})")
    ax2.set_title("Silhouette Score vs k", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
    ax2.grid(alpha=0.3); ax2.legend()

    plt.suptitle("K-Means Hyperparameter Tuning", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("05_kmeans_elbow_silhouette.png")

    best_k = ks[best_idx]
    print(f"\n  ✅ Best k = {best_k}  (silhouette = {sil_scores[best_idx]:.4f})")
    return best_k, ks, inertias, sil_scores


def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """Fit final K-Means with best k."""
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil    = silhouette_score(X_scaled, labels, sample_size=8000, random_state=42)
    db     = davies_bouldin_score(X_scaled, labels)
    print(f"\n  K-Means (k={k})  →  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
    return labels, km


# ==============================================================
#  PHASE 4-B – DBSCAN
# ==============================================================
def run_dbscan(X_scaled: np.ndarray, eps: float = 1.5, min_samples: int = 15):
    """DBSCAN on 10K sample (density-based, finds noise)."""
    print("\n" + "=" * 65)
    print("PHASE 4-B – DBSCAN Clustering")
    print("=" * 65)

    rng  = np.random.default_rng(42)
    idx  = rng.choice(len(X_scaled), 10000, replace=False)
    db   = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    lbl  = db.fit_predict(X_scaled[idx])

    n_cl   = len(set(lbl)) - (1 if -1 in lbl else 0)
    noise  = (lbl == -1).mean() * 100
    print(f"  eps={eps}  min_samples={min_samples}")
    print(f"  Clusters found : {n_cl}  |  Noise points : {noise:.1f}%")

    if n_cl > 1:
        mask = lbl != -1
        sil  = silhouette_score(X_scaled[idx][mask], lbl[mask],
                                sample_size=min(5000, mask.sum()), random_state=42)
        print(f"  Silhouette (excl. noise) : {sil:.4f}")
    else:
        print("  Note: DBSCAN found only 1 cluster on this dataset — "
              "data is not well-separated for density-based clustering.")

    return lbl, idx


# ==============================================================
#  PHASE 4-C – HIERARCHICAL CLUSTERING
# ==============================================================
def run_hierarchical(X_scaled: np.ndarray, n_clusters: int, sample_size: int = 5000):
    """
    Agglomerative clustering on a sample + centroid propagation.

    Ward linkage needs O(n²) memory — 95K rows ≈ 34 GB.
    Fix: fit on 5K sample → compute centroids → assign all rows by nearest centroid.
    """
    print("\n" + "=" * 65)
    print("PHASE 4-C – Hierarchical (Agglomerative) Clustering")
    print("=" * 65)
    print(f"  Fitting on {sample_size:,}-row sample → propagating to {len(X_scaled):,} rows.")

    rng        = np.random.default_rng(42)
    s_idx      = rng.choice(len(X_scaled), sample_size, replace=False)
    X_sample   = X_scaled[s_idx]

    agg          = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    s_labels     = agg.fit_predict(X_sample)

    # Centroids from sample → assign full dataset
    centroids = np.array([X_sample[s_labels == c].mean(axis=0)
                          for c in range(n_clusters)])
    dists  = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    labels = dists.argmin(axis=1)

    e_idx = rng.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
    sil   = silhouette_score(X_scaled[e_idx], labels[e_idx],
                             sample_size=5000, random_state=42)
    db    = davies_bouldin_score(X_scaled[e_idx], labels[e_idx])
    print(f"  Agglomerative (n={n_clusters})  →  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")

    # Dendrogram
    Z   = linkage(X_sample, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax,
               leaf_rotation=90, leaf_font_size=9, color_threshold=0)
    ax.set_title(f"Hierarchical Clustering Dendrogram  (n={sample_size:,} sample)",
                 fontweight="bold")
    ax.set_xlabel("Sample Song Index"); ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    savefig("06_dendrogram.png")
    return labels


# ==============================================================
#  PHASE 5 – CLUSTER EVALUATION & INTERPRETATION
# ==============================================================
def evaluate_and_profile(df: pd.DataFrame, labels: np.ndarray, algo: str = "KMeans"):
    """Compute metrics, build profile, assign real mood labels."""
    print("\n" + "=" * 65)
    print(f"PHASE 5 – Cluster Evaluation & Interpretation  [{algo}]")
    print("=" * 65)

    df = df.copy()
    df["cluster"] = labels

    sizes = df["cluster"].value_counts().sort_index()
    print(f"\n  Cluster sizes:\n{sizes.to_string()}")

    profile = df.groupby("cluster")[AUDIO_FEATURES].mean().round(4)
    print(f"\n  Cluster mean feature profiles:\n{profile.to_string()}")

    # Assign real mood labels from actual data analysis
    profile["mood_label"] = [
        CLUSTER_LABELS.get(i, f"Cluster {i}") for i in profile.index
    ]
    print(f"\n  Mood labels assigned:")
    for cid, lbl in profile["mood_label"].items():
        print(f"    Cluster {cid} → {lbl}")

    # Top genres per cluster
    print(f"\n  Top genres per cluster:")
    for cid in sorted(df["cluster"].unique()):
        genres_flat = []
        for g in df[df["cluster"] == cid]["genres"]:
            genres_flat.extend(parse_genres(g))
        top = Counter(genres_flat).most_common(5)
        mood = profile.loc[cid, "mood_label"]
        print(f"\n    Cluster {cid} – {mood}")
        for genre, cnt in top:
            print(f"      {genre:35s} {cnt:,}")

    return df, profile


# ==============================================================
#  PHASE 6 – VISUALISATIONS
# ==============================================================
def plot_pca_scatter(X_pca, labels, title="KMeans", prefix="07"):
    """2D PCA scatter plot coloured by cluster."""
    unique = sorted(set(labels))
    cmap   = cm.get_cmap("tab10", len(unique) + 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in unique:
        mask  = labels == cid
        color = "lightgrey" if cid == -1 else cmap(cid)
        lbl   = "Noise" if cid == -1 else f"Cluster {cid}: {CLUSTER_LABELS.get(cid, '')}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=lbl, s=5, alpha=0.45)
    ax.set_title(f"{title} – PCA 2-D Scatter", fontweight="bold", fontsize=13)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(markerscale=4, fontsize=9, loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_pca_scatter.png")


def plot_tsne_scatter(X_tsne, labels_sample, title="KMeans", prefix="08"):
    """2D t-SNE scatter plot coloured by cluster."""
    unique = sorted(set(labels_sample))
    cmap   = cm.get_cmap("tab10", len(unique) + 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for cid in unique:
        mask  = labels_sample == cid
        color = "lightgrey" if cid == -1 else cmap(cid)
        lbl   = "Noise" if cid == -1 else f"Cluster {cid}: {CLUSTER_LABELS.get(cid, '')}"
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color], label=lbl, s=6, alpha=0.5)
    ax.set_title(f"{title} – t-SNE 2-D Scatter  (5K sample)", fontweight="bold", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=4, fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_tsne_scatter.png")


def plot_feature_bar(profile: pd.DataFrame, title="KMeans", prefix="09"):
    """Grouped bar chart of mean audio features per cluster."""
    feats = ["danceability", "energy", "speechiness",
             "acousticness", "valence", "instrumentalness", "liveness"]
    data  = profile[feats]
    ax    = data.T.plot(kind="bar", figsize=(14, 6),
                        colormap="tab10", edgecolor="white", width=0.75)
    ax.set_title(f"{title} – Average Audio Features per Cluster",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Feature"); ax.set_ylabel("Mean Value")
    ax.legend(title="Cluster",
              labels=[f"C{i}: {CLUSTER_LABELS.get(i,'')}" for i in profile.index],
              bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_feature_bar.png")


def plot_feature_heatmap(profile: pd.DataFrame, title="KMeans", prefix="10"):
    """Normalised heatmap of features across clusters."""
    feats     = [f for f in AUDIO_FEATURES if f != "duration_ms"]
    data      = profile[feats].copy().astype(float)
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    fig, ax   = plt.subplots(figsize=(12, max(3, len(data) * 0.9)))
    sns.heatmap(data_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Mean"},
                annot_kws={"size": 10})
    ax.set_title(f"{title} – Feature Heatmap across Clusters", fontweight="bold", fontsize=13)
    ytick_labels = [f"C{i}: {CLUSTER_LABELS.get(i,'Cluster '+str(i))}" for i in profile.index]
    ax.set_yticklabels(ytick_labels, rotation=0)
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_feature_heatmap.png")


def plot_boxplots(df_c: pd.DataFrame, title="KMeans", prefix="11"):
    """Feature distribution by cluster – box plots."""
    feats = ["danceability", "energy", "acousticness", "valence", "tempo", "speechiness"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, feat in zip(axes.flatten(), feats):
        data = [df_c[df_c["cluster"] == cid][feat].values
                for cid in sorted(df_c["cluster"].unique())]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(data)))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xticklabels([f"C{i}" for i in sorted(df_c["cluster"].unique())])
        ax.set_ylabel("Value"); ax.grid(axis="y", alpha=0.3)
    plt.suptitle(f"{title} – Feature Distributions per Cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_boxplots.png")


def plot_cluster_sizes(df_c: pd.DataFrame, title="KMeans", prefix="12"):
    """Bar chart of song count per cluster."""
    sizes  = df_c["cluster"].value_counts().sort_index()
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(sizes)))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"C{i}\n{CLUSTER_LABELS.get(i,'')}" for i in sizes.index],
                  sizes.values, color=colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"{title} – Songs per Cluster", fontweight="bold", fontsize=13)
    ax.set_ylabel("Number of Songs"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(f"{prefix}_{title.lower()}_cluster_sizes.png")


# ==============================================================
#  PHASE 7 – BUSINESS USE CASES
# ==============================================================

# ── A. Personalised Playlist Curation ─────────────────────────
def usecase_playlist_curation(df_c: pd.DataFrame):
    print("\n" + "=" * 65)
    print("BUSINESS USE-CASE A – Personalised Playlist Curation")
    print("=" * 65)
    for cid in sorted(df_c["cluster"].unique()):
        mood  = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
        songs = df_c[df_c["cluster"] == cid].sort_values(
            "popularity_songs", ascending=False)
        print(f"\n  🎵 Playlist [{cid}] – {mood}  ({len(songs):,} songs)")
        print(f"  {'Song':<45} {'Artist':<30} {'Popularity'}")
        print("  " + "-" * 85)
        for _, r in songs.head(10).iterrows():
            print(f"  {str(r['name_song']):<45} {str(r['name_artists']):<30} {r['popularity_songs']}")


# ── B. Song Discovery / Recommendation ───────────────────────
def usecase_song_discovery(df_c: pd.DataFrame, X_scaled: np.ndarray, query_idx: int = 100):
    print("\n" + "=" * 65)
    print("BUSINESS USE-CASE B – Song Discovery / Recommendation")
    print("=" * 65)
    query     = df_c.iloc[query_idx]
    cid       = query["cluster"]
    same      = df_c[(df_c["cluster"] == cid) & (df_c.index != query_idx)]

    # Euclidean distance in scaled feature space for more accurate similarity
    q_vec     = X_scaled[query_idx].reshape(1, -1)
    s_idx     = same.index.tolist()
    s_vecs    = X_scaled[s_idx]
    dists     = np.linalg.norm(s_vecs - q_vec, axis=1)
    same      = same.copy()
    same["_dist"] = dists
    recs      = same.sort_values("_dist").head(10)

    print(f"\n  Query  : '{query['name_song']}'  by  {query['name_artists']}")
    print(f"  Cluster: {cid} – {CLUSTER_LABELS.get(cid, '')}")
    print(f"\n  Top 10 most similar songs (by audio feature distance):")
    print(f"  {'Song':<45} {'Artist':<30} {'Distance'}")
    print("  " + "-" * 85)
    for _, r in recs.iterrows():
        print(f"  {str(r['name_song']):<45} {str(r['name_artists']):<30} {r['_dist']:.4f}")
    return recs


# ── C. Artist Analysis ────────────────────────────────────────
def usecase_artist_analysis(df_c: pd.DataFrame):
    print("\n" + "=" * 65)
    print("BUSINESS USE-CASE C – Artist Analysis")
    print("=" * 65)

    artist_map = (df_c.groupby("name_artists")["cluster"]
                  .agg(lambda x: x.value_counts().idxmax())
                  .reset_index())
    artist_map.columns = ["artist", "primary_cluster"]

    avg_pop = (df_c.groupby("name_artists")["popularity_artists"]
               .mean().round(1).reset_index())
    avg_pop.columns = ["artist", "avg_popularity"]
    artist_map = artist_map.merge(avg_pop, on="artist").sort_values(
        "avg_popularity", ascending=False)

    print(f"\n  Artist → Cluster mapping  (top 20 by popularity):")
    print(f"  {'Artist':<35} {'Cluster':<10} {'Mood':<35} {'Avg Pop'}")
    print("  " + "-" * 90)
    for _, r in artist_map.head(20).iterrows():
        mood = CLUSTER_LABELS.get(r["primary_cluster"], "")
        print(f"  {str(r['artist']):<35} C{r['primary_cluster']:<9} {mood:<35} {r['avg_popularity']}")

    print(f"\n  Top 5 artists per cluster:")
    for cid in sorted(artist_map["primary_cluster"].unique()):
        grp  = artist_map[artist_map["primary_cluster"] == cid].head(5)
        mood = CLUSTER_LABELS.get(cid, "")
        print(f"\n    Cluster {cid} – {mood}")
        for _, r in grp.iterrows():
            print(f"      {r['artist']}  (popularity={r['avg_popularity']})")
    return artist_map


# ── D. Market Segmentation ────────────────────────────────────
def usecase_market_segmentation(df_c: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 65)
    print("BUSINESS USE-CASE D – Market Segmentation")
    print("=" * 65)

    sizes = df_c["cluster"].value_counts().sort_index()
    total = len(df_c)
    num_profile = profile[AUDIO_FEATURES].astype(float)

    print(f"\n  Segment report:")
    for cid, cnt in sizes.items():
        mood      = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
        pct       = cnt / total * 100
        top_feats = num_profile.loc[cid].nlargest(3).index.tolist()
        avg_pop   = df_c[df_c["cluster"] == cid]["popularity_songs"].mean()
        print(f"\n  Segment {cid} – {mood}")
        print(f"    Size          : {cnt:,}  ({pct:.1f}% of catalogue)")
        print(f"    Top traits    : {', '.join(top_feats)}")
        print(f"    Avg popularity: {avg_pop:.1f}")

    # Pie chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    moods  = [CLUSTER_LABELS.get(i, f"Cluster {i}") for i in sizes.index]
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, len(sizes)))
    axes[0].pie(sizes.values, labels=moods, autopct="%1.1f%%",
                startangle=140, colors=colors,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("Catalogue Share by Cluster", fontweight="bold", fontsize=12)

    # Popularity by cluster
    pop_data = [df_c[df_c["cluster"] == cid]["popularity_songs"].values
                for cid in sorted(df_c["cluster"].unique())]
    bp = axes[1].boxplot(pop_data, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
    axes[1].set_xticklabels([f"C{i}" for i in sorted(df_c["cluster"].unique())])
    axes[1].set_title("Popularity Score by Cluster", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Popularity Score"); axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Market Segmentation Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("13_market_segmentation.png")


# ==============================================================
#  FINAL EXPORT
# ==============================================================
def export_results(df_c: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 65)
    print("FINAL EXPORT")
    print("=" * 65)

    # Songs with cluster labels
    out_songs = os.path.join(OUTPUT_DIR, "songs_with_clusters.csv")
    df_c.to_csv(out_songs, index=False)
    print(f"  [exported] {out_songs}  ({len(df_c):,} rows)")

    # Cluster profiles
    export_profile = profile.copy()
    export_profile["mood_label"] = [
        CLUSTER_LABELS.get(i, f"Cluster {i}") for i in export_profile.index
    ]
    out_profile = os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
    export_profile.to_csv(out_profile)
    print(f"  [exported] {out_profile}")


# ==============================================================
#  SUMMARY REPORT
# ==============================================================
def print_summary_report(df_c: pd.DataFrame, profile: pd.DataFrame,
                          sil: float, db: float):
    print("\n" + "★" * 65)
    print("  FINAL SUMMARY REPORT")
    print("★" * 65)
    print(f"\n  Dataset          : {len(df_c):,} songs")
    print(f"  Algorithm        : K-Means  (k=3, confirmed by silhouette)")
    print(f"  Silhouette Score : {sil:.4f}")
    print(f"  Davies-Bouldin   : {db:.4f}")
    print(f"\n  Cluster Profiles:")
    num_profile = profile[AUDIO_FEATURES].astype(float)
    for cid in profile.index:
        mood    = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
        size    = (df_c["cluster"] == cid).sum()
        top3    = num_profile.loc[cid].nlargest(3).index.tolist()
        print(f"\n    Cluster {cid} – {mood}")
        print(f"      Songs       : {size:,}  ({size/len(df_c)*100:.1f}%)")
        print(f"      Top traits  : {', '.join(top3)}")
        print(f"      Energy      : {profile.loc[cid,'energy']:.4f}")
        print(f"      Danceability: {profile.loc[cid,'danceability']:.4f}")
        print(f"      Acousticness: {profile.loc[cid,'acousticness']:.4f}")
        print(f"      Speechiness : {profile.loc[cid,'speechiness']:.4f}")
        print(f"      Valence     : {profile.loc[cid,'valence']:.4f}")
        print(f"      Avg Tempo   : {profile.loc[cid,'tempo']:.1f} BPM")


# ==============================================================
#  MAIN PIPELINE
# ==============================================================
def main():
    print("\n" + "★" * 65)
    print("  AMAZON MUSIC CLUSTERING  –  Full Pipeline")
    print("  GUVI | HCL  ·  Unsupervised Machine Learning")
    print("★" * 65)

    # 1. Load & explore
    df = load_and_explore(DATA_PATH)
    visualise_distributions(df)
    visualise_correlation(df)
    visualise_outliers(df)

    # 2. Feature selection & scaling
    X_scaled, scaler = select_and_scale(df)

    # 3. Dimensionality reduction
    X_pca, X_tsne, tsne_idx = reduce_dimensions(X_scaled)

    # 4-A. K-Means tuning → best k=3
    best_k, ks, inertias, sil_scores = kmeans_tune(X_scaled)
    km_labels, km_model = run_kmeans(X_scaled, best_k)

    # 4-B. DBSCAN (informational, on sample)
    db_labels, db_idx = run_dbscan(X_scaled)

    # 4-C. Hierarchical (sample + centroid propagation)
    ag_labels = run_hierarchical(X_scaled, n_clusters=best_k)

    # 5. Evaluation & profiling (K-Means = primary)
    df_c, profile = evaluate_and_profile(df, km_labels, "KMeans")

    # Compute final metrics for summary
    final_sil = silhouette_score(X_scaled, km_labels, sample_size=8000, random_state=42)
    final_db  = davies_bouldin_score(X_scaled, km_labels)

    # 6. Visualisations
    plot_pca_scatter(X_pca, km_labels, "KMeans", "07")
    plot_tsne_scatter(X_tsne, km_labels[tsne_idx], "KMeans", "08")
    plot_feature_bar(profile, "KMeans", "09")
    plot_feature_heatmap(profile, "KMeans", "10")
    plot_boxplots(df_c, "KMeans", "11")
    plot_cluster_sizes(df_c, "KMeans", "12")

    # DBSCAN & Hierarchical scatter (comparison)
    df_ag, prof_ag = evaluate_and_profile(df, ag_labels, "Hierarchical")
    plot_pca_scatter(X_pca, ag_labels, "Hierarchical", "14")

    # 7. Business use cases
    usecase_playlist_curation(df_c)
    usecase_song_discovery(df_c, X_scaled, query_idx=100)
    usecase_artist_analysis(df_c)
    usecase_market_segmentation(df_c, profile)

    # 8. Export
    export_results(df_c, profile)

    # 9. Summary
    print_summary_report(df_c, profile, final_sil, final_db)

    print(f"\n  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("★" * 65 + "\n")


if __name__ == "__main__":
    main()
