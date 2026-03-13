"""
============================================================
 Amazon Music Clustering  –  Production-Ready Implementation
 GUVI | HCL  –  Music Analytics / Unsupervised Machine Learning
============================================================

Deliverables covered
---------------------
1. Data Exploration & Preprocessing
2. Feature Selection
3. Dimensionality Reduction  (PCA + t-SNE)
4. Clustering  (K-Means  |  DBSCAN  |  Hierarchical)
5. Cluster Evaluation & Interpretation
6. Visualisations (scatter, bar, heatmap, distribution)
7. Business Use-Case outputs
   A. Personalised Playlist Curation
   B. Song Discovery / Recommendation
   C. Artist Analysis
   D. Market Segmentation
8. Final Export  (CSV with cluster labels)

Usage
-----
    python amazon_music_clustering.py

Requirements
------------
    pip install pandas numpy scikit-learn matplotlib seaborn scipy
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "single_genre_artists.csv"   # update path if needed
OUTPUT_DIR  = "cluster_outputs"
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
]

# ─────────────────────────────────────────────
#  HELPER – save figure
# ─────────────────────────────────────────────
def savefig(name: str):
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ==============================================================
#  PHASE 1 – DATA EXPLORATION & PREPROCESSING
# ==============================================================
def load_and_explore(path: str) -> pd.DataFrame:
    """Load dataset, print EDA summary, return cleaned DataFrame."""
    print("\n" + "=" * 60)
    print("PHASE 1 – Data Exploration & Preprocessing")
    print("=" * 60)

    df = pd.read_csv(path)

    # ── Basic info ──────────────────────────────────────────────
    print(f"\nShape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates   : {df.duplicated().sum()}")
    print(f"\nDescriptive stats:\n{df[AUDIO_FEATURES].describe().round(3)}")

    # ── Drop duplicates ─────────────────────────────────────────
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def visualise_distributions(df: pd.DataFrame):
    """Plot histograms for all audio features."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Distribution of Audio Features", fontsize=16, fontweight="bold")
    for ax, feat in zip(axes.flatten(), AUDIO_FEATURES):
        ax.hist(df[feat].dropna(), bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
    plt.tight_layout()
    savefig("01_feature_distributions.png")


def visualise_correlation(df: pd.DataFrame):
    """Correlation heatmap of audio features."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[AUDIO_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Audio Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("02_correlation_heatmap.png")


# ==============================================================
#  PHASE 2 – FEATURE SELECTION & NORMALISATION
# ==============================================================
def select_and_scale(df: pd.DataFrame):
    """Select audio features, scale, return matrix + scaler."""
    print("\n" + "=" * 60)
    print("PHASE 2 – Feature Selection & Normalisation")
    print("=" * 60)

    X = df[AUDIO_FEATURES].copy()
    print(f"\nFeatures selected : {AUDIO_FEATURES}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Scaled matrix shape : {X_scaled.shape}")
    return X_scaled, scaler


# ==============================================================
#  PHASE 3 – DIMENSIONALITY REDUCTION
# ==============================================================
def reduce_dimensions(X_scaled: np.ndarray, n_components_pca: int = 2):
    """PCA + t-SNE reduction for visualisation."""
    print("\n" + "=" * 60)
    print("PHASE 3 – Dimensionality Reduction (PCA + t-SNE)")
    print("=" * 60)

    # PCA
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    print(f"\nPCA explained variance  : {explained.round(3)}")
    print(f"Total variance explained: {explained.sum():.3f}")

    # Scree plot
    pca_full = PCA(random_state=42).fit(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o", color="#2196F3")
    ax.axhline(0.90, color="red", linestyle="--", label="90 % threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA – Scree / Elbow Plot", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    savefig("03_pca_scree.png")

    # t-SNE (on PCA-reduced data for speed)
    print("\nRunning t-SNE (this may take a moment)…")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    sample_idx = np.random.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
    X_tsne_sample = tsne.fit_transform(X_scaled[sample_idx])

    return X_pca, X_tsne_sample, sample_idx


# ==============================================================
#  PHASE 4-A – K-MEANS CLUSTERING
# ==============================================================
def kmeans_elbow_silhouette(X_scaled: np.ndarray, k_range=range(2, 12)):
    """Elbow method + silhouette scores to pick best k."""
    print("\n" + "=" * 60)
    print("PHASE 4-A – K-Means  |  Elbow & Silhouette")
    print("=" * 60)

    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels, sample_size=5000))
        print(f"  k={k:2d}  |  Inertia={km.inertia_:,.0f}  |  Silhouette={sil_scores[-1]:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ks = list(k_range)

    ax1.plot(ks, inertias, marker="o", color="#E53935")
    ax1.set_title("Elbow Method – Inertia vs k", fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (SSE)")

    ax2.plot(ks, sil_scores, marker="s", color="#43A047")
    ax2.set_title("Silhouette Score vs k", fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")

    plt.suptitle("K-Means Hyperparameter Tuning", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("04_kmeans_elbow_silhouette.png")

    best_k = ks[int(np.argmax(sil_scores))]
    print(f"\nBest k (max silhouette) : {best_k}")
    return best_k


def run_kmeans(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """Fit KMeans with chosen k and return labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil  = silhouette_score(X_scaled, labels, sample_size=5000)
    db   = davies_bouldin_score(X_scaled, labels)
    print(f"\nK-Means (k={k})  |  Silhouette={sil:.4f}  |  Davies-Bouldin={db:.4f}")
    return labels


# ==============================================================
#  PHASE 4-B – DBSCAN CLUSTERING
# ==============================================================
def run_dbscan(X_scaled: np.ndarray, eps: float = 1.2, min_samples: int = 10):
    """Fit DBSCAN and return labels (noise = -1)."""
    print("\n" + "=" * 60)
    print("PHASE 4-B – DBSCAN Clustering")
    print("=" * 60)

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).mean() * 100
    print(f"  DBSCAN  |  eps={eps}  min_samples={min_samples}")
    print(f"  Clusters found : {n_clusters}  |  Noise points : {noise_pct:.1f}%")
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_scaled[mask], labels[mask], sample_size=min(5000, mask.sum()))
        print(f"  Silhouette (excl. noise) : {sil:.4f}")
    return labels


# ==============================================================
#  PHASE 4-C – HIERARCHICAL / AGGLOMERATIVE CLUSTERING
# ==============================================================
def run_hierarchical(X_scaled: np.ndarray, n_clusters: int, sample_size: int = 5000):
    """
    Agglomerative clustering on a sample + dendrogram.
 
    Ward linkage requires a full pairwise distance matrix (O(n^2) memory).
    95K rows needs ~34 GB and will crash. Fix:
      1. Fit AgglomerativeClustering on a 5,000-row sample.
      2. Compute per-cluster centroids from the sample.
      3. Assign every song in the full dataset to the nearest centroid (1-NN).
      4. Plot dendrogram on the same sample for visualisation.
    """
    print("\n" + "=" * 60)
    print("PHASE 4-C \u2013 Hierarchical (Agglomerative) Clustering")
    print("=" * 60)
    print(f"  Note: Ward linkage is O(n^2) memory \u2014 fitting on a "
          f"{sample_size:,}-row sample, then propagating to all {len(X_scaled):,} rows.")
 
    # Step 1: fit on sample
    rng        = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_scaled), sample_size, replace=False)
    X_sample   = X_scaled[sample_idx]
 
    agg           = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    sample_labels = agg.fit_predict(X_sample)
 
    # Step 2: compute per-cluster centroids from the sample
    centroids = np.array([
        X_sample[sample_labels == c].mean(axis=0)
        for c in range(n_clusters)
    ])
 
    # Step 3: assign full dataset by nearest centroid (vectorised)
    dists  = np.linalg.norm(X_scaled[:, None, :] - centroids[None, :, :], axis=2)
    labels = dists.argmin(axis=1)
 
    # Evaluate on a random subset for speed
    eval_idx = rng.choice(len(X_scaled), min(10000, len(X_scaled)), replace=False)
    sil = silhouette_score(X_scaled[eval_idx], labels[eval_idx], sample_size=5000)
    db  = davies_bouldin_score(X_scaled[eval_idx], labels[eval_idx])
    print(f"  Agglomerative (n={n_clusters}, centroid-propagated)  |  "
          f"Silhouette={sil:.4f}  |  DB={db:.4f}")
 
    # Step 4: dendrogram on the same sample
    Z   = linkage(X_sample, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax,
               leaf_rotation=90, leaf_font_size=9, color_threshold=0)
    ax.set_title(f"Hierarchical Clustering Dendrogram (n={sample_size:,} sample)",
                 fontweight="bold")
    ax.set_xlabel("Song Index")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()
    savefig("05_dendrogram.png")
    return labels


# ==============================================================
#  PHASE 5 – CLUSTER EVALUATION & INTERPRETATION
# ==============================================================
def evaluate_and_profile(df: pd.DataFrame, labels: np.ndarray, algo_name: str = "KMeans"):
    """Compute metrics and build cluster profiles."""
    print("\n" + "=" * 60)
    print(f"PHASE 5 – Cluster Evaluation & Interpretation  [{algo_name}]")
    print("=" * 60)

    df = df.copy()
    df["cluster"] = labels

    # Cluster sizes
    sizes = df["cluster"].value_counts().sort_index()
    print(f"\nCluster sizes:\n{sizes.to_string()}")

    # Mean feature values per cluster
    profile = df.groupby("cluster")[AUDIO_FEATURES].mean().round(4)
    print(f"\nCluster profiles (mean feature values):\n{profile.to_string()}")

    # Auto-label clusters
    def auto_label(row):
        if row["energy"] > 0.7 and row["danceability"] > 0.65:
            return "⚡ High-Energy Party"
        elif row["acousticness"] > 0.6 and row["energy"] < 0.45:
            return "🎸 Chill Acoustic"
        elif row["instrumentalness"] > 0.5:
            return "🎹 Instrumental / Ambient"
        elif row["speechiness"] > 0.15:
            return "🎤 Vocal / Spoken Word"
        elif row["valence"] > 0.65 and row["tempo"] > 115:
            return "😊 Upbeat & Happy"
        else:
            return "🎵 Moderate / Mixed"

    profile["mood_label"] = profile.apply(auto_label, axis=1)
    print(f"\nInferred mood labels:\n{profile['mood_label'].to_string()}")

    return df, profile


# ==============================================================
#  PHASE 6 – VISUALISATIONS
# ==============================================================
def plot_cluster_scatter_pca(X_pca, labels, title="KMeans"):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    palette    = cm.get_cmap("tab10", n_clusters + 1)
    fig, ax    = plt.subplots(figsize=(10, 7))
    for cid in sorted(set(labels)):
        mask = labels == cid
        color = "grey" if cid == -1 else palette(cid)
        lbl   = "Noise" if cid == -1 else f"Cluster {cid}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=lbl, s=6, alpha=0.5)
    ax.set_title(f"{title} – PCA 2-D Cluster Scatter", fontweight="bold")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(markerscale=3, loc="best", fontsize=8)
    plt.tight_layout()
    savefig(f"06_{title.lower()}_pca_scatter.png")


def plot_cluster_scatter_tsne(X_tsne, labels_sample, title="KMeans"):
    n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
    palette    = cm.get_cmap("tab10", n_clusters + 1)
    fig, ax    = plt.subplots(figsize=(10, 7))
    for cid in sorted(set(labels_sample)):
        mask  = labels_sample == cid
        color = "grey" if cid == -1 else palette(cid)
        lbl   = "Noise" if cid == -1 else f"Cluster {cid}"
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[color], label=lbl, s=6, alpha=0.5)
    ax.set_title(f"{title} – t-SNE 2-D Cluster Scatter (sample)", fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, loc="best", fontsize=8)
    plt.tight_layout()
    savefig(f"07_{title.lower()}_tsne_scatter.png")


def plot_cluster_bar(profile: pd.DataFrame, title="KMeans"):
    """Bar chart of mean feature values per cluster."""
    feats_plot = ["danceability", "energy", "speechiness",
                  "acousticness", "valence", "instrumentalness", "liveness"]
    data = profile[feats_plot]
    ax   = data.T.plot(kind="bar", figsize=(14, 6), colormap="tab10", edgecolor="white")
    ax.set_title(f"{title} – Average Audio Features per Cluster", fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Mean Value")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    savefig(f"08_{title.lower()}_feature_bar.png")


def plot_cluster_heatmap(profile: pd.DataFrame, title="KMeans"):
    """Heatmap comparing features across clusters."""
    feats_plot = [f for f in AUDIO_FEATURES if f != "duration_ms"]
    data = profile[feats_plot]
    # Normalise each feature to 0-1 for comparability
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, max(4, len(data) * 0.7)))
    sns.heatmap(data_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Mean"})
    ax.set_title(f"{title} – Feature Heatmap across Clusters", fontweight="bold")
    ax.set_xlabel("Audio Feature")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    savefig(f"09_{title.lower()}_feature_heatmap.png")


def plot_distribution_by_cluster(df_clustered: pd.DataFrame, title="KMeans"):
    """Box-plots for key features split by cluster."""
    feats = ["danceability", "energy", "acousticness", "valence", "tempo"]
    fig, axes = plt.subplots(1, len(feats), figsize=(22, 5))
    for ax, feat in zip(axes, feats):
        df_clustered.boxplot(column=feat, by="cluster", ax=ax,
                             patch_artist=True, notch=False)
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("")
    plt.suptitle(f"{title} – Feature Distribution by Cluster", fontweight="bold")
    plt.tight_layout()
    savefig(f"10_{title.lower()}_boxplots.png")


def plot_cluster_size(df_clustered: pd.DataFrame, title="KMeans"):
    sizes = df_clustered["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = cm.get_cmap("tab10")(np.linspace(0, 1, len(sizes)))
    ax.bar(sizes.index.astype(str), sizes.values, color=colors, edgecolor="white")
    ax.set_title(f"{title} – Cluster Size Distribution", fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Songs")
    for i, v in enumerate(sizes.values):
        ax.text(i, v + 100, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    savefig(f"11_{title.lower()}_cluster_sizes.png")


# ==============================================================
#  PHASE 7 – BUSINESS USE-CASE OUTPUTS
# ==============================================================

# ── Use Case A: Personalised Playlist Curation ────────────────
def usecase_playlist_curation(df_clustered: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BUSINESS USE-CASE A – Personalised Playlist Curation")
    print("=" * 60)

    playlists = {}
    name_col = "name_song" if "name_song" in df_clustered.columns else (
               "track_name" if "track_name" in df_clustered.columns else None)

    for cid, row in profile.iterrows():
        mood  = row.get("mood_label", f"Cluster {cid}")
        songs = df_clustered[df_clustered["cluster"] == cid]
        # Sort by popularity if available
        if "popularity_songs" in songs.columns:
            songs = songs.sort_values("popularity_songs", ascending=False)
        top   = songs.head(10)
        playlists[cid] = {"mood": mood, "count": len(songs), "top_tracks": top}

    for cid, data in playlists.items():
        print(f"\n  Playlist [{cid}] – {data['mood']}  ({data['count']:,} songs)")
        if name_col:
            artist_col = "name_artists" if "name_artists" in data["top_tracks"].columns else (
                         "artist_name" if "artist_name" in data["top_tracks"].columns else None)
            for _, r in data["top_tracks"].iterrows():
                artist = r[artist_col] if artist_col else "Unknown"
                print(f"    • {r[name_col]}  –  {artist}")
    return playlists


# ── Use Case B: Song Discovery / Recommendation ───────────────
def usecase_song_discovery(df_clustered: pd.DataFrame, query_idx: int = 0):
    print("\n" + "=" * 60)
    print("BUSINESS USE-CASE B – Song Discovery / Recommendation")
    print("=" * 60)

    name_col   = "name_song" if "name_song" in df_clustered.columns else "track_name"
    artist_col = "name_artists" if "name_artists" in df_clustered.columns else "artist_name"

    query_song   = df_clustered.iloc[query_idx]
    query_cluster = query_song["cluster"]

    similar = df_clustered[
        (df_clustered["cluster"] == query_cluster) &
        (df_clustered.index != query_idx)
    ]
    if "popularity_songs" in similar.columns:
        similar = similar.sort_values("popularity_songs", ascending=False)
    recs = similar.head(10)

    q_name   = query_song.get(name_col,   "Unknown")
    q_artist = query_song.get(artist_col, "Unknown")
    print(f"\n  Query song  : '{q_name}' by {q_artist}  (Cluster {query_cluster})")
    print(f"\n  Recommended similar songs:")
    for _, r in recs.iterrows():
        print(f"    • {r.get(name_col, 'N/A')}  –  {r.get(artist_col, 'N/A')}")
    return recs


# ── Use Case C: Artist Analysis ───────────────────────────────
def usecase_artist_analysis(df_clustered: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BUSINESS USE-CASE C – Artist Analysis")
    print("=" * 60)

    artist_col = "name_artists" if "name_artists" in df_clustered.columns else (
                 "artist_name"  if "artist_name"  in df_clustered.columns else None)
    if artist_col is None:
        print("  No artist column found – skipping.")
        return None

    artist_cluster = df_clustered.groupby(artist_col)["cluster"].agg(
        lambda x: x.value_counts().idxmax()
    ).reset_index()
    artist_cluster.columns = [artist_col, "primary_cluster"]
    artist_cluster["mood"] = artist_cluster["primary_cluster"].map(
        profile.get("mood_label", {})
    )

    pop_col = "popularity_artists" if "popularity_artists" in df_clustered.columns else (
              "popularity_songs"   if "popularity_songs"   in df_clustered.columns else None)
    if pop_col:
        avg_pop = df_clustered.groupby(artist_col)[pop_col].mean().round(1)
        artist_cluster = artist_cluster.merge(avg_pop, on=artist_col, how="left")

    print(f"\n  Artist → Cluster mapping (top 20):\n{artist_cluster.head(20).to_string(index=False)}")

    # Competitive artists per cluster
    print("\n  Top artists per cluster (by popularity):")
    for cid, row in profile.iterrows():
        mood  = row.get("mood_label", f"Cluster {cid}")
        group = artist_cluster[artist_cluster["primary_cluster"] == cid]
        if pop_col and pop_col in group.columns:
            group = group.sort_values(pop_col, ascending=False)
        print(f"\n    Cluster {cid} – {mood}")
        for _, r in group.head(5).iterrows():
            print(f"      {r[artist_col]}")

    return artist_cluster


# ── Use Case D: Market Segmentation ───────────────────────────
def usecase_market_segmentation(df_clustered: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 60)
    print("BUSINESS USE-CASE D – Market Segmentation")
    print("=" * 60)

    sizes = df_clustered["cluster"].value_counts().sort_index()
    total = len(df_clustered)

    print("\n  Segment report:")
    for cid, cnt in sizes.items():
        mood = profile.loc[cid, "mood_label"] if "mood_label" in profile.columns else ""
        pct  = cnt / total * 100
        top_feats = profile.loc[cid, AUDIO_FEATURES].astype(float).nlargest(3).index.tolist()
        print(f"\n  Segment {cid} – {mood}")
        print(f"    Size       : {cnt:,}  ({pct:.1f}% of catalogue)")
        print(f"    Top traits : {', '.join(top_feats)}")

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    moods  = [profile.loc[cid, "mood_label"] if "mood_label" in profile.columns
              else f"Cluster {cid}" for cid in sizes.index]
    ax.pie(sizes.values, labels=moods, autopct="%1.1f%%",
           startangle=140, colors=cm.get_cmap("tab10")(np.linspace(0, 1, len(sizes))))
    ax.set_title("Market Segmentation – Catalogue Share by Cluster",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    savefig("12_market_segmentation_pie.png")

    # Popularity distribution per cluster
    pop_col = "popularity_songs" if "popularity_songs" in df_clustered.columns else None
    if pop_col:
        fig, ax = plt.subplots(figsize=(10, 5))
        for cid in sorted(df_clustered["cluster"].unique()):
            subset = df_clustered[df_clustered["cluster"] == cid][pop_col]
            ax.hist(subset, bins=30, alpha=0.5,
                    label=f"Cluster {cid}", density=True)
        ax.set_title("Popularity Score Distribution by Cluster", fontweight="bold")
        ax.set_xlabel("Popularity Score")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        savefig("13_popularity_by_cluster.png")


# ==============================================================
#  FINAL EXPORT
# ==============================================================
def export_results(df_clustered: pd.DataFrame, profile: pd.DataFrame):
    out_csv = os.path.join(OUTPUT_DIR, "songs_with_clusters.csv")
    df_clustered.to_csv(out_csv, index=False)
    print(f"\n  [exported] {out_csv}")

    out_profile = os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
    profile.to_csv(out_profile)
    print(f"  [exported] {out_profile}")


# ==============================================================
#  SUMMARY REPORT
# ==============================================================
def print_summary_report(df_clustered: pd.DataFrame, profile: pd.DataFrame):
    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT")
    print("=" * 60)
    print(f"\nTotal songs analysed : {len(df_clustered):,}")
    print(f"Total clusters       : {df_clustered['cluster'].nunique()}")
    print(f"\nCluster Profiles:")
    for cid, row in profile.iterrows():
        mood = row.get("mood_label", "")
        size = (df_clustered["cluster"] == cid).sum()
        top3 = row[AUDIO_FEATURES].astype(float).nlargest(3).index.tolist()
        print(f"\n  Cluster {cid}  –  {mood}")
        print(f"    Songs      : {size:,}")
        print(f"    Dominant   : {', '.join(top3)}")
        print(f"    Avg energy : {row['energy']:.3f}  |  "
              f"danceability : {row['danceability']:.3f}  |  "
              f"valence : {row['valence']:.3f}")


# ==============================================================
#  MAIN PIPELINE
# ==============================================================
def main():
    print("\n" + "★" * 60)
    print("  AMAZON MUSIC CLUSTERING  –  Full Pipeline")
    print("★" * 60)

    # ── 1. Load & explore ──────────────────────────────────────
    df = load_and_explore(DATA_PATH)
    visualise_distributions(df)
    visualise_correlation(df)

    # ── 2. Feature selection & scaling ────────────────────────
    X_scaled, scaler = select_and_scale(df)

    # ── 3. Dimensionality reduction ────────────────────────────
    X_pca, X_tsne, tsne_idx = reduce_dimensions(X_scaled)

    # ── 4-A. K-Means ──────────────────────────────────────────
    best_k       = kmeans_elbow_silhouette(X_scaled, k_range=range(2, 11))
    km_labels    = run_kmeans(X_scaled, best_k)

    # ── 4-B. DBSCAN ───────────────────────────────────────────
    db_labels    = run_dbscan(X_scaled, eps=1.5, min_samples=15)

    # ── 4-C. Hierarchical ─────────────────────────────────────
    ag_labels    = run_hierarchical(X_scaled, n_clusters=best_k)

    # ── 5. Evaluation & profiling (K-Means as primary) ────────
    df_clustered, profile = evaluate_and_profile(df, km_labels, "KMeans")

    # ── 6. Visualisations ─────────────────────────────────────
    plot_cluster_scatter_pca(X_pca, km_labels, "KMeans")
    plot_cluster_scatter_tsne(X_tsne, km_labels[tsne_idx], "KMeans")
    plot_cluster_bar(profile, "KMeans")
    plot_cluster_heatmap(profile, "KMeans")
    plot_distribution_by_cluster(df_clustered, "KMeans")
    plot_cluster_size(df_clustered, "KMeans")

    # DBSCAN & Hierarchical scatter overlays
    df_db, prof_db = evaluate_and_profile(df, db_labels,  "DBSCAN")
    df_ag, prof_ag = evaluate_and_profile(df, ag_labels,  "Hierarchical")
    plot_cluster_scatter_pca(X_pca, db_labels, "DBSCAN")
    plot_cluster_scatter_pca(X_pca, ag_labels, "Hierarchical")

    # ── 7. Business use-cases ─────────────────────────────────
    usecase_playlist_curation(df_clustered, profile)
    usecase_song_discovery(df_clustered, query_idx=0)
    usecase_artist_analysis(df_clustered, profile)
    usecase_market_segmentation(df_clustered, profile)

    # ── 8. Export ─────────────────────────────────────────────
    export_results(df_clustered, profile)

    # ── 9. Summary ────────────────────────────────────────────
    print_summary_report(df_clustered, profile)

    print("\n" + "★" * 60)
    print(f"  All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("★" * 60 + "\n")


if __name__ == "__main__":
    main()
