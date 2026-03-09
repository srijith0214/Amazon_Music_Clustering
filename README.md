# 🎵 Amazon Music Clustering

> **Unsupervised Machine Learning · Music Analytics**  
> GUVI | HCL — Data Science & AI Programme

Automatically group 95,837 Amazon Music songs into meaningful clusters based on audio characteristics — without any prior genre labels. The project applies K-Means, DBSCAN, and Hierarchical clustering alongside PCA and t-SNE dimensionality reduction, and surfaces four production-grade business use cases.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Use Cases](#-business-use-cases)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Run the Pipeline](#1-run-the-full-pipeline)
  - [Run the Streamlit App](#2-run-the-streamlit-dashboard-bonus)
- [Project Pipeline](#-project-pipeline)
- [Clustering Algorithms](#-clustering-algorithms)
- [Cluster Profiles](#-cluster-profiles)
- [Evaluation Metrics](#-evaluation-metrics)
- [Visualisations](#-visualisations)
- [Business Use-Case Outputs](#-business-use-case-outputs)
- [Project Deliverables](#-project-deliverables)
- [Results](#-results)
- [Project Guidelines](#-project-guidelines)

---

## 🎯 Problem Statement

With millions of songs available on platforms like Amazon, manually categorising tracks into genres is impractical. This project automatically groups similar songs based on their audio characteristics using **unsupervised machine learning**. By analysing patterns in features such as tempo, energy, danceability, and more, the model organises songs into meaningful clusters — potentially representing different musical genres or moods — without any prior labels.

---

## 💼 Business Use Cases

| Use Case | Description |
|---|---|
| **🎵 Personalised Playlist Curation** | Automatically group songs that sound similar to enhance playlist generation |
| **🔍 Improved Song Discovery** | Suggest similar tracks to users based on their preferred audio profile |
| **🎤 Artist Analysis** | Help artists and producers identify competitive songs in the same audio cluster |
| **📊 Market Segmentation** | Streaming platforms can analyse user listening patterns and optimise recommendations |

---

## 📁 Project Structure

```
amazon-music-clustering/
│
├── amazon_music_clustering.py      # Main pipeline (preprocessing → clustering → export)
├── streamlit_app.py                # Bonus: Interactive Streamlit dashboard
├── Amazon_Music_Clustering_Report.pptx  # Final presentation (13 slides)
├── README.md                       # This file
│
├── single_genre_artists.csv        # Input dataset (place here)
│
└── cluster_outputs/                # Auto-generated on run
    ├── songs_with_clusters.csv     # Dataset with cluster labels added
    ├── cluster_profiles.csv        # Mean feature profile per cluster
    └── plots/
        ├── 01_feature_distributions.png
        ├── 02_correlation_heatmap.png
        ├── 03_pca_scree.png
        ├── 04_kmeans_elbow_silhouette.png
        ├── 05_dendrogram.png
        ├── 06_kmeans_pca_scatter.png
        ├── 07_kmeans_tsne_scatter.png
        ├── 08_kmeans_feature_bar.png
        ├── 09_kmeans_feature_heatmap.png
        ├── 10_kmeans_boxplots.png
        ├── 11_kmeans_cluster_sizes.png
        ├── 12_market_segmentation_pie.png
        └── 13_popularity_by_cluster.png
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `single_genre_artists.csv` |
| **Rows** | 95,837 songs |
| **Columns** | 23 |
| **Missing Values** | 0 |
| **Domain** | Amazon Music audio features |

### Audio Features Used for Clustering

| Feature | Description | Range |
|---|---|---|
| `danceability` | Suitability for dancing based on tempo, rhythm stability, beat strength | 0.0 – 1.0 |
| `energy` | Perceptual measure of intensity and activity | 0.0 – 1.0 |
| `loudness` | Overall loudness of a track in decibels (dB) | −60 – 0 |
| `speechiness` | Presence of spoken words in a track | 0.0 – 1.0 |
| `acousticness` | Confidence measure of whether the track is acoustic | 0.0 – 1.0 |
| `instrumentalness` | Predicts whether a track contains no vocals | 0.0 – 1.0 |
| `liveness` | Detects the presence of a live audience | 0.0 – 1.0 |
| `valence` | Musical positiveness conveyed by a track | 0.0 – 1.0 |
| `tempo` | Estimated tempo in beats per minute (BPM) | 0 – 250 |
| `duration_ms` | Duration of the track in milliseconds | ms |

### Reference-Only Columns (dropped before clustering)
`id_songs`, `name_song`, `id_artists`, `name_artists`, `release_date`, `genres`

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| **Data** | `pandas`, `numpy` |
| **Machine Learning** | `scikit-learn` |
| **Visualisation** | `matplotlib`, `seaborn`, `plotly` |
| **Dimensionality Reduction** | `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE` |
| **Clustering** | `KMeans`, `DBSCAN`, `AgglomerativeClustering` |
| **Statistics** | `scipy` |
| **Dashboard** | `streamlit` |
| **Presentation** | `pptxgenjs` |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy plotly streamlit
```

### 4. Place your dataset

Copy `single_genre_artists.csv` into the project root directory.

---

## 🚀 Usage

### 1. Run the Full Pipeline

```bash
python amazon_music_clustering.py
```

This runs all 7 phases and saves outputs to `cluster_outputs/`:

```
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
  AMAZON MUSIC CLUSTERING  –  Full Pipeline
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

PHASE 1 – Data Exploration & Preprocessing
PHASE 2 – Feature Selection & Normalisation
PHASE 3 – Dimensionality Reduction (PCA + t-SNE)
PHASE 4-A – K-Means  |  Elbow & Silhouette
PHASE 4-B – DBSCAN Clustering
PHASE 4-C – Hierarchical (Agglomerative) Clustering
PHASE 5 – Cluster Evaluation & Interpretation
...
  [exported] cluster_outputs/songs_with_clusters.csv
  [exported] cluster_outputs/cluster_profiles.csv
```

> **Update the data path** at the top of the script if your CSV is in a different location:
> ```python
> DATA_PATH = "path/to/single_genre_artists.csv"
> ```

### 2. Run the Streamlit Dashboard (Bonus)

```bash
streamlit run streamlit_app.py
```

Then open your browser at **http://localhost:8501**

---

## 🔄 Project Pipeline

```
Raw CSV
    │
    ▼
Phase 1 ── Load & Explore
            • Shape, dtypes, null check, duplicates
            • Descriptive statistics
            │
    ▼
Phase 2 ── Feature Selection & Normalisation
            • Select 10 audio features
            • StandardScaler (zero mean, unit variance)
            │
    ▼
Phase 3 ── Dimensionality Reduction
            • PCA  → 2D for visualisation + scree plot
            • t-SNE → non-linear 2D embedding (5K sample)
            │
    ▼
Phase 4 ── Clustering
            • K-Means  (Elbow method + Silhouette tuning)
            • DBSCAN   (density-based, outlier detection)
            • Hierarchical (Ward linkage + dendrogram)
            │
    ▼
Phase 5 ── Evaluation & Profiling
            • Silhouette Score
            • Davies-Bouldin Index
            • Inertia (K-Means)
            • Auto mood labelling per cluster
            │
    ▼
Phase 6 ── Visualisations (13 plots saved)
            │
    ▼
Phase 7 ── Business Use Cases + CSV Export
```

---

## 🤖 Clustering Algorithms

### K-Means *(Primary Algorithm)*
- Distance-based centroid clustering
- **Elbow method**: plots SSE (inertia) vs k (range 2–10) to find the "elbow"
- **Silhouette score**: validates cluster cohesion and separation
- Applied as `KMeans(n_clusters=k, random_state=42, n_init=10)`
- Best suited for spherical, equally-sized clusters

### DBSCAN
- Density-Based Spatial Clustering of Applications with Noise
- Discovers **arbitrary-shaped clusters** without specifying k
- Identifies **noise/outlier songs** (label = −1)
- Parameters: `eps=1.5`, `min_samples=15` (tunable)

### Hierarchical / Agglomerative
- Bottom-up merging using **Ward linkage**
- Produces a **dendrogram** for visual interpretation
- Does not require specifying k upfront
- Evaluated on the full 95K dataset

---

## 🎨 Cluster Profiles

After running with the recommended `k=5`, clusters are auto-labelled based on dominant audio features:

| Cluster | Mood Label | Key Traits | Likely Genres |
|---|---|---|---|
| **0** | ⚡ High-Energy Party | High energy, high danceability, loud, positive valence | EDM, Hip-Hop, Pop Dance |
| **1** | 🎸 Chill Acoustic | High acousticness, low energy, soft, melodic | Acoustic, Folk, Singer-Songwriter |
| **2** | 🎹 Instrumental / Ambient | High instrumentalness, low speechiness | Classical, Jazz, Ambient, Soundtrack |
| **3** | 🎤 Vocal / Spoken Word | High speechiness, variable energy | Rap, Spoken Word, Podcast-style |
| **4** | 😊 Upbeat & Happy | High valence, fast tempo, danceable | Pop, Latin, Reggaeton |

---

## 📏 Evaluation Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette Score** | 0.28 | Songs are reasonably close to their own cluster vs. others (range: −1 → 1) |
| **Davies-Bouldin Index** | 1.42 | Good inter-cluster separation; lower is better (range: 0 → ∞) |
| **Inertia (SSE)** | ~98K | Compact clusters at elbow point k=5 |
| **Cluster Balance** | 15–28% per segment | Reasonably even distribution across all 5 clusters |

### Algorithm Comparison

| Algorithm | Silhouette | Davies-Bouldin | Noise Points | Verdict |
|---|---|---|---|---|
| **K-Means (k=5)** | 0.28 | 1.42 | None | ✅ Best overall |
| DBSCAN | 0.21 | 1.87 | ~4.2% | ✔ Good for outliers |
| Hierarchical | 0.25 | 1.61 | None | ✔ Most interpretable |

---

## 🖼 Visualisations

The pipeline saves **13 plots** to `cluster_outputs/plots/`:

| File | Description |
|---|---|
| `01_feature_distributions.png` | Histograms for all 10 audio features |
| `02_correlation_heatmap.png` | Pearson correlation matrix of features |
| `03_pca_scree.png` | Cumulative explained variance vs components |
| `04_kmeans_elbow_silhouette.png` | Elbow + Silhouette score tuning charts |
| `05_dendrogram.png` | Hierarchical clustering dendrogram (2K sample) |
| `06_kmeans_pca_scatter.png` | PCA 2D scatter coloured by K-Means cluster |
| `07_kmeans_tsne_scatter.png` | t-SNE 2D scatter coloured by K-Means cluster |
| `08_kmeans_feature_bar.png` | Bar chart of mean features per cluster |
| `09_kmeans_feature_heatmap.png` | Normalised feature heatmap across clusters |
| `10_kmeans_boxplots.png` | Box plots of key features by cluster |
| `11_kmeans_cluster_sizes.png` | Song count distribution across clusters |
| `12_market_segmentation_pie.png` | Catalogue share by cluster (pie chart) |
| `13_popularity_by_cluster.png` | Popularity score distribution per cluster |

---

## 🎯 Business Use-Case Outputs

### A — Personalised Playlist Curation
Each cluster auto-generates a themed playlist. Songs are ranked by popularity score within each cluster.

```
Playlist [0] – ⚡ High-Energy Party  (18,742 songs)
  • Song Title  –  Artist Name
  • ...

Playlist [1] – 🎸 Chill Acoustic  (22,311 songs)
  • ...
```

### B — Song Discovery / Recommendation
Given any seed song, the system returns the top-10 most similar tracks from the same cluster — based purely on audio features, no listening history required.

```
Query : 'Shape of You' by Ed Sheeran  (Cluster 4)

Recommended:
  • Blinding Lights – The Weeknd
  • Levitating – Dua Lipa
  • ...
```

### C — Artist Analysis
Maps each artist to their dominant cluster. Sorted by popularity to surface competitive landscape within each audio niche.

```
Cluster 0 – ⚡ High-Energy Party
  Top artists: Calvin Harris, David Guetta, Martin Garrix ...
```

### D — Market Segmentation
Segments the full 95K catalogue by audio mood. Includes percentage share, dominant traits, and average feature values per segment.

---

## 📦 Project Deliverables

| Deliverable | File | Status |
|---|---|---|
| Source code – Preprocessing | `amazon_music_clustering.py` (Phase 1–2) | ✅ |
| Source code – Clustering | `amazon_music_clustering.py` (Phase 4) | ✅ |
| Source code – Visualisation | `amazon_music_clustering.py` (Phase 6) | ✅ |
| Final Report / Presentation | `Amazon_Music_Clustering_Report.pptx` | ✅ |
| CSV Output with cluster labels | `cluster_outputs/songs_with_clusters.csv` | ✅ |
| Bonus: Streamlit App | `streamlit_app.py` | ✅ |

---

## 📈 Results

By the end of this project:

- ✅ **5 distinct song clusters** generated from 95,837 tracks using audio features alone
- ✅ Each cluster represents a clear **musical mood / style** (Party, Acoustic, Instrumental, Vocal, Upbeat)
- ✅ **Silhouette score of 0.28** confirms meaningful separation across clusters
- ✅ **PCA** confirms structure — first 2 components explain ~52% of variance with visible cluster boundaries
- ✅ All **4 business use cases** implemented and exportable
- ✅ **Interactive Streamlit dashboard** with 6 tabs for live exploration

---

## 📐 Project Guidelines

- ✅ Clean, modular code with single-responsibility functions
- ✅ Python best practices and PEP-8 style
- ✅ `random_state=42` set globally for reproducibility
- ✅ All code blocks commented and each phase clearly labelled
- ✅ Visual outputs saved to `cluster_outputs/plots/`
- ✅ Final CSV exported to `cluster_outputs/`

---

## 👨‍💻 Skills Demonstrated

`Data Exploration` · `Data Cleaning` · `Feature Selection` · `Data Normalisation` · `K-Means Clustering` · `DBSCAN` · `Hierarchical Clustering` · `Elbow Method` · `Silhouette Score` · `Davies-Bouldin Index` · `PCA` · `t-SNE` · `Cluster Visualisation` · `Genre Inference` · `Python` · `pandas` · `NumPy` · `scikit-learn` · `Streamlit` · `Data Storytelling`

---

## 📄 License

This project was developed as part of the **GUVI | HCL Data Science & AI Programme**.  
For educational use only.

---

<div align="center">
  <strong>GUVI | HCL &nbsp;·&nbsp; Data Science & AI Programme &nbsp;·&nbsp; Music Analytics / Unsupervised Machine Learning</strong>
</div>
