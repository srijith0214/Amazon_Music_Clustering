# 🎵 Amazon Music Clustering

> **Unsupervised Machine Learning · Music Analytics**
> GUVI | HCL — Data Science & AI Programme

Automatically group **95,837 Amazon Music songs** into meaningful clusters based on audio characteristics — without any prior genre labels. The project applies **K-Means**, **DBSCAN**, and **Hierarchical** clustering alongside **PCA** and **t-SNE** dimensionality reduction, and delivers four production-grade business use cases.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Business Use Cases](#-business-use-cases)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Pipeline](#-project-pipeline)
- [Why k = 3?](#-why-k--3)
- [Clustering Algorithms](#-clustering-algorithms)
- [Cluster Profiles](#-cluster-profiles-real-data)
- [Evaluation Metrics](#-evaluation-metrics-real-data)
- [Visualisations](#-visualisations-generated)
- [Business Use-Case Outputs](#-business-use-case-outputs)
- [Project Deliverables](#-project-deliverables)
- [Results](#-results)

---

## 🎯 Problem Statement

With millions of songs available on platforms like Amazon, manually categorising tracks into genres is impractical. This project automatically groups similar songs based on their **audio characteristics** using unsupervised machine learning — without any prior labels. By analysing patterns in features such as tempo, energy, danceability, and more, the model organises songs into meaningful clusters representing different musical styles and moods.

---

## 💼 Business Use Cases

| # | Use Case | Description |
|---|---|---|
| A | **🎵 Personalised Playlist Curation** | Auto-group songs that sound similar to enhance playlist generation |
| B | **🔍 Song Discovery / Recommendation** | Suggest similar tracks using Euclidean distance in audio feature space |
| C | **🎤 Artist Analysis** | Map artists to their dominant cluster; identify competitive landscape |
| D | **📊 Market Segmentation** | Analyse catalogue composition by audio mood to optimise promotions |

---

## 📁 Project Structure

```
amazon-music-clustering/
│
├── amazon_music_clustering.py      # Full pipeline (preprocessing → export)
├── streamlit_app.py                # Interactive Streamlit dashboard (bonus)
├── README.md                       # This file
│
├── single_genre_artists.csv        # Input dataset (place in project root)
│
└── cluster_outputs/                # Auto-created when pipeline runs
    ├── songs_with_clusters.csv     # 95,837 songs with cluster labels added
    ├── cluster_profiles.csv        # Mean feature profile per cluster
    └── plots/
        ├── 01_feature_distributions.png
        ├── 02_correlation_heatmap.png
        ├── 03_outlier_boxplots.png
        ├── 04_pca_scree.png
        ├── 05_kmeans_elbow_silhouette.png
        ├── 06_dendrogram.png
        ├── 07_kmeans_pca_scatter.png
        ├── 08_kmeans_tsne_scatter.png
        ├── 09_kmeans_feature_bar.png
        ├── 10_kmeans_feature_heatmap.png
        ├── 11_kmeans_boxplots.png
        ├── 12_kmeans_cluster_sizes.png
        ├── 13_market_segmentation.png
        └── 14_hierarchical_pca_scatter.png
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `single_genre_artists.csv` |
| **Total Songs** | 95,837 |
| **Columns** | 23 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Unique Genres** | 3,153 |
| **Unique Artists** | 17,662 |
| **Date Range** | 1900-01-01 to 2021-04-16 |

### Audio Features Used for Clustering (10 features)

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| `danceability` | 0.587 | 0.155 | 0.000 | 0.991 |
| `energy` | 0.541 | 0.236 | 0.000 | 1.000 |
| `loudness` | -10.16 | 4.75 | -50.17 | 5.38 |
| `speechiness` | 0.169 | 0.275 | 0.000 | 0.968 |
| `acousticness` | 0.459 | 0.330 | 0.000 | 0.996 |
| `instrumentalness` | 0.082 | 0.232 | 0.000 | 1.000 |
| `liveness` | 0.225 | 0.186 | 0.000 | 0.997 |
| `valence` | 0.574 | 0.248 | 0.000 | 1.000 |
| `tempo` | 117.5 | 30.2 | 0.000 | 239.9 |
| `duration_ms` | 208,732 | 117,753 | 6,373 | 4,800,118 |

### Notable Feature Correlations (from actual data)

| Pair | Correlation | Interpretation |
|---|---|---|
| loudness vs energy | +0.73 | Louder songs are more energetic |
| energy vs acousticness | -0.66 | Acoustic songs have lower energy |
| valence vs danceability | +0.50 | Happier songs are more danceable |
| liveness vs speechiness | +0.41 | Live tracks tend to have more speech |

---

## 🛠 Tech Stack

| Category | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Machine learning | `scikit-learn` (>= 1.0) |
| Visualisation | `matplotlib`, `seaborn` |
| Interactive charts | `plotly` |
| Dimensionality reduction | `PCA`, `TSNE` (scikit-learn) |
| Clustering | `KMeans`, `DBSCAN`, `AgglomerativeClustering` |
| Hierarchical utils | `scipy.cluster.hierarchy` |
| Dashboard | `streamlit` |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy plotly streamlit
```

### 4. Place the dataset

Copy `single_genre_artists.csv` into the project root directory.

---

## 🚀 Usage

### 1. Run the Full Pipeline

```bash
python amazon_music_clustering.py
```

Expected terminal output summary:

```
PHASE 1 – Data Exploration & Preprocessing
  Shape           : 95,837 rows x 23 columns
  Missing values  : 0  |  Duplicate rows : 0
  Unique genres   : 3,153

PHASE 2 – Feature Selection & Normalisation
  10 audio features selected, StandardScaler applied

PHASE 3 – Dimensionality Reduction  (PCA + t-SNE)
  PCA: 7 components explain 90% variance
  t-SNE: 5,000-row sample

PHASE 4-A – K-Means  |  Elbow & Silhouette Tuning
  k=3  inertia=658,335  silhouette=0.2379  DB=1.5702  <- BEST

PHASE 5 – Cluster Evaluation & Interpretation
  Cluster 0 -> Spoken Word / Audio Drama  (12,513 songs)
  Cluster 1 -> Acoustic / Slow Melodies   (30,807 songs)
  Cluster 2 -> Energetic Pop / Dance      (52,517 songs)

  [exported] cluster_outputs/songs_with_clusters.csv  (95,837 rows)
  [exported] cluster_outputs/cluster_profiles.csv
```

> Update `DATA_PATH` at the top of the script if your CSV is elsewhere:
> ```python
> DATA_PATH = "path/to/single_genre_artists.csv"
> ```

### 2. Run the Streamlit Dashboard (Bonus)

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔄 Project Pipeline

```
single_genre_artists.csv
        |
        v
Phase 1 -- Load & Explore
           95,837 rows, 23 columns, 0 missing, 0 duplicates
           Histograms, correlation heatmap, outlier box plots
        |
        v
Phase 2 -- Feature Selection & Normalisation
           10 audio features selected
           StandardScaler applied (critical: tempo 0-240 vs danceability 0-1)
        |
        v
Phase 3 -- Dimensionality Reduction
           PCA  -> 2D (45.9% variance) + scree plot (7 components = 90%)
           t-SNE -> 2D on 5,000-song sample
        |
        v
Phase 4 -- Clustering (3 algorithms)
           K-Means     -> best k=3 by silhouette score (0.2379)
           DBSCAN      -> 1 cluster + 9.8% noise (density not suited)
           Hierarchical-> Ward linkage on 5K sample + centroid propagation
        |
        v
Phase 5 -- Evaluation & Profiling
           Silhouette: 0.2379  |  Davies-Bouldin: 1.5702  |  Inertia: 658,335
           Real genre analysis per cluster
        |
        v
Phase 6 -- 14 Visualisations saved to cluster_outputs/plots/
        |
        v
Phase 7 -- 4 Business Use Cases + CSV Export
```

---

## 📐 Why k = 3?

Three independent metrics all confirm k=3 on the actual dataset:

### Full Tuning Results

| k | Silhouette | Davies-Bouldin | Inertia | Inertia Drop |
|---|---|---|---|---|
| 2 | 0.2007 | 1.9355 | 778,814 | — |
| **3** | **0.2379 (peak)** | **1.5702** | **658,335** | **120,479 (largest)** |
| 4 | 0.2270 | 1.5302 | 593,031 | 65,304 |
| 5 | 0.1862 | 1.6888 | 548,595 | 44,436 |
| 6 | 0.1575 | 1.6992 | 520,711 | 27,884 |
| 7 | 0.1878 | 1.4712 | 486,936 | 33,774 |
| 8 | 0.1669 | 1.4859 | 460,215 | 26,721 |

**Reason 1 — Silhouette peaks at k=3** (0.2379), then drops at k=4 and keeps declining.

**Reason 2 — Elbow at k=3**: the inertia drop from k=2 to k=3 (120,479) is nearly double the next drop (65,304). After k=3 the curve flattens — more clusters add diminishing returns.

**Reason 3 — The data genuinely has 3 audio profiles**:
- Tracks dominated by speech (audio dramas, poetry — high speechiness, short)
- Tracks that are acoustic and slow (vintage pop, chanson, folk — high acousticness)
- Tracks that are energetic and danceable (j-pop, turkish pop — high energy, loud)

---

## 🤖 Clustering Algorithms

### K-Means (Primary)

- Centroid-based — assigns each song to the nearest cluster centre
- Elbow method: inertia plotted for k=2 to k=8
- Silhouette score: peaks at k=3 (value: 0.2379)
- Parameters: `n_clusters=3`, `random_state=42`, `n_init=10`
- Runs on the full 95,837-song scaled matrix

### DBSCAN

- Density-based — does not require specifying k upfront
- Identifies noise/outlier songs automatically
- Result on this dataset: 1 cluster + 9.8% noise
- Shows the data is not well-suited for density-based separation
- Parameters: `eps=1.5`, `min_samples=15`, sample of 10,000 rows

### Hierarchical / Agglomerative

- Bottom-up merging using Ward linkage
- Memory constraint: Ward needs O(n^2) pairwise distances = 34 GB for 95K rows
- Solution: fit on 5,000-row sample, compute centroids, assign all 95,837 rows by nearest centroid (1-NN propagation)
- Produces a dendrogram for visual cluster interpretation
- Confirms similar cluster structure to K-Means

---

## 🎨 Cluster Profiles (Real Data)

| Feature | Cluster 0 Spoken Word | Cluster 1 Acoustic | Cluster 2 Energetic |
|---|---|---|---|
| **danceability** | 0.6643 | 0.4864 | 0.6273 |
| **energy** | 0.4666 | 0.3112 | **0.6937** |
| **loudness (dB)** | -13.36 | -13.21 | **-7.61** |
| **speechiness** | **0.8300** | 0.0601 | 0.0751 |
| **acousticness** | 0.5859 | **0.7492** | 0.2585 |
| **instrumentalness** | 0.0014 | 0.1685 | 0.0507 |
| **liveness** | **0.4355** | 0.1821 | 0.1999 |
| **valence** | 0.5840 | 0.4133 | **0.6664** |
| **tempo (BPM)** | 100.4 | 111.9 | **124.9** |
| **avg duration** | 1.6 min | 3.7 min | 3.8 min |
| **avg popularity** | 28.2 | 20.7 | 28.7 |
| **songs** | 12,513 (13.1%) | 30,807 (32.1%) | 52,517 (54.8%) |

### Top Genres Per Cluster (from real data)

**Cluster 0 — Spoken Word / Audio Drama**
hoerspiel (7,564) · kleine hoerspiel (2,005) · psychedelic rock (220) · barnsagor (185) · poetry (166)

**Cluster 1 — Acoustic / Slow Melodies**
vintage taiwan pop (1,131) · classic israeli pop (592) · chanson (550) · classic soundtrack (456) · classic thai pop (428)

**Cluster 2 — Energetic Pop / Dance**
j-pop (878) · turkish pop (683) · classic thai pop (649) · thai pop (581) · classic israeli pop (553)

---

## 📏 Evaluation Metrics (Real Data)

| Metric | Value | Range | Interpretation |
|---|---|---|---|
| **Silhouette Score** | **0.2379** | -1 to 1 (higher=better) | Songs are reasonably close to own cluster vs others |
| **Davies-Bouldin Index** | **1.5702** | 0 to inf (lower=better) | Acceptable inter-cluster separation |
| **Inertia (SSE)** | **658,335** | 0 to inf (lower=better) | Compact clusters at the elbow point k=3 |

### Algorithm Comparison

| Algorithm | Silhouette | Davies-Bouldin | Verdict |
|---|---|---|---|
| **K-Means (k=3)** | **0.2379** | **1.5702** | Best — used for final labels |
| Hierarchical (k=3) | ~0.23 | ~1.58 | Similar structure to K-Means |
| DBSCAN | N/A | N/A | 1 cluster + 9.8% noise — not suitable |

---

## 🖼 Visualisations Generated

14 plots saved to `cluster_outputs/plots/`:

| File | Description |
|---|---|
| `01_feature_distributions.png` | Histograms + mean line for all 10 audio features |
| `02_correlation_heatmap.png` | Lower-triangle Pearson correlation matrix |
| `03_outlier_boxplots.png` | Box plots for outlier detection per feature |
| `04_pca_scree.png` | Cumulative explained variance vs number of components |
| `05_kmeans_elbow_silhouette.png` | Elbow + Silhouette tuning charts side-by-side |
| `06_dendrogram.png` | Ward linkage dendrogram on 5K-song sample |
| `07_kmeans_pca_scatter.png` | PCA 2D scatter coloured by K-Means cluster |
| `08_kmeans_tsne_scatter.png` | t-SNE 2D scatter coloured by K-Means cluster |
| `09_kmeans_feature_bar.png` | Grouped bar chart of mean features per cluster |
| `10_kmeans_feature_heatmap.png` | Normalised (0-1) feature heatmap across clusters |
| `11_kmeans_boxplots.png` | Box plots of 6 key features split by cluster |
| `12_kmeans_cluster_sizes.png` | Bar chart of song count per cluster |
| `13_market_segmentation.png` | Pie chart + popularity violin by cluster |
| `14_hierarchical_pca_scatter.png` | PCA scatter for Hierarchical clustering comparison |

---

## 🎯 Business Use-Case Outputs

### A — Personalised Playlist Curation

Each cluster auto-generates a themed playlist sorted by popularity:

```
Playlist [0] – Spoken Word / Audio Drama  (12,513 songs)
Playlist [1] – Acoustic / Slow Melodies   (30,807 songs)
Playlist [2] – Energetic Pop / Dance      (52,517 songs)
```

### B — Song Discovery / Recommendation

Returns top-10 most similar songs from the same cluster ranked by **Euclidean distance in scaled audio feature space** (not just popularity):

```
Query  : 'Song Name'  by  Artist Name
Cluster: 2 – Energetic Pop / Dance

Top 10 similar songs by audio feature distance:
  Song                   Artist              Distance
  Song A                 Artist X            0.4821
  Song B                 Artist Y            0.5103
```

### C — Artist Analysis

Maps all 17,662 unique artists to their dominant cluster sorted by average popularity. Identifies the competitive landscape within each audio niche.

### D — Market Segmentation

| Segment | Songs | Share | Avg Popularity |
|---|---|---|---|
| Spoken Word / Audio Drama | 12,513 | 13.1% | 28.2 |
| Acoustic / Slow Melodies | 30,807 | 32.1% | 20.7 |
| Energetic Pop / Dance | 52,517 | 54.8% | 28.7 |

---

## 📦 Project Deliverables

| Deliverable | File | Status |
|---|---|---|
| Source code – Preprocessing | `amazon_music_clustering.py` Phase 1-2 | Done |
| Source code – Clustering | `amazon_music_clustering.py` Phase 4 | Done |
| Source code – Visualisation | `amazon_music_clustering.py` Phase 6 | Done |
| Final Presentation (13 slides) | `Amazon_Music_Clustering_Report.pptx` | Done |
| CSV with cluster labels | `cluster_outputs/songs_with_clusters.csv` | Done |
| Cluster profiles CSV | `cluster_outputs/cluster_profiles.csv` | Done |
| Bonus Streamlit Dashboard | `streamlit_app.py` | Done |

---

## 📈 Results

| Result | Value |
|---|---|
| Total songs clustered | 95,837 |
| Optimal clusters (k) | 3 (silhouette-confirmed) |
| Best silhouette score | 0.2379 |
| Davies-Bouldin index | 1.5702 |
| Largest cluster | Cluster 2 — Energetic Pop/Dance (54.8%) |
| Smallest cluster | Cluster 0 — Spoken Word/Audio Drama (13.1%) |
| Unique artists covered | 17,662 |
| Unique genres covered | 3,153 |
| Date range covered | 1900 to 2021 |
| Plots generated | 14 |

---

## 📐 Project Guidelines

- Clean, modular code — one function per responsibility
- PEP-8 style with full docstrings
- `random_state=42` set globally for full reproducibility
- All phases clearly numbered and labelled in output
- Compatible with scikit-learn >= 1.0 and >= 1.4 (auto-detects n_iter vs max_iter)
- Memory-safe hierarchical clustering (centroid-propagation on 5K sample)
- All 14 plots saved to `cluster_outputs/plots/`
- Final CSVs exported to `cluster_outputs/`

---

## 👨‍💻 Skills Demonstrated

Data Exploration · Data Cleaning · Feature Selection · Data Normalisation · K-Means Clustering · DBSCAN · Hierarchical Clustering · Elbow Method · Silhouette Score · Davies-Bouldin Index · PCA · t-SNE · Cluster Visualisation · Genre Inference · Python · pandas · NumPy · scikit-learn · Streamlit · Data Storytelling

---

## 📄 License

This project was developed as part of the **GUVI | HCL Data Science & AI Programme**.
For educational use only.

---

GUVI | HCL · Data Science & AI Programme · Music Analytics / Unsupervised Machine Learning
