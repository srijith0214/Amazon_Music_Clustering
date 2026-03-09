"""
============================================================
 Amazon Music Clustering – Streamlit Dashboard (Bonus)
 GUVI | HCL  ·  Unsupervised Machine Learning
============================================================

Run:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn scipy plotly
    streamlit run streamlit_app.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F7FB; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-left: 4px solid #1565C0;
    }
    .metric-val  { font-size: 2.2rem; font-weight: 700; color: #1565C0; }
    .metric-lbl  { font-size: 0.85rem; color: #64748B; margin-top: 2px; }
    .cluster-card {
        background: white; border-radius: 10px; padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.07); margin-bottom: 0.8rem;
    }
    .section-header {
        font-size: 1.25rem; font-weight: 700; color: #1A2340;
        border-bottom: 2px solid #1565C0; padding-bottom: 6px; margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
    div[data-testid="stSidebarContent"] { background-color: #0A1628; }
    div[data-testid="stSidebarContent"] * { color: #E0EAF8 !important; }
    div[data-testid="stSidebarContent"] .stSelectbox label,
    div[data-testid="stSidebarContent"] .stSlider label { color: #A8C0D8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
]
CLUSTER_COLORS = px.colors.qualitative.Bold
MOOD_MAP = {
    0: "⚡ High-Energy Party",
    1: "🎸 Chill Acoustic",
    2: "🎹 Instrumental",
    3: "🎤 Vocal / Spoken",
    4: "😊 Upbeat & Happy",
}


# ─────────────────────────────────────────────
#  DATA LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def get_scaled(df: pd.DataFrame):
    X = df[AUDIO_FEATURES].fillna(0)
    scaler = StandardScaler()
    return scaler.fit_transform(X)


@st.cache_data(show_spinner=False)
def get_pca(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled), pca


@st.cache_data(show_spinner=False)
def run_kmeans(X_scaled, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X_scaled)


@st.cache_data(show_spinner=False)
def run_dbscan(X_scaled, eps: float, min_samples: int):
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    return db.fit_predict(X_scaled)


@st.cache_data(show_spinner=False)
def run_hierarchical(X_scaled, k: int):
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return ag.fit_predict(X_scaled)


@st.cache_data(show_spinner=False)
def elbow_data(X_scaled):
    inertias, sil_scores = [], []
    ks = list(range(2, 11))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, lbl, sample_size=5000))
    return ks, inertias, sil_scores


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Amazon Music\n### Clustering Dashboard")
    st.markdown("---")

    data_path = st.text_input("CSV path", value="single_genre_artists.csv")

    st.markdown("### Algorithm")
    algo = st.selectbox("Clustering method", ["K-Means", "DBSCAN", "Hierarchical"])

    if algo == "K-Means":
        k = st.slider("Number of clusters (k)", 2, 10, 5)
    elif algo == "DBSCAN":
        eps = st.slider("eps", 0.3, 3.0, 1.5, 0.1)
        min_samples = st.slider("min_samples", 5, 50, 15)
        k = None
    else:
        k = st.slider("Number of clusters (k)", 2, 10, 5)

    st.markdown("---")
    st.markdown("### Visualisation")
    show_tsne = st.checkbox("Show t-SNE (slow)", False)
    color_feature = st.selectbox("Color scatter by", ["cluster"] + AUDIO_FEATURES)

    st.markdown("---")
    st.markdown("**GUVI | HCL**\nUnsupervised ML Project")


# ─────────────────────────────────────────────
#  MAIN – Load data
# ─────────────────────────────────────────────
st.markdown("# 🎵 Amazon Music Clustering Dashboard")
st.markdown("Unsupervised machine learning on 95,837 songs · K-Means · DBSCAN · Hierarchical · PCA · t-SNE")
st.markdown("---")

try:
    df = load_data(data_path)
except FileNotFoundError:
    st.error(f"Dataset not found at **{data_path}**. Update the path in the sidebar.")
    st.stop()

X_scaled = get_scaled(df)
X_pca, pca_model = get_pca(X_scaled)

# Run selected algorithm
if algo == "K-Means":
    labels = run_kmeans(X_scaled, k)
elif algo == "DBSCAN":
    labels = run_dbscan(X_scaled, eps, min_samples)
    k = len(set(labels)) - (1 if -1 in labels else 0)
else:
    labels = run_hierarchical(X_scaled, k)

df_c = df.copy()
df_c["cluster"] = labels

# ─────────────────────────────────────────────
#  METRIC CARDS
# ─────────────────────────────────────────────
valid_mask = labels != -1
n_valid = valid_mask.sum()
sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask], sample_size=min(5000, n_valid)) if n_valid > 1 else 0
db  = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]) if n_valid > 1 else 0
noise_pct = (labels == -1).mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in [
    (c1, f"{len(df):,}", "Total Songs"),
    (c2, str(len(set(labels)) - (1 if -1 in labels else 0)), "Clusters Found"),
    (c3, f"{sil:.3f}", "Silhouette Score"),
    (c4, f"{db:.3f}", "Davies-Bouldin"),
    (c5, f"{noise_pct:.1f}%", "Noise Points"),
]:
    col.markdown(f"""<div class="metric-card">
        <div class="metric-val">{val}</div>
        <div class="metric-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 EDA", "🔬 Tuning", "🗺️ Clusters", "📈 Profiles",
    "🎯 Use Cases", "📤 Export"
])


# ══════════════════════════════════════════════
#  TAB 1 – EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Distributions**")
        feat_sel = st.selectbox("Select feature", AUDIO_FEATURES, key="eda_feat")
        fig = px.histogram(df, x=feat_sel, nbins=60, color_discrete_sequence=["#1565C0"],
                           title=f"Distribution of {feat_sel}")
        fig.update_layout(bargap=0.05, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Correlation Heatmap**")
        corr = df[AUDIO_FEATURES].corr()
        fig_h, ax_h = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    linewidths=0.4, ax=ax_h, annot_kws={"size": 8})
        ax_h.set_title("Audio Feature Correlation", fontsize=12, fontweight="bold")
        st.pyplot(fig_h, use_container_width=True)
        plt.close()

    st.markdown("**Pairplot – Key Features**")
    pair_feats = ["danceability", "energy", "acousticness", "valence"]
    fig_pair = px.scatter_matrix(df.sample(min(3000, len(df))), dimensions=pair_feats,
                                  color_discrete_sequence=["#1565C0"],
                                  title="Scatter Matrix – Key Audio Features")
    fig_pair.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.5))
    fig_pair.update_layout(height=500)
    st.plotly_chart(fig_pair, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 – TUNING
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">K-Means Hyperparameter Tuning</div>', unsafe_allow_html=True)

    with st.spinner("Computing elbow & silhouette…"):
        ks, inertias, sil_scores = elbow_data(X_scaled)

    col1, col2 = st.columns(2)
    with col1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                                    marker=dict(color="#E53935", size=8),
                                    line=dict(color="#E53935", width=3)))
        fig_e.update_layout(title="Elbow Method – Inertia vs k",
                             xaxis_title="Number of Clusters (k)",
                             yaxis_title="Inertia (SSE)",
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_e, use_container_width=True)

    with col2:
        best_k_idx = int(np.argmax(sil_scores))
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=ks, y=sil_scores, mode="lines+markers",
                                    marker=dict(color="#43A047", size=8),
                                    line=dict(color="#43A047", width=3)))
        fig_s.add_vline(x=ks[best_k_idx], line_dash="dash", line_color="#1565C0",
                        annotation_text=f"Best k={ks[best_k_idx]}")
        fig_s.update_layout(title="Silhouette Score vs k",
                             xaxis_title="k", yaxis_title="Silhouette Score",
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_s, use_container_width=True)

    st.info(f"💡 Recommended k = **{ks[best_k_idx]}** based on maximum silhouette score ({sil_scores[best_k_idx]:.4f})")

    # PCA scree
    st.markdown("**PCA Scree Plot**")
    pca_full = PCA(random_state=42).fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    fig_scree = px.line(x=list(range(1, len(cum_var)+1)), y=cum_var,
                         labels={"x": "Components", "y": "Cumulative Variance"},
                         title="PCA Cumulative Explained Variance",
                         markers=True, color_discrete_sequence=["#1565C0"])
    fig_scree.add_hline(y=0.90, line_dash="dash", line_color="red",
                        annotation_text="90% threshold")
    fig_scree.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_scree, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 – CLUSTER SCATTER
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Cluster Visualisation (PCA 2D)</div>', unsafe_allow_html=True)

    sample_n = st.slider("Sample size for scatter", 1000, min(20000, len(df)), 8000, 1000)
    idx = np.random.choice(len(df_c), sample_n, replace=False)
    df_samp = df_c.iloc[idx].copy()
    df_samp["PC1"] = X_pca[idx, 0]
    df_samp["PC2"] = X_pca[idx, 1]

    if color_feature == "cluster":
        df_samp["color_col"] = df_samp["cluster"].astype(str)
        color_arg = "color_col"
    else:
        color_arg = color_feature

    name_col   = "name_song"   if "name_song"   in df_samp.columns else "track_name"
    artist_col = "name_artists" if "name_artists" in df_samp.columns else "artist_name"

    fig_scatter = px.scatter(
        df_samp, x="PC1", y="PC2", color=color_arg,
        hover_data=[name_col, artist_col, "cluster"] if name_col in df_samp.columns else ["cluster"],
        title=f"PCA Cluster Scatter – {algo} (n={sample_n:,})",
        color_discrete_sequence=CLUSTER_COLORS,
        opacity=0.65,
    )
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", height=520,
        legend_title_text="Cluster"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Cluster size bar
    sizes = df_c["cluster"].value_counts().sort_index().reset_index()
    sizes.columns = ["Cluster", "Songs"]
    fig_sz = px.bar(sizes, x="Cluster", y="Songs", color="Cluster",
                    title="Songs per Cluster", color_discrete_sequence=CLUSTER_COLORS,
                    text="Songs")
    fig_sz.update_traces(textposition="outside")
    fig_sz.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_sz, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 4 – PROFILES
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Cluster Profiles & Feature Analysis</div>', unsafe_allow_html=True)

    profile = df_c.groupby("cluster")[AUDIO_FEATURES].mean().round(4)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Mean Feature Values per Cluster**")
        display_feats = [f for f in AUDIO_FEATURES if f != "duration_ms"]
        fig_bar = px.bar(
            profile[display_feats].T.reset_index().melt(id_vars="index"),
            x="index", y="value", color="cluster",
            barmode="group", color_discrete_sequence=CLUSTER_COLORS,
            labels={"index": "Feature", "value": "Mean Value", "cluster": "Cluster"},
            title="Average Audio Features by Cluster",
        )
        fig_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("**Feature Heatmap (Normalised)**")
        data_norm = profile[display_feats].copy()
        for col in data_norm.columns:
            rng = data_norm[col].max() - data_norm[col].min()
            data_norm[col] = (data_norm[col] - data_norm[col].min()) / (rng + 1e-9)
        fig_heat = px.imshow(
            data_norm, color_continuous_scale="YlOrRd",
            title="Normalised Feature Heatmap",
            labels=dict(color="Normalised\nMean"),
            text_auto=".2f",
        )
        fig_heat.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**Box Plots – Feature by Cluster**")
    box_feat = st.selectbox("Feature", display_feats, key="box_feat")
    fig_box = px.box(df_c, x="cluster", y=box_feat, color="cluster",
                     color_discrete_sequence=CLUSTER_COLORS,
                     title=f"{box_feat} distribution per cluster",
                     notched=True)
    fig_box.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("**Cluster Profiles Table**")
    st.dataframe(profile.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 – BUSINESS USE CASES
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Business Use-Case Outcomes</div>', unsafe_allow_html=True)

    uc_tab = st.tabs(["🎵 Playlist Curation", "🔍 Song Discovery",
                       "🎤 Artist Analysis", "📊 Market Segmentation"])

    name_col   = "name_song"   if "name_song"   in df_c.columns else "track_name"
    artist_col = "name_artists" if "name_artists" in df_c.columns else "artist_name"
    pop_col    = "popularity_songs" if "popularity_songs" in df_c.columns else None

    # ── Playlist Curation ──────────────────────────────────────
    with uc_tab[0]:
        st.markdown("#### Auto-Generated Playlists by Cluster")
        for cid in sorted(df_c["cluster"].unique()):
            if cid == -1:
                continue
            songs = df_c[df_c["cluster"] == cid]
            if pop_col:
                songs = songs.sort_values(pop_col, ascending=False)
            mood = MOOD_MAP.get(cid, f"Cluster {cid}")
            with st.expander(f"Playlist {cid} – {mood}  ({len(songs):,} songs)"):
                cols_show = [c for c in [name_col, artist_col, pop_col] if c and c in songs.columns]
                st.dataframe(songs[cols_show].head(20).reset_index(drop=True), use_container_width=True)

    # ── Song Discovery ─────────────────────────────────────────
    with uc_tab[1]:
        st.markdown("#### Find Similar Songs")
        if name_col in df_c.columns:
            song_names = df_c[name_col].dropna().unique()[:3000]
            selected = st.selectbox("Select a song", song_names)
            n_recs = st.slider("Recommendations", 5, 25, 10)
            row = df_c[df_c[name_col] == selected].iloc[0]
            cid = row["cluster"]
            same_cluster = df_c[(df_c["cluster"] == cid) & (df_c[name_col] != selected)]
            if pop_col:
                same_cluster = same_cluster.sort_values(pop_col, ascending=False)
            recs = same_cluster.head(n_recs)

            st.info(f"🎵 **{selected}** → Cluster {cid} ({MOOD_MAP.get(cid, '')})")
            cols_show = [c for c in [name_col, artist_col, pop_col] + AUDIO_FEATURES[:4] if c and c in recs.columns]
            st.dataframe(recs[cols_show].reset_index(drop=True), use_container_width=True)
        else:
            st.warning("No song name column found.")

    # ── Artist Analysis ─────────────────────────────────────────
    with uc_tab[2]:
        st.markdown("#### Artist → Cluster Mapping")
        if artist_col in df_c.columns:
            artist_cluster = df_c.groupby(artist_col)["cluster"].agg(
                lambda x: x.value_counts().idxmax()
            ).reset_index()
            artist_cluster.columns = [artist_col, "primary_cluster"]
            if pop_col:
                avg_pop = df_c.groupby(artist_col)[pop_col].mean().round(1).reset_index()
                artist_cluster = artist_cluster.merge(avg_pop, on=artist_col)
                artist_cluster = artist_cluster.sort_values(pop_col, ascending=False)

            fig_ac = px.bar(
                artist_cluster.head(30), x=artist_col, y=pop_col if pop_col else artist_col,
                color="primary_cluster", color_discrete_sequence=CLUSTER_COLORS,
                title="Top 30 Artists by Popularity & Their Cluster",
                labels={artist_col: "Artist", pop_col: "Avg Popularity", "primary_cluster": "Cluster"},
            )
            fig_ac.update_layout(xaxis_tickangle=-45, plot_bgcolor="white",
                                   paper_bgcolor="white", height=480)
            st.plotly_chart(fig_ac, use_container_width=True)

            cluster_filter = st.selectbox("Filter by cluster", ["All"] + sorted(artist_cluster["primary_cluster"].unique().tolist()))
            display_df = artist_cluster if cluster_filter == "All" else artist_cluster[artist_cluster["primary_cluster"] == int(cluster_filter)]
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
        else:
            st.warning("No artist column found.")

    # ── Market Segmentation ─────────────────────────────────────
    with uc_tab[3]:
        st.markdown("#### Catalogue Market Segmentation")
        sizes = df_c[df_c["cluster"] != -1]["cluster"].value_counts().sort_index()
        labels_pie = [MOOD_MAP.get(i, f"Cluster {i}") for i in sizes.index]

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=sizes.values, names=labels_pie,
                              title="Catalogue Share by Cluster",
                              color_discrete_sequence=CLUSTER_COLORS, hole=0.35)
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if pop_col:
                fig_pop = px.violin(df_c[df_c["cluster"] != -1], x="cluster", y=pop_col,
                                     color="cluster", color_discrete_sequence=CLUSTER_COLORS,
                                     title="Popularity Distribution by Cluster",
                                     box=True, points=False)
                fig_pop.update_layout(showlegend=False, plot_bgcolor="white",
                                       paper_bgcolor="white")
                st.plotly_chart(fig_pop, use_container_width=True)

        # Segment report
        st.markdown("**Segment Summary**")
        profile_seg = df_c.groupby("cluster")[AUDIO_FEATURES].mean().round(3)
        seg_rows = []
        for cid, cnt in sizes.items():
            row = profile_seg.loc[cid]
            seg_rows.append({
                "Cluster": cid,
                "Mood": MOOD_MAP.get(cid, f"Cluster {cid}"),
                "Songs": f"{cnt:,}",
                "Share": f"{cnt/len(df_c)*100:.1f}%",
                "Top Feature": row[AUDIO_FEATURES].idxmax(),
                "Avg Energy": f"{row['energy']:.3f}",
                "Avg Danceability": f"{row['danceability']:.3f}",
                "Avg Valence": f"{row['valence']:.3f}",
            })
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 6 – EXPORT
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Songs with Cluster Labels**")
        st.markdown(f"Total rows: **{len(df_c):,}**")
        st.dataframe(df_c.head(50), use_container_width=True)
        csv_songs = df_c.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download songs_with_clusters.csv",
            data=csv_songs,
            file_name="songs_with_clusters.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("**Cluster Profiles**")
        profile_export = df_c.groupby("cluster")[AUDIO_FEATURES].mean().round(4)
        profile_export["mood"] = [MOOD_MAP.get(i, f"Cluster {i}") for i in profile_export.index]
        st.dataframe(profile_export, use_container_width=True)
        csv_profile = profile_export.to_csv().encode("utf-8")
        st.download_button(
            label="⬇️ Download cluster_profiles.csv",
            data=csv_profile,
            file_name="cluster_profiles.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("**Evaluation Summary**")
    summary = {
        "Algorithm": algo,
        "Clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "Silhouette Score": round(sil, 4),
        "Davies-Bouldin Index": round(db, 4),
        "Noise Points (%)": round(noise_pct, 2),
        "Total Songs": len(df_c),
    }
    st.json(summary)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748B;font-size:0.85rem;'>"
    "Amazon Music Clustering Dashboard &nbsp;·&nbsp; GUVI | HCL &nbsp;·&nbsp; "
    "Unsupervised Machine Learning Project"
    "</div>",
    unsafe_allow_html=True
)
