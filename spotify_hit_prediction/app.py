import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, classification_report
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLOR_MAP = {0: "#636EFA", 1: "#EF553B"}
LABEL_MAP = {0: "Not a Hit", 1: "Hit"}

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]
CATEGORICAL_FEATURES = ["explicit", "key", "mode", "time_signature", "track_genre"]
RADAR_FEATURES = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]

MODEL_NAMES = ["logistic_regression", "random_forest", "knn", "naive_bayes"]

# ---------------------------------------------------------------------------
# Data & model loading (cached so they only run once)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/dataset.csv")
    df = df[df["popularity"] > 0].reset_index(drop=True)
    df["is_hit"] = (df["popularity"] >= 70).astype(int)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    X = df[AUDIO_FEATURES + CATEGORICAL_FEATURES]
    y = df["is_hit"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    df_test = X_test.copy()
    df_test["is_hit"] = y_test.values
    return df, df_test


@st.cache_resource
def load_models():
    return {name: joblib.load(f"models/{name}.joblib") for name in MODEL_NAMES}


# ---------------------------------------------------------------------------
# Helper: predict with correct features per model
# ---------------------------------------------------------------------------
def predict(model_name, pipeline, X):
    if model_name == "naive_bayes":
        return pipeline.predict(X[AUDIO_FEATURES]), pipeline.predict_proba(X[AUDIO_FEATURES])[:, 1]
    return pipeline.predict(X), pipeline.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def make_radar(df):
    mins = {f: df[f].min() for f in RADAR_FEATURES}
    maxs = {f: df[f].max() for f in RADAR_FEATURES}

    def normalise(row):
        return [(row[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in RADAR_FEATURES]

    fig = go.Figure()
    for label, display in LABEL_MAP.items():
        means = df[df["is_hit"] == label][RADAR_FEATURES].mean()
        normed = [(means[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in RADAR_FEATURES]
        fig.add_trace(go.Scatterpolar(
            r=normed + [normed[0]],
            theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
            fill="toself",
            name=display,
            line_color=COLOR_MAP[label],
            opacity=0.7,
        ))
    fig.update_layout(
        title="Audio DNA: Hits vs Non-Hits",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h"),
    )
    return fig


def make_confusion_matrix(y_test, y_pred, normalise=False):
    cm = confusion_matrix(y_test, y_pred)
    if normalise:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"
    labels = ["Not a Hit", "Hit"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=[[f"{v:{fmt}}" for v in row] for row in cm],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def make_roc_curves(df_test, models):
    X = df_test.drop(columns=["is_hit"])
    y = df_test["is_hit"]
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash", color="grey", width=1))
    for name, pipeline in models.items():
        _, y_proba = predict(name, pipeline, X)
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC={auc:.3f})"))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def make_pr_curves(df_test, models):
    X = df_test.drop(columns=["is_hit"])
    y = df_test["is_hit"]
    baseline = y.mean()
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=baseline, x1=1, y1=baseline,
                  line=dict(dash="dash", color="grey", width=1))
    for name, pipeline in models.items():
        _, y_proba = predict(name, pipeline, X)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        ap = average_precision_score(y, y_proba)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                                 name=f"{name} (AP={ap:.3f})"))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def make_feature_importance(model_name, pipeline, feature_names):
    step_name = "clasifier" if "clasifier" in dict(pipeline.steps) else "classifier"
    clf = pipeline.named_steps[step_name]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = "Feature Importances (Random Forest)"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        title = "Coefficient Magnitudes (Logistic Regression)"
    else:
        return None, None
    return importances, title


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Musical DNA", page_icon="🎵", layout="wide")
st.title("🎵 Musical DNA: Spotify Hit Predictor")
st.caption("Predict whether a song's audio fingerprint signals chart success.")

df, df_test = load_data()
models = load_models()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Model", MODEL_NAMES, index=1)
threshold = st.sidebar.slider("Hit threshold (popularity)", 50, 90, 70, step=5)
selected_genre = st.sidebar.selectbox(
    "Filter genre (EDA tab)", ["All"] + sorted(df["track_genre"].unique().tolist())
)

# Re-apply threshold if changed
df["is_hit"] = (df["popularity"] >= threshold).astype(int)
df_filtered = df if selected_genre == "All" else df[df["track_genre"] == selected_genre]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_eda, tab_results, tab_predictor = st.tabs(["📊 EDA", "🏆 Model Results", "🎛️ Hit Predictor"])

# ── EDA Tab ─────────────────────────────────────────────────────────────────
with tab_eda:
    hit_rate = df_filtered["is_hit"].mean() * 100
    top_genre = (df_filtered.groupby("track_genre")["is_hit"].mean()
                 .sort_values(ascending=False).index[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tracks", f"{len(df_filtered):,}")
    col2.metric("Hit Rate", f"{hit_rate:.1f}%")
    col3.metric("Top Genre by Hit Rate", top_genre)

    st.plotly_chart(make_radar(df_filtered), width='stretch')

    col_left, col_right = st.columns(2)
    with col_left:
        fig = px.histogram(df_filtered, x="popularity", nbins=50,
                           color_discrete_sequence=["#636EFA"],
                           title="Popularity Distribution",
                           labels={"popularity": "Popularity Score"})
        st.plotly_chart(fig, width='stretch')

    with col_right:
        genre_hit_rate = (df_filtered.groupby("track_genre")["is_hit"]
                          .mean().sort_values(ascending=False).head(10).reset_index())
        genre_hit_rate.columns = ["Genre", "Hit Rate"]
        fig = px.bar(genre_hit_rate, x="Hit Rate", y="Genre", orientation="h",
                     color="Hit Rate", color_continuous_scale="Blues",
                     title="Top 10 Genres by Hit Rate")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Audio Feature Distributions by Class")
    feature_select = st.selectbox("Feature", AUDIO_FEATURES)
    fig = px.box(df_filtered.sample(min(10000, len(df_filtered)), random_state=42),
                 x="is_hit", y=feature_select,
                 color="is_hit", color_discrete_map=COLOR_MAP,
                 labels={"is_hit": "Hit", feature_select: feature_select},
                 title=f"{feature_select} by Hit / Not Hit")
    fig.update_layout(xaxis=dict(ticktext=["Not a Hit", "Hit"], tickvals=[0, 1]))
    st.plotly_chart(fig, width='stretch')

# ── Model Results Tab ────────────────────────────────────────────────────────
with tab_results:
    pipeline = models[selected_model]
    X_t = df_test.drop(columns=["is_hit"])
    y_t = df_test["is_hit"]
    y_pred, y_proba = predict(selected_model, pipeline, X_t)

    report = classification_report(y_t, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_t, y_proba)
    pr_auc  = average_precision_score(y_t, y_proba)
    f1_hit  = report["1"]["f1-score"]

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC",  f"{roc_auc:.4f}")
    col2.metric("PR-AUC",   f"{pr_auc:.4f}")
    col3.metric("F1 (Hit)", f"{f1_hit:.4f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Confusion Matrix (counts)")
        st.plotly_chart(make_confusion_matrix(y_t, y_pred, normalise=False),
                        width='stretch')
    with col_right:
        st.subheader("Confusion Matrix (normalised)")
        st.plotly_chart(make_confusion_matrix(y_t, y_pred, normalise=True),
                        width='stretch')

    st.plotly_chart(make_roc_curves(df_test, models), width='stretch')
    st.plotly_chart(make_pr_curves(df_test, models), width='stretch')

    importances, imp_title = make_feature_importance(selected_model, pipeline, AUDIO_FEATURES)
    if importances is not None:
        # Get feature names after preprocessing
        try:
            pre = pipeline.named_steps["preprocessor"]
            feat_names = pre.get_feature_names_out()
        except Exception:
            feat_names = [f"feature_{i}" for i in range(len(importances))]
        top_n = 20
        idx = np.argsort(importances)[-top_n:]
        fig = px.bar(x=importances[idx], y=feat_names[idx], orientation="h",
                     title=imp_title,
                     labels={"x": "Importance", "y": "Feature"})
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Feature importance not available for the selected model.")

# ── Hit Predictor Tab ────────────────────────────────────────────────────────
@st.fragment
def predictor_section(df, models, selected_model):
    st.subheader("Enter a song's audio features")

    col1, col2, col3 = st.columns(3)
    with col1:
        danceability    = st.slider("Danceability",      0.0, 1.0, 0.5, 0.01)
        energy          = st.slider("Energy",            0.0, 1.0, 0.5, 0.01)
        loudness        = st.slider("Loudness (dB)",   -60.0, 0.0, -8.0, 0.5)
        speechiness     = st.slider("Speechiness",      0.0, 1.0, 0.05, 0.01)
    with col2:
        acousticness    = st.slider("Acousticness",     0.0, 1.0, 0.2, 0.01)
        instrumentalness= st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness        = st.slider("Liveness",         0.0, 1.0, 0.15, 0.01)
        valence         = st.slider("Valence",          0.0, 1.0, 0.5, 0.01)
    with col3:
        tempo           = st.slider("Tempo (BPM)",      60, 200, 120, 1)
        duration_ms     = st.number_input("Duration (ms)", value=210000, step=1000)
        explicit        = st.selectbox("Explicit", ["False", "True"])
        key             = st.selectbox("Key", list(range(12)))
        mode            = st.selectbox("Mode", [0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
        time_sig        = st.selectbox("Time Signature", [3, 4, 5])
        genre           = st.selectbox("Genre", sorted(df["track_genre"].unique().tolist()))

    if st.button("🎵 Predict Hit Probability", type="primary"):
        input_df = pd.DataFrame([{
            "danceability": danceability, "energy": energy,
            "loudness": loudness, "speechiness": speechiness,
            "acousticness": acousticness, "instrumentalness": instrumentalness,
            "liveness": liveness, "valence": valence,
            "tempo": float(tempo), "duration_ms": float(duration_ms),
            "explicit": str(explicit), "key": str(key),
            "mode": str(mode), "time_signature": str(time_sig),
            "track_genre": genre,
        }])

        pipeline = models[selected_model]
        _, proba = predict(selected_model, pipeline, input_df)
        prob = float(proba[0])
        verdict = "🔥 Hit" if prob >= 0.5 else "📉 Not a Hit"

        col_m, col_g = st.columns([1, 2])
        with col_m:
            st.metric("Hit Probability", f"{prob:.1%}", delta=verdict)

        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLOR_MAP[1] if prob >= 0.5 else COLOR_MAP[0]},
                    "steps": [
                        {"range": [0, 50],  "color": "#f0f0f0"},
                        {"range": [50, 100], "color": "#ffe0e0"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                },
                title={"text": "Hit Probability"},
            ))
            st.plotly_chart(fig, width='stretch')

        st.subheader("Your song vs. average hit profile")
        mins = {f: df[f].min() for f in RADAR_FEATURES}
        maxs = {f: df[f].max() for f in RADAR_FEATURES}

        user_vals = {
            "danceability": danceability, "energy": energy,
            "speechiness": speechiness, "acousticness": acousticness,
            "instrumentalness": instrumentalness, "liveness": liveness,
            "valence": valence,
        }
        user_normed  = [(user_vals[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in RADAR_FEATURES]
        hit_means    = df[df["is_hit"] == 1][RADAR_FEATURES].mean()
        hit_normed   = [(hit_means[f] - mins[f]) / (maxs[f] - mins[f] + 1e-9) for f in RADAR_FEATURES]

        fig = go.Figure()
        for vals, name_r, color in [
            (hit_normed,  "Avg Hit",    COLOR_MAP[1]),
            (user_normed, "Your Song",  "#00CC96"),
        ]:
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=RADAR_FEATURES + [RADAR_FEATURES[0]],
                fill="toself", name=name_r,
                line_color=color, opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Set the audio features above and click Predict.")


with tab_predictor:
    predictor_section(df, models, selected_model)
