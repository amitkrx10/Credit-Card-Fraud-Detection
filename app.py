# ── Imports ──────────────────────────────────────────────────────────────────
import io
import time as time_module
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)

warnings.filterwarnings("ignore")

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="FraudGuard — ML Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# THEME & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "bg": "#0d1117",
    "surface": "#161b22",
    "surface2": "#21262d",
    "border": "#30363d",
    "accent": "#238636",          # green – safe / normal
    "danger": "#da3633",          # red – fraud
    "warning": "#d29922",         # amber
    "info": "#1f6feb",            # blue
    "text": "#e6edf3",
    "muted": "#8b949e",
    "chart1": "#58a6ff",
    "chart2": "#f78166",
    "chart3": "#3fb950",
    "chart4": "#d2a8ff",
}

st.markdown(
    f"""
    <style>
    /* ── Reset & base ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {PALETTE['surface']};
        border-right: 1px solid {PALETTE['border']};
    }}
    /* ── Headings ── */
    h1 {{ font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }}
    h2 {{ font-size: 1.4rem !important; font-weight: 600 !important; }}
    h3 {{ font-size: 1.1rem !important; font-weight: 500 !important; color: {PALETTE['muted']}; }}
    /* ── Metric cards ── */
    [data-testid="metric-container"] {{
        background: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 10px;
        padding: 14px 18px !important;
    }}
    [data-testid="stMetricLabel"] {{ color: {PALETTE['muted']} !important; font-size: .78rem !important; text-transform: uppercase; letter-spacing: .06em; }}
    [data-testid="stMetricValue"] {{ color: {PALETTE['text']} !important; font-size: 1.7rem !important; font-weight: 700; }}
    [data-testid="stMetricDelta"] {{ font-size: .8rem !important; }}
    /* ── Buttons ── */
    .stButton > button {{
        background: {PALETTE['info']};
        color: #fff;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
        transition: opacity .2s;
    }}
    .stButton > button:hover {{ opacity: .85; }}
    /* ── Inputs ── */
    .stNumberInput input, .stTextInput input, .stSelectbox select {{
        background: {PALETTE['surface2']} !important;
        color: {PALETTE['text']} !important;
        border: 1px solid {PALETTE['border']} !important;
        border-radius: 6px;
    }}
    /* ── Dataframe ── */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    /* ── Alert boxes ── */
    .alert-fraud {{
        background: rgba(218,54,51,.15);
        border: 1px solid {PALETTE['danger']};
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
    }}
    .alert-safe {{
        background: rgba(35,134,54,.15);
        border: 1px solid {PALETTE['accent']};
        border-radius: 10px;
        padding: 18px 22px;
        margin: 10px 0;
    }}
    /* ── Section divider ── */
    .section-divider {{
        border-top: 1px solid {PALETTE['border']};
        margin: 24px 0;
    }}
    /* ── Badge ── */
    .badge {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: .72rem;
        font-weight: 600;
    }}
    .badge-fraud {{ background: rgba(218,54,51,.25); color: {PALETTE['danger']}; }}
    .badge-safe  {{ background: rgba(35,134,54,.25);  color: {PALETTE['accent']}; }}
    /* ── Sidebar nav ── */
    .sidebar-title {{
        font-weight: 700;
        font-size: 1.15rem;
        color: {PALETTE['text']};
        padding: 10px 0 6px;
        border-bottom: 1px solid {PALETTE['border']};
        margin-bottom: 12px;
    }}
    /* ── Expander ── */
    [data-testid="stExpander"] {{
        background: {PALETTE['surface']} !important;
        border: 1px solid {PALETTE['border']} !important;
        border-radius: 8px !important;
    }}
    /* ── File uploader ── */
    [data-testid="stFileUploader"] {{
        background: {PALETTE['surface2']};
        border: 1px dashed {PALETTE['border']};
        border-radius: 8px;
    }}
    /* ── Plotly chart bg ── */
    .js-plotly-plot .plotly {{ border-radius: 10px; }}
    /* ── Progress bar ── */
    .stProgress > div > div {{ background: {PALETTE['info']}; }}
    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: {PALETTE['surface']};
        border-radius: 8px 8px 0 0;
        gap: 2px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: {PALETTE['muted']};
        border-radius: 6px;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {PALETTE['surface2']} !important;
        color: {PALETTE['text']} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS & CACHES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model and scaler from disk (cached across reruns)."""
    model_path  = Path("fraud_model.pkl")
    scaler_path = Path("scaler.pkl")
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load & lightly cache the CSV dataset."""
    csv_path = Path("creditcard.csv")
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return df


def plotly_theme():
    """Return common Plotly layout kwargs for dark theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PALETTE["surface"],
        font_color=PALETTE["text"],
        font_family="Inter, Segoe UI, sans-serif",
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        margin=dict(l=10, r=10, t=40, b=10),
    )


def feature_names():
    v_cols = [f"V{i}" for i in range(1, 29)]
    return ["Time"] + v_cols + ["Amount"]


def scale_input(scaler, time_val: float, amount_val: float, v_vals: np.ndarray):
    """
    Build a 30-feature vector and apply the scaler.
    The original scaler was fitted on [Time, V1–V28, Amount] in that order.
    """
    row = np.concatenate([[time_val], v_vals, [amount_val]]).reshape(1, -1)
    try:
        row_scaled = scaler.transform(row)
    except Exception:
        # Fallback: scale only Time & Amount columns (index 0 and 29)
        row_scaled = row.copy()
        row_scaled[0, 0]  = scaler.transform([[time_val]])[0][0]
        row_scaled[0, -1] = scaler.transform([[amount_val]])[0][0]
    return row_scaled


def bulk_predict(model, scaler, df: pd.DataFrame):
    """Run predictions on a DataFrame that has the standard creditcard columns."""
    required = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"

    X = df[feature_names()].values
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X_scaled = X.copy()
        X_scaled[:, 0]  = scaler.transform(X[:, 0].reshape(-1, 1)).ravel()
        X_scaled[:, -1] = scaler.transform(X[:, -1].reshape(-1, 1)).ravel()

    preds  = model.predict(X_scaled)
    probas = model.predict_proba(X_scaled)[:, 1]
    return preds, probas


def fraud_highlight(row):
    """Pandas Styler: highlight fraud rows red."""
    color = (
        f"background-color: rgba(218,54,51,.18); color: {PALETTE['danger']};"
        if row.get("Prediction") == "🚨 Fraud"
        else ""
    )
    return [color] * len(row)


def df_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Predictions")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">🛡️ FraudGuard ML</div>',
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Single Prediction", "📂 Bulk Prediction",
         "📊 Data Analysis", "🧪 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Artifact status
    model, scaler = load_artifacts()
    df_raw        = load_dataset()

    def status_row(label, ok):
        icon  = "✅" if ok else "❌"
        color = PALETTE["accent"] if ok else PALETTE["danger"]
        st.markdown(
            f'<small style="color:{color}">{icon} {label}</small>',
            unsafe_allow_html=True,
        )

    st.markdown("**System status**")
    status_row("fraud_model.pkl", model  is not None)
    status_row("scaler.pkl",      scaler is not None)
    status_row("creditcard.csv",  df_raw is not None)

    if model is not None:
        st.markdown(
            f'<small style="color:{PALETTE["muted"]}">Model: '
            f'{type(model).__name__}</small>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Confidence threshold slider
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.10, max_value=0.90, value=0.50, step=0.01,
        help="Predictions with probability ≥ this value are flagged as fraud.",
    )
    st.caption(f"Current threshold: **{threshold:.0%}**")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"🕐 {datetime.now().strftime('%d %b %Y  %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED GUARD
# ══════════════════════════════════════════════════════════════════════════════

def require_model():
    if model is None or scaler is None:
        st.error(
            "**Model or scaler not found.** Place `fraud_model.pkl` and "
            "`scaler.pkl` in the same directory as `app.py` and restart."
        )
        st.stop()


def require_dataset():
    if df_raw is None:
        st.warning(
            "**`creditcard.csv` not found.** This page needs the raw dataset."
        )
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    # Hero
    st.markdown(
        f"""
        <div style="padding:32px 0 8px;">
          <h1 style="margin-bottom:4px;">🛡️ FraudGuard ML Dashboard</h1>
          <p style="color:{PALETTE['muted']};font-size:1.05rem;margin-top:0;">
            Real-time credit card fraud detection powered by an ensemble ML model.
            Monitor transactions, investigate anomalies, and export reports — all in one place.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Key metrics ──────────────────────────────────────────────────────────
    if df_raw is not None:
        total       = len(df_raw)
        fraud_count = int(df_raw["Class"].sum())
        normal_count= total - fraud_count
        fraud_pct   = fraud_count / total * 100
        normal_pct  = 100 - fraud_pct
        avg_amount  = df_raw["Amount"].mean()
        fraud_avg   = df_raw[df_raw["Class"] == 1]["Amount"].mean()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Transactions", f"{total:,}")
        c2.metric("Fraud Cases",   f"{fraud_count:,}",  f"{fraud_pct:.3f}%",  delta_color="inverse")
        c3.metric("Normal Cases",  f"{normal_count:,}", f"{normal_pct:.2f}%")
        c4.metric("Avg Amount",    f"${avg_amount:,.2f}")
        c5.metric("Avg Fraud Amt", f"${fraud_avg:,.2f}", delta_color="inverse")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Mini charts row ──────────────────────────────────────────────────
        cc1, cc2 = st.columns([1, 2])

        with cc1:
            st.markdown("##### Class distribution")
            fig_pie = go.Figure(go.Pie(
                labels=["Normal", "Fraud"],
                values=[normal_count, fraud_count],
                hole=0.6,
                marker_colors=[PALETTE["chart3"], PALETTE["chart2"]],
                textfont_size=13,
            ))
            fig_pie.update_layout(
                **plotly_theme(),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                height=270,
                annotations=[dict(
                    text=f"{fraud_pct:.2f}%<br>fraud",
                    x=0.5, y=0.5, font_size=15, showarrow=False,
                    font_color=PALETTE["danger"],
                )],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with cc2:
            st.markdown("##### Transaction amount over time (sample 2 000)")
            sample = df_raw.sample(min(2000, len(df_raw)), random_state=42).sort_values("Time")
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scattergl(
                x=sample["Time"], y=sample["Amount"],
                mode="markers",
                marker=dict(
                    color=sample["Class"].map({0: PALETTE["chart3"], 1: PALETTE["chart2"]}),
                    size=4, opacity=0.7,
                ),
                text=sample["Class"].map({0: "Normal", 1: "Fraud"}),
                name="",
            ))
            fig_ts.update_layout(
                **plotly_theme(), height=270,
                xaxis_title="Time (s)", yaxis_title="Amount ($)",
                showlegend=False,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # ── Recent transactions table ────────────────────────────────────────
        st.markdown("##### Recent transactions (last 20 rows)")
        tail = df_raw.tail(20)[["Time", "Amount", "Class"]].copy()
        tail["Class"] = tail["Class"].map({0: "✅ Normal", 1: "🚨 Fraud"})
        tail["Amount"] = tail["Amount"].map("${:,.2f}".format)
        st.dataframe(tail.reset_index(drop=True), use_container_width=True, height=320)

    else:
        st.info(
            "Place `creditcard.csv` in the app directory to see live metrics. "
            "The prediction pages work without it."
        )

    # ── Feature guide ────────────────────────────────────────────────────────
    with st.expander("📖 Dataset & model overview", expanded=False):
        st.markdown(
            """
            | Feature | Description |
            |---------|-------------|
            | `Time`  | Seconds elapsed since the first transaction in the dataset |
            | `V1–V28` | PCA-transformed features (anonymised for privacy) |
            | `Amount` | Transaction amount in USD |
            | `Class` | **0** = Normal · **1** = Fraud |

            **Model:** RandomForest Classifier trained on the Kaggle *Credit Card Fraud Detection* dataset  
            **Scaler:** StandardScaler applied to `Time` and `Amount`  
            **Imbalance handling:** SMOTE oversampling / class weighting  
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Single Prediction":
    require_model()

    st.markdown("## 🔍 Single Transaction Analysis")
    st.caption("Enter known values. Hidden PCA features (V1–V28) can be filled from a random sample or set to zero.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    mode_tab, manual_tab = st.tabs(["🎲 Random sample mode", "✏️ Manual input mode"])

    # ── Random sample mode ───────────────────────────────────────────────────
    with mode_tab:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_random = st.button("🎲 Draw random transaction", use_container_width=True)

        if run_random:
            if df_raw is None:
                st.error("creditcard.csv not found.")
            else:
                sample   = df_raw.sample(1, random_state=int(time_module.time())).copy()
                true_cls = int(sample["Class"].values[0])
                X        = sample[feature_names()].values
                try:
                    X_sc = scaler.transform(X)
                except Exception:
                    X_sc = X.copy()
                    X_sc[0, 0]  = scaler.transform([[X[0, 0]]])[0][0]
                    X_sc[0, -1] = scaler.transform([[X[0, -1]]])[0][0]

                proba = model.predict_proba(X_sc)[0][1]
                pred  = 1 if proba >= threshold else 0

                # ── Result card ──────────────────────────────────────────────
                if pred == 1:
                    st.markdown(
                        f'<div class="alert-fraud">'
                        f'<h2 style="margin:0;color:{PALETTE["danger"]}">🚨 FRAUD DETECTED</h2>'
                        f'<p style="margin:8px 0 0;color:{PALETTE["text"]};">'
                        f'Fraud probability: <b>{proba:.2%}</b> &nbsp;|&nbsp; '
                        f'True label: <b>{"Fraud" if true_cls==1 else "Normal"}</b>'
                        f'</p></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="alert-safe">'
                        f'<h2 style="margin:0;color:{PALETTE["accent"]}">✅ NORMAL TRANSACTION</h2>'
                        f'<p style="margin:8px 0 0;color:{PALETTE["text"]};">'
                        f'Fraud probability: <b>{proba:.2%}</b> &nbsp;|&nbsp; '
                        f'True label: <b>{"Fraud" if true_cls==1 else "Normal"}</b>'
                        f'</p></div>',
                        unsafe_allow_html=True,
                    )

                # ── Gauge chart ──────────────────────────────────────────────
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number=dict(suffix="%", font_size=36),
                    title=dict(text="Fraud probability", font_size=16),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor=PALETTE["muted"]),
                        bar=dict(color=PALETTE["danger"] if pred == 1 else PALETTE["accent"]),
                        bgcolor=PALETTE["surface2"],
                        bordercolor=PALETTE["border"],
                        steps=[
                            dict(range=[0, threshold*100],   color=PALETTE["surface"]),
                            dict(range=[threshold*100, 100], color="rgba(218,54,51,.10)"),
                        ],
                        threshold=dict(
                            line=dict(color=PALETTE["warning"], width=3),
                            thickness=0.85,
                            value=threshold * 100,
                        ),
                    ),
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color=PALETTE["text"], height=280,
                )
                g1, g2 = st.columns([1, 1])
                g1.plotly_chart(fig_gauge, use_container_width=True)

                with g2:
                    st.markdown("**Transaction details**")
                    details = {
                        "Time (s)":   f"{sample['Time'].values[0]:,.0f}",
                        "Amount ($)": f"${sample['Amount'].values[0]:,.2f}",
                        "True class": "🚨 Fraud" if true_cls == 1 else "✅ Normal",
                        "Predicted":  "🚨 Fraud" if pred == 1 else "✅ Normal",
                        "Correct":    "✔️ Yes" if pred == true_cls else "❌ No",
                    }
                    for k, v in details.items():
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;'
                            f'padding:5px 0;border-bottom:1px solid {PALETTE["border"]};">'
                            f'<span style="color:{PALETTE["muted"]}">{k}</span>'
                            f'<span style="font-weight:600">{v}</span></div>',
                            unsafe_allow_html=True,
                        )

                # ── V-feature bar chart ──────────────────────────────────────
                with st.expander("📊 View PCA feature values (V1–V28)", expanded=False):
                    v_vals  = sample[[f"V{i}" for i in range(1, 29)]].values.flatten()
                    v_names = [f"V{i}" for i in range(1, 29)]
                    fig_bar = go.Figure(go.Bar(
                        x=v_names, y=v_vals,
                        marker_color=[
                            PALETTE["danger"] if v < -2 or v > 2 else PALETTE["chart1"]
                            for v in v_vals
                        ],
                    ))
                    fig_bar.update_layout(
                        **plotly_theme(), height=280,
                        title="PCA features — values outside ±2σ highlighted",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Manual input mode ────────────────────────────────────────────────────
    with manual_tab:
        st.markdown("Enter `Time` and `Amount`. V1–V28 default to **0** (typical for an average transaction).")

        col_t, col_a = st.columns(2)
        m_time   = col_t.number_input("Time (seconds since first txn)", min_value=0.0, value=50000.0, step=100.0)
        m_amount = col_a.number_input("Amount ($)", min_value=0.0, value=50.0, step=1.0)

        # FIX: initialise v_inputs before the expander so it always has a value
        v_inputs = [0.0] * 28

        with st.expander("⚙️ Advanced — set V1–V28 manually (optional)", expanded=False):
            v_inputs = []
            cols = st.columns(7)
            for i in range(1, 29):
                idx = (i - 1) % 7
                v_inputs.append(cols[idx].number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}"))

        if st.button("🔮 Predict this transaction", use_container_width=False):
            v_arr   = np.array(v_inputs)
            X_input = scale_input(scaler, m_time, m_amount, v_arr)
            proba   = model.predict_proba(X_input)[0][1]
            pred    = 1 if proba >= threshold else 0

            if pred == 1:
                st.markdown(
                    f'<div class="alert-fraud">'
                    f'<b style="color:{PALETTE["danger"]}">🚨 Fraud detected</b>'
                    f' — probability <b>{proba:.2%}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="alert-safe">'
                    f'<b style="color:{PALETTE["accent"]}">✅ Normal transaction</b>'
                    f' — fraud probability <b>{proba:.2%}</b></div>',
                    unsafe_allow_html=True,
                )

            # Probability bar
            st.progress(proba)
            st.caption(
                f"Fraud probability: **{proba:.2%}** | Threshold: **{threshold:.0%}**"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BULK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📂 Bulk Prediction":
    require_model()

    st.markdown("## 📂 Bulk Transaction Screening")
    st.caption("Upload a CSV with columns V1–V28, Time, Amount (Class optional). The app will flag fraud transactions.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Upload ───────────────────────────────────────────────────────────────
    upload = st.file_uploader("Drop CSV file here", type=["csv"])

    use_sample = False
    n_sample = 500
    if upload is None and df_raw is not None:
        st.markdown("— or —")
        n_sample = st.slider("Use a random sample from `creditcard.csv`", 100, 5000, 500, 100)
        use_sample = st.checkbox("Run on random sample", value=False)

    if upload is not None or use_sample:
        with st.spinner("Running predictions…"):
            if upload is not None:
                df_bulk = pd.read_csv(upload)
            else:
                df_bulk = df_raw.sample(n_sample, random_state=42).reset_index(drop=True)

            preds, probas = bulk_predict(model, scaler, df_bulk)

            if preds is None:
                st.error(probas)   # probas holds the error string here
            else:
                # Apply threshold
                preds_thresh = (probas >= threshold).astype(int)

                results = df_bulk.copy()
                results["Fraud_Probability"] = probas.round(4)
                results["Prediction"] = np.where(preds_thresh == 1, "🚨 Fraud", "✅ Normal")
                if "Class" in results.columns:
                    results["True_Label"] = results["Class"].map({0: "✅ Normal", 1: "🚨 Fraud"})

                # ── Summary metrics ──────────────────────────────────────────
                total_b   = len(results)
                fraud_b   = int(preds_thresh.sum())
                normal_b  = total_b - fraud_b
                fraud_pct = fraud_b / total_b * 100 if total_b else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Screened",       f"{total_b:,}")
                m2.metric("Flagged Fraud",  f"{fraud_b:,}",  f"{fraud_pct:.2f}%",  delta_color="inverse")
                m3.metric("Normal",         f"{normal_b:,}")
                m4.metric("Avg Fraud Prob", f"{probas.mean():.2%}")

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

                # ── Filters ──────────────────────────────────────────────────
                cf1, cf2, cf3 = st.columns(3)
                show_filter = cf1.selectbox("Show", ["All", "Fraud only", "Normal only"])
                sort_col    = cf2.selectbox("Sort by", ["Fraud_Probability", "Amount", "Time"])
                sort_asc    = cf3.checkbox("Ascending order", value=False)

                df_show = results.copy()
                if show_filter == "Fraud only":
                    df_show = df_show[df_show["Prediction"] == "🚨 Fraud"]
                elif show_filter == "Normal only":
                    df_show = df_show[df_show["Prediction"] == "✅ Normal"]
                df_show = df_show.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

                # ── Table ────────────────────────────────────────────────────
                display_cols = ["Time", "Amount", "Fraud_Probability", "Prediction"]
                if "True_Label" in df_show.columns:
                    display_cols.append("True_Label")
                st.dataframe(
                    df_show[display_cols]
                    .style.apply(fraud_highlight, axis=1),
                    use_container_width=True, height=400,
                )

                # ── Distribution chart ───────────────────────────────────────
                c_dist1, c_dist2 = st.columns(2)
                with c_dist1:
                    fig_prob = px.histogram(
                        results, x="Fraud_Probability", nbins=50,
                        color_discrete_sequence=[PALETTE["chart1"]],
                        title="Fraud probability distribution",
                    )
                    fig_prob.add_vline(x=threshold, line_dash="dash",
                                       line_color=PALETTE["warning"],
                                       annotation_text=f"Threshold {threshold:.0%}")
                    fig_prob.update_layout(**plotly_theme(), height=280)
                    st.plotly_chart(fig_prob, use_container_width=True)

                with c_dist2:
                    fraud_rows   = results[results["Prediction"] == "🚨 Fraud"]
                    normal_rows  = results[results["Prediction"] != "🚨 Fraud"]
                    fig_amt = go.Figure()
                    fig_amt.add_trace(go.Box(
                        y=normal_rows["Amount"], name="Normal",
                        marker_color=PALETTE["chart3"],
                    ))
                    fig_amt.add_trace(go.Box(
                        y=fraud_rows["Amount"], name="Fraud",
                        marker_color=PALETTE["chart2"],
                    ))
                    fig_amt.update_layout(**plotly_theme(), height=280,
                                          title="Amount distribution by class")
                    st.plotly_chart(fig_amt, use_container_width=True)

                # ── Export ───────────────────────────────────────────────────
                st.markdown("#### Export results")
                e1, e2, e3 = st.columns(3)
                csv_bytes  = results.to_csv(index=False).encode()
                excel_bytes = df_to_excel(results)

                e1.download_button("⬇ Download CSV",   csv_bytes,
                                   "fraud_results.csv", "text/csv")
                e2.download_button("⬇ Download Excel", excel_bytes,
                                   "fraud_results.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                # ZIP (CSV + JSON summary)
                summary = {
                    "total": total_b, "fraud": fraud_b, "normal": normal_b,
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat(),
                }
                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    zf.writestr("predictions.csv", results.to_csv(index=False))
                    zf.writestr("summary.json", pd.Series(summary).to_json())
                e3.download_button("⬇ Download ZIP", zip_buf.getvalue(),
                                   "fraud_report.zip", "application/zip")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Data Analysis":
    require_dataset()

    st.markdown("## 📊 Exploratory Data Analysis")
    st.caption("Deep-dive into the raw dataset to understand fraud patterns.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "Amount", "Time", "PCA Features", "Correlation"]
    )

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Class imbalance")
        c1, c2 = st.columns(2)
        counts  = df_raw["Class"].value_counts()
        labels  = ["Normal", "Fraud"]
        vals    = [counts.get(0, 0), counts.get(1, 0)]

        fig_bar = go.Figure(go.Bar(
            x=labels, y=vals,
            marker_color=[PALETTE["chart3"], PALETTE["chart2"]],
            text=[f"{v:,}" for v in vals],
            textposition="outside",
        ))
        fig_bar.update_layout(**plotly_theme(), height=320,
                               title="Transaction count by class")
        c1.plotly_chart(fig_bar, use_container_width=True)

        fig_pie2 = go.Figure(go.Pie(
            labels=labels, values=vals, hole=0.55,
            marker_colors=[PALETTE["chart3"], PALETTE["chart2"]],
        ))
        fig_pie2.update_layout(**plotly_theme(), height=320,
                                title="Class proportion")
        c2.plotly_chart(fig_pie2, use_container_width=True)

        st.markdown("#### Statistical summary")
        st.dataframe(df_raw.describe().T.style.format("{:.4f}"),
                     use_container_width=True)

    # ── Amount ───────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Amount distribution")
        fraud_df  = df_raw[df_raw["Class"] == 1]
        normal_df = df_raw[df_raw["Class"] == 0]

        max_amt = st.slider("Cap amount at ($)", 10, int(df_raw["Amount"].quantile(0.99)), 500, 10)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=normal_df["Amount"].clip(upper=max_amt),
            nbinsx=80, name="Normal",
            marker_color=PALETTE["chart3"], opacity=0.7,
        ))
        fig_hist.add_trace(go.Histogram(
            x=fraud_df["Amount"].clip(upper=max_amt),
            nbinsx=80, name="Fraud",
            marker_color=PALETTE["chart2"], opacity=0.85,
        ))
        fig_hist.update_layout(
            **plotly_theme(), barmode="overlay", height=340,
            xaxis_title="Amount ($)", yaxis_title="Count",
            title=f"Amount distribution (capped at ${max_amt})",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # FIX: use correct color_discrete_map keys matching mapped string labels
        df_amt_filtered = df_raw[df_raw["Amount"] <= max_amt].copy()
        df_amt_filtered["Class_Label"] = df_amt_filtered["Class"].map({0: "Normal", 1: "Fraud"})

        fig_box = px.box(
            df_amt_filtered,
            x="Class_Label",
            y="Amount",
            color="Class_Label",
            color_discrete_map={"Normal": PALETTE["chart3"], "Fraud": PALETTE["chart2"]},
            title="Amount box plot by class",
        )
        fig_box.update_layout(**plotly_theme(), height=320, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Normal — mean",   f"${normal_df['Amount'].mean():,.2f}")
        a2.metric("Normal — max",    f"${normal_df['Amount'].max():,.2f}")
        a3.metric("Fraud — mean",    f"${fraud_df['Amount'].mean():,.2f}")
        a4.metric("Fraud — max",     f"${fraud_df['Amount'].max():,.2f}")

    # ── Time ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Fraud rate across time")
        bin_size = st.select_slider("Time bin (hours)", [1, 2, 4, 6, 12, 24], 6)
        bin_sec  = bin_size * 3600

        # FIX: work on a copy to avoid mutating the cached df_raw
        df_time = df_raw.copy()
        df_time["TimeBin"] = (df_time["Time"] // bin_sec).astype(int)
        time_grp = df_time.groupby("TimeBin").agg(
            Total=("Class", "count"),
            Fraud=("Class", "sum"),
        ).reset_index()
        time_grp["FraudRate"] = time_grp["Fraud"] / time_grp["Total"] * 100
        time_grp["Hour"]      = time_grp["TimeBin"] * bin_size

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=time_grp["Hour"], y=time_grp["FraudRate"],
            mode="lines+markers", name="Fraud rate (%)",
            line=dict(color=PALETTE["danger"], width=2),
            fill="tozeroy", fillcolor="rgba(218,54,51,.08)",
        ))
        fig_line.update_layout(
            **plotly_theme(), height=320,
            xaxis_title=f"Time (hours, {bin_size}h bins)",
            yaxis_title="Fraud rate (%)", title="Fraud rate over time",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # ── PCA Features ─────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### V-feature mean comparison: Fraud vs Normal")
        v_cols   = [f"V{i}" for i in range(1, 29)]
        v_fraud  = df_raw[df_raw["Class"] == 1][v_cols].mean()
        v_normal = df_raw[df_raw["Class"] == 0][v_cols].mean()

        fig_v = go.Figure()
        fig_v.add_trace(go.Bar(x=v_cols, y=v_normal, name="Normal",
                               marker_color=PALETTE["chart3"], opacity=0.8))
        fig_v.add_trace(go.Bar(x=v_cols, y=v_fraud,  name="Fraud",
                               marker_color=PALETTE["chart2"], opacity=0.8))
        fig_v.update_layout(
            **plotly_theme(), barmode="group", height=360,
            title="Mean PCA feature values by class",
        )
        st.plotly_chart(fig_v, use_container_width=True)

        # Violin plots for a selected feature
        sel_v = st.selectbox("Violin plot for feature", v_cols, index=0)
        fig_violin = go.Figure()
        fig_violin.add_trace(go.Violin(
            y=df_raw[df_raw["Class"] == 0][sel_v], name="Normal",
            fillcolor=PALETTE["chart3"], line_color=PALETTE["chart3"],
            opacity=0.7, box_visible=True, meanline_visible=True,
        ))
        fig_violin.add_trace(go.Violin(
            y=df_raw[df_raw["Class"] == 1][sel_v], name="Fraud",
            fillcolor=PALETTE["chart2"], line_color=PALETTE["chart2"],
            opacity=0.7, box_visible=True, meanline_visible=True,
        ))
        fig_violin.update_layout(
            **plotly_theme(), height=320,
            title=f"Distribution of {sel_v} by class",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Correlation ──────────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Correlation heatmap")
        n_feat = st.slider("Features to include (top N by variance)", 5, 30, 15)
        top_cols = df_raw[v_cols + ["Amount", "Time"]].var().nlargest(n_feat).index.tolist()
        corr_df  = df_raw[top_cols + ["Class"]].corr()

        fig_heat, ax = plt.subplots(figsize=(12, 9))
        fig_heat.patch.set_facecolor(PALETTE["surface"])
        ax.set_facecolor(PALETTE["surface"])
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", linewidths=0.4,
            cmap="RdYlGn", center=0, ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax.tick_params(colors=PALETTE["muted"], labelsize=8)
        plt.title("Feature correlation matrix", color=PALETTE["text"], pad=14)
        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Model Performance":
    require_model()
    require_dataset()

    st.markdown("## 🧪 Model Performance Evaluation")
    st.caption("Evaluate the loaded model on a held-out subset of the dataset.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    test_size = st.slider("Test set size (rows)", 500, min(20000, len(df_raw)), 5000, 500)

    if st.button("▶ Evaluate model", use_container_width=False):
        with st.spinner("Evaluating…"):
            df_test = df_raw.sample(test_size, random_state=99).reset_index(drop=True)
            X_test  = df_test[feature_names()].values
            y_test  = df_test["Class"].values

            try:
                X_sc = scaler.transform(X_test)
            except Exception:
                X_sc = X_test.copy()
                X_sc[:, 0]  = scaler.transform(X_test[:, 0].reshape(-1, 1)).ravel()
                X_sc[:, -1] = scaler.transform(X_test[:, -1].reshape(-1, 1)).ravel()

            y_pred  = (model.predict_proba(X_sc)[:, 1] >= threshold).astype(int)
            y_proba = model.predict_proba(X_sc)[:, 1]

            # ── Headline metrics ─────────────────────────────────────────────
            roc    = roc_auc_score(y_test, y_proba)
            ap     = average_precision_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            prec   = report.get("1", {}).get("precision", 0)
            rec    = report.get("1", {}).get("recall",    0)
            f1     = report.get("1", {}).get("f1-score",  0)
            acc    = report.get("accuracy", 0)

            h1, h2, h3, h4, h5, h6 = st.columns(6)
            h1.metric("Accuracy",   f"{acc:.3f}")
            h2.metric("ROC-AUC",    f"{roc:.3f}")
            h3.metric("Avg Prec",   f"{ap:.3f}")
            h4.metric("Precision",  f"{prec:.3f}")
            h5.metric("Recall",     f"{rec:.3f}")
            h6.metric("F1-Score",   f"{f1:.3f}")
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # ── ROC & PR curves ──────────────────────────────────────────────
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            prc, rcl, _ = precision_recall_curve(y_test, y_proba)

            rc1, rc2 = st.columns(2)
            with rc1:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc:.3f})",
                    line=dict(color=PALETTE["chart1"], width=2),
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(color=PALETTE["muted"], dash="dash"),
                ))
                fig_roc.update_layout(**plotly_theme(), height=320,
                                       title="ROC Curve",
                                       xaxis_title="FPR", yaxis_title="TPR")
                st.plotly_chart(fig_roc, use_container_width=True)

            with rc2:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=rcl, y=prc, mode="lines", name=f"PR (AP={ap:.3f})",
                    line=dict(color=PALETTE["chart4"], width=2),
                    fill="tozeroy", fillcolor="rgba(210,168,255,.08)",
                ))
                fig_pr.update_layout(**plotly_theme(), height=320,
                                      title="Precision-Recall Curve",
                                      xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig_pr, use_container_width=True)

            # ── Confusion matrix ─────────────────────────────────────────────
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=["Predicted Normal", "Predicted Fraud"],
                y=["Actual Normal",    "Actual Fraud"],
                colorscale=[
                    [0, PALETTE["surface"]],
                    [0.5, PALETTE["info"]],
                    [1, PALETTE["chart1"]],
                ],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=22, color=PALETTE["text"]),
            ))
            fig_cm.update_layout(
                **plotly_theme(), height=340,
                title="Confusion Matrix",
            )
            fig_cm.update_xaxes(side="bottom")
            st.plotly_chart(fig_cm, use_container_width=True)

            # ── Feature importances ──────────────────────────────────────────
            if hasattr(model, "feature_importances_"):
                st.markdown("#### Feature importances")
                importances = model.feature_importances_
                fi_df = (
                    pd.DataFrame({"Feature": feature_names(), "Importance": importances})
                    .sort_values("Importance", ascending=False)
                    .head(20)
                )
                fig_fi = go.Figure(go.Bar(
                    y=fi_df["Feature"][::-1], x=fi_df["Importance"][::-1],
                    orientation="h",
                    marker_color=PALETTE["chart1"],
                ))
                fig_fi.update_layout(
                    **plotly_theme(), height=420,
                    title="Top-20 Feature Importances",
                    xaxis_title="Importance", yaxis_title="",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # ── Full report ──────────────────────────────────────────────────
            with st.expander("📋 Full classification report"):
                st.text(classification_report(y_test, y_pred,
                                              target_names=["Normal", "Fraud"]))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About FraudGuard ML")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        **FraudGuard ML** is a fully self-contained Streamlit dashboard for real-time
        credit card fraud detection. It wraps a trained scikit-learn model (typically
        a `RandomForestClassifier`) and a `StandardScaler` loaded via `joblib`.

        ---

        ### Feature highlights

        | Feature | Description |
        |---------|-------------|
        | **Home** | Live KPIs, class balance, transaction scatter |
        | **Single Prediction** | Random draw + manual input with gauge viz |
        | **Bulk Prediction** | CSV upload, threshold filtering, CSV/Excel/ZIP export |
        | **Data Analysis** | 5-tab EDA: overview, amount, time, PCA, correlation |
        | **Model Performance** | ROC, PR curves, confusion matrix, feature importances |
        | **Threshold slider** | Sidebar-global — adjusts all predictions live |

        ---

        ### File requirements

        ```
        app.py             ← this file
        fraud_model.pkl    ← trained sklearn classifier
        scaler.pkl         ← fitted sklearn scaler
        creditcard.csv     ← raw dataset (optional for prediction pages)
        ```

        ### Run

        ```bash
        pip install streamlit pandas numpy joblib scikit-learn \\
                    matplotlib seaborn plotly openpyxl xlsxwriter
        streamlit run app.py
        ```

        ---
        <small style="color:{PALETTE['muted']}">
        Built with Streamlit · Plotly · scikit-learn · Seaborn
        </small>
        """,
        unsafe_allow_html=True,
    )