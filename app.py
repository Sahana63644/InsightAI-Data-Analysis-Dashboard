import json
from io import BytesIO, StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(
    page_title="InsightAI | Data Intelligence",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colorful custom UI. The app uses dark blue-purple text on a soft gradient,
# with translucent panels and brightly colored accents.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #183532;
        --muted: #617873;
        --teal: #087f72;
        --teal-dark: #075e54;
        --mint: #bce8d9;
        --coral: #e78a67;
        --gold: #d7a84b;
        --panel: #ffffff;
        --line: #dce9e3;
        --canvas: #f4f7f2;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp {
        background:
          radial-gradient(circle at 92% 2%, rgba(188,232,217,.30), transparent 24rem),
          linear-gradient(145deg, #f7f8f3 0%, #f1f6f1 55%, #f8f7f1 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: rgba(247,248,243,.94); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123d39 0%, #174a43 58%, #103832 100%);
        border-right: 1px solid #d2e3da;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption { color: #e0f0e8 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.08); border-color: #80b9a7;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] div[data-baseweb="input"] > div { background: #f9fcf8; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #183532 !important; letter-spacing: -0.035em; }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }
    p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #405d56; }
    .hero {
        padding: 29px 32px; border-radius: 23px; margin: 2px 0 22px 0; color: white;
        background: linear-gradient(115deg, #123d39 0%, #087f72 58%, #43a58e 100%);
        box-shadow: 0 14px 32px rgba(14,75,64,.15); position: relative; overflow: hidden;
    }
    .hero:after { content: "✳"; position: absolute; right: 34px; top: 0px; font-size: 104px; opacity: 0.12; color: white; }
    .hero h1 { color: white !important; margin: 0 0 8px 0; font-size: clamp(2rem, 3.5vw, 3.05rem); }
    .hero p { color: rgba(255,255,255,0.91) !important; margin: 0; font-size: 1.02rem; }
    .hero-tag { display: inline-block; font-size: .72rem; font-weight: 800; letter-spacing: .11em; text-transform: uppercase;
        color: #f1fff7; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.25);
        padding: 6px 11px; border-radius: 100px; margin-bottom: 12px; }
    .section-heading { display:flex; align-items:center; gap:11px; margin:26px 0 13px 0;
        font-family:'Manrope',sans-serif; font-weight:800; font-size:1.35rem; color:#183532; }
    .section-heading .bubble { display:inline-flex; width:37px; height:37px; align-items:center; justify-content:center;
        border-radius:12px; color:white; background:linear-gradient(135deg,#087f72,#43a58e); box-shadow:0 5px 13px rgba(8,127,114,.18); }
    .kpi { border-radius:18px; padding:19px 20px 17px; min-height:130px; border:1px solid #dce9e3;
        box-shadow:0 7px 20px rgba(27,67,54,.055); background:rgba(255,255,255,.96); position:relative; overflow:hidden;
        transition:transform .18s ease, box-shadow .18s ease; }
    .kpi:hover { transform:translateY(-2px); box-shadow:0 11px 24px rgba(27,67,54,.09); }
    .kpi:after { content:""; position:absolute; width:88px; height:88px; border-radius:50%; right:-25px; top:-25px; background:var(--accent); opacity:.12; }
    .kpi-label { font-size:.79rem; font-weight:800; letter-spacing:.045em; color:#617873; margin-bottom:10px; }
    .kpi-value { font-family:'Manrope',sans-serif; font-size:clamp(1.5rem,2.2vw,2.1rem); font-weight:800;
        letter-spacing:-.04em; line-height:1.12; color:var(--accent); }
    .kpi-foot { font-size:.75rem; color:#728780; margin-top:9px; }
    .glass-note { padding:14px 17px; border-radius:15px; background:rgba(255,255,255,.96); color:#183532 !important; border:1px solid #dce9e3; box-shadow:0 4px 13px rgba(27,67,54,.04); }
    .glass-note b, .glass-note strong, .glass-note span { color:inherit !important; }
    [data-testid="stMetric"] { background:rgba(255,255,255,.96); border:1px solid #dce9e3; padding:14px 16px; border-radius:15px; box-shadow:0 4px 13px rgba(27,67,54,.04); }
    [data-testid="stMetricLabel"] { color:#617873 !important; }
    [data-testid="stMetricValue"] { color:#183532 !important; }
    div[data-testid="stTabs"] button { border-radius:10px 10px 0 0; font-weight:700; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:#087f72; border-bottom-color:#087f72; }
    /* Consistent action buttons: readable labels, visible icons, accessible focus state. */
    .stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
        min-height: 2.75rem;
        border-radius: 12px;
        border: 1px solid #087f72;
        color: #ffffff !important;
        font-weight: 750;
        letter-spacing: .005em;
        background: linear-gradient(135deg, #087f72 0%, #0a6d62 100%);
        box-shadow: 0 5px 14px rgba(8,127,114,.16);
        transition: transform .16s ease, background .16s ease, box-shadow .16s ease;
    }
    .stButton > button p, .stDownloadButton > button p,
    .stButton > button span, .stDownloadButton > button span {
        color: #ffffff !important;
        font-weight: 750;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #ffffff !important;
        background: linear-gradient(135deg, #075e54 0%, #064b44 100%);
        border-color: #075e54;
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(8,95,83,.22);
    }
    .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
        outline: 3px solid #e78a67 !important;
        outline-offset: 3px;
    }
    .stDownloadButton [data-testid="stMarkdownContainer"] p { margin: 0; }
    .stDownloadButton svg, .stButton svg { color: #ffffff !important; fill: currentColor; }
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,.92);
        border: 1.5px dashed #63a997;
        padding: 10px;
        border-radius: 15px;
    }
    [data-testid="stFileUploader"] section {
        border-radius: 11px;
        background: rgba(8,127,114,.045);
    }
    [data-testid="stFileUploader"] section button {
        background: #ffffff !important;
        color: #075e54 !important;
        border: 1px solid #087f72 !important;
        box-shadow: none !important;
    }
    [data-testid="stFileUploader"] section button p,
    [data-testid="stFileUploader"] section button span { color: #075e54 !important; }
    .action-card {
        display:flex; align-items:center; gap:13px; min-height:78px;
        padding:14px 15px; margin-bottom:9px; border-radius:15px;
        border:1px solid #dce9e3; background:#ffffff;
        box-shadow:0 5px 16px rgba(27,67,54,.045);
    }
    .action-icon {
        display:flex; flex:0 0 42px; width:42px; height:42px;
        align-items:center; justify-content:center; border-radius:13px;
        color:#075e54; background:#e1f4ed; font-size:21px;
    }
    .action-title { color:#183532; font-weight:800; font-size:.95rem; line-height:1.25; }
    .action-description { color:#617873; font-size:.78rem; line-height:1.4; margin-top:4px; }
    .sidebar-guide-icon {
        display:inline-flex; width:31px; height:31px; align-items:center; justify-content:center;
        border-radius:10px; background:rgba(188,232,217,.13); margin-right:8px;
        font-size:16px; vertical-align:middle;
    }
    [data-testid="stFileUploader"] { background:rgba(255,255,255,.78); border:1.5px dashed #8bbdad; padding:10px; border-radius:15px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-radius:12px; overflow:hidden; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="textarea"] > div { border-radius:10px; background:#fff; }
    [data-testid="stExpander"] { border-radius:13px; border-color:#dce9e3; background:rgba(255,255,255,.78); }
    hr { border-color:#dce9e3; }
    .small-muted { color:#617873; font-size:.84rem; }
    .footer { margin-top:32px; padding:18px; text-align:center; color:#617873; font-size:.82rem; border-top:1px solid #dce9e3; }

    /* SIDEBAR CONTRAST FIX: Streamlit headings inherit the main canvas color by default.
       Explicitly style every common text element so no sidebar text disappears on dark teal. */
    [data-testid="stSidebar"] { color:#f3fff9 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 { color:#f3fff9 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] b,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color:#e0f0e8 !important; }
    [data-testid="stSidebar"] .glass-note,
    [data-testid="stSidebar"] .glass-note b,
    [data-testid="stSidebar"] .glass-note strong,
    [data-testid="stSidebar"] .glass-note span { color:#183532 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] { color:#f3fff9 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button { border-color:#8bcab7 !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section { background:rgba(255,255,255,.06); }
    [data-testid="stSidebar"] hr { border-color:rgba(224,240,232,.28) !important; }
    [data-testid="stSidebar"] [data-testid="stAlert"] { color:#183532 !important; }
    [data-testid="stSidebar"] [data-testid="stAlert"] p,
    [data-testid="stSidebar"] [data-testid="stAlert"] span { color:#183532 !important; }
    [data-testid="stSidebar"] div[data-baseweb="select"] *,
    [data-testid="stSidebar"] div[data-baseweb="input"] * { color:#183532; }
    /* Keep the dashboard guide readable even if Streamlit theme defaults change. */
    .sidebar-guide { display:grid; gap:11px; font-size:.88rem; color:#e0f0e8; }
    .sidebar-guide-item { padding:8px 10px; border:1px solid rgba(224,240,232,.16); border-radius:11px; background:rgba(255,255,255,.055); }
    .sidebar-guide-title { color:#ffffff !important; font-weight:750; }
    .sidebar-guide-desc { color:#c8e8dc !important; font-size:.76rem; line-height:1.4; }
    /* Sidebar uploader's Browse/Clear buttons need dark text on their white surface. */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button {
        background:#ffffff !important; color:#075e54 !important; border-color:#8bcab7 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button p,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section button span {
        color:#075e54 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- HELPERS -----------------------------
def section(title: str, icon: str):
    st.markdown(
        f'<div class="section-heading"><span class="bubble">{icon}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, foot: str, accent: str):
    st.markdown(
        f"""
        <div class="kpi" style="--accent:{accent}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    used = {}
    cols = []
    for col in df.columns:
        name = str(col).strip() or "Unnamed"
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0
        cols.append(name)
    df.columns = cols
    return df


def read_csv_bytes(file_bytes: bytes):
    """Read CSV files with common encodings and delimiter detection."""
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            text = file_bytes.decode(encoding)
            try:
                df = pd.read_csv(StringIO(text), sep=None, engine="python")
            except Exception:
                df = pd.read_csv(StringIO(text))
            if df.shape[1] <= 1 and "\t" in text:
                df = pd.read_csv(StringIO(text), sep="\t")
            if df.shape[1] <= 1 and ";" in text:
                df = pd.read_csv(StringIO(text), sep=";")
            return df, encoding
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not read this CSV file. Details: {last_error}")


def load_uploaded_dataset(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    detected_encoding = None

    if name.endswith(".csv"):
        df, detected_encoding = read_csv_bytes(data)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(data))
    elif name.endswith(".json"):
        parsed = json.loads(data.decode("utf-8-sig"))
        if isinstance(parsed, dict):
            if isinstance(parsed.get("data"), list):
                parsed = parsed["data"]
            elif isinstance(parsed.get("records"), list):
                parsed = parsed["records"]
        if isinstance(parsed, list):
            df = pd.json_normalize(parsed)
        elif isinstance(parsed, dict):
            df = pd.json_normalize(parsed)
        else:
            raise ValueError("JSON must contain an object or a list of records.")
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, XLS, or JSON.")

    if df.empty:
        raise ValueError("The uploaded file contains no data.")
    df = normalize_columns(df)
    return df, detected_encoding


def make_sample_data():
    rng = np.random.default_rng(17)
    n = 250
    categories = np.array(["Technology", "Furniture", "Office Supplies"])
    regions = np.array(["West", "East", "Central", "South"])
    sales = np.round(rng.lognormal(mean=4.7, sigma=0.8, size=n), 2)
    discount = rng.choice([0, 0.05, 0.1, 0.2, 0.3], size=n)
    profit = np.round(sales * rng.normal(0.15, 0.18, size=n) - sales * discount * 0.3, 2)
    return pd.DataFrame(
        {
            "Order Date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "Category": rng.choice(categories, size=n),
            "Region": rng.choice(regions, size=n),
            "Sales": sales,
            "Profit": profit,
            "Quantity": rng.integers(1, 9, size=n),
            "Discount": discount,
            "Customer Segment": rng.choice(["Consumer", "Corporate", "Home Office"], size=n),
        }
    )


def missing_count(df):
    return int(df.isna().sum().sum())


def format_number(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def make_report(df_original, df_cleaned, df_filtered, filename):
    lines = [
        "INSIGHTAI - DATA ANALYSIS REPORT",
        "=" * 42,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source file: {filename}",
        "",
        "DATASET OVERVIEW",
        f"Original rows: {len(df_original):,}",
        f"Original columns: {len(df_original.columns):,}",
        f"Missing cells in original data: {missing_count(df_original):,}",
        f"Duplicate rows in original data: {int(df_original.duplicated().sum()):,}",
        "",
        "CLEANING SUMMARY",
        f"Rows after cleaning: {len(df_cleaned):,}",
        f"Missing cells after cleaning: {missing_count(df_cleaned):,}",
        f"Rows in current filtered view: {len(df_filtered):,}",
        "",
        "NUMERIC SUMMARY",
    ]
    numeric = df_filtered.select_dtypes(include=np.number)
    if numeric.empty:
        lines.append("No numeric columns are available in the filtered data.")
    else:
        lines.append(numeric.describe().round(3).to_string())
    lines += [
        "",
        "CATEGORICAL SUMMARY",
    ]
    cat_cols = df_filtered.select_dtypes(exclude=np.number).columns
    if len(cat_cols) == 0:
        lines.append("No categorical columns are available.")
    else:
        for col in list(cat_cols)[:12]:
            mode = df_filtered[col].mode(dropna=True)
            if not mode.empty:
                lines.append(f"{col}: most common value = {mode.iloc[0]!r}")
    lines += [
        "",
        "NOTE",
        "Insights are generated using descriptive statistics and simple rules.",
        "Correlation does not establish causation.",
    ]
    return "\n".join(lines)


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:8px 3px 17px 3px">
            <div style="display:flex;align-items:center;gap:10px">
              <div style="display:flex;align-items:center;justify-content:center;width:43px;height:43px;border-radius:14px;background:#087f72;color:#fff;font-size:23px">✦</div>
              <div><div style="font-family:Manrope,sans-serif;font-size:1.45rem;font-weight:850;color:#f3fff9">InsightAI</div>
              <div style="color:#c8e8dc;font-size:.78rem;letter-spacing:.08em">DATA INTELLIGENCE</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🧭 Workspace")
    st.caption("Start by importing a dataset.")
    st.markdown('<div class="action-card"><div class="action-icon">📤</div><div><div class="action-title">Upload dataset</div><div class="action-description">CSV, Excel, or JSON · up to 200 MB</div></div></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json"],
        help="Supported formats: CSV, Excel, and JSON.",
    )
    st.markdown(
        """
        <div class="glass-note" style="font-size:.8rem">
        🔒 <b>Privacy note</b><br>
        Your uploaded data is processed in this app session. Avoid uploading confidential data to a public deployment.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("### 🗺️ Dashboard guide")
    st.markdown("""
    <div class="sidebar-guide">
      <div class="sidebar-guide-item"><span class="sidebar-guide-icon">📊</span><span class="sidebar-guide-title">Overview</span><br><span class="sidebar-guide-desc">Dataset summary and preview</span></div>
      <div class="sidebar-guide-item"><span class="sidebar-guide-icon">🧹</span><span class="sidebar-guide-title">Clean data</span><br><span class="sidebar-guide-desc">Duplicates and missing values</span></div>
      <div class="sidebar-guide-item"><span class="sidebar-guide-icon">🔎</span><span class="sidebar-guide-title">Explore</span><br><span class="sidebar-guide-desc">Filters and visualizations</span></div>
      <div class="sidebar-guide-item"><span class="sidebar-guide-icon">💡</span><span class="sidebar-guide-title">Insights</span><br><span class="sidebar-guide-desc">Statistics and correlations</span></div>
      <div class="sidebar-guide-item"><span class="sidebar-guide-icon">📤</span><span class="sidebar-guide-title">Export & download</span><br><span class="sidebar-guide-desc">Cleaned data and reports</span></div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------- LOAD DATA -----------------------------
if uploaded_file is not None:
    try:
        df_original, detected_encoding = load_uploaded_dataset(uploaded_file)
        filename = uploaded_file.name
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
        st.success("Dataset loaded successfully.")
        if detected_encoding:
            st.caption(f"CSV encoding detected: `{detected_encoding}`")
    except Exception as exc:
        st.error(f"Could not load the file: {exc}")
        st.stop()
else:
    df_original = make_sample_data()
    filename = "InsightAI_Sample_Data.csv"
    detected_encoding = None
    st.info("Showing sample data. Upload your own CSV, Excel, or JSON file using the sidebar to begin.")
    sample_csv = df_original.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥  Download sample dataset",
        data=sample_csv,
        file_name="InsightAI_Sample_Data.csv",
        mime="text/csv",
    )

df_original = normalize_columns(df_original)
df_working = df_original.copy()


# ----------------------------- HERO -----------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-tag">✦ DATA INTELLIGENCE STUDIO</div>
        <h1>InsightAI</h1>
        <p>Turn raw data into clear stories, discover patterns, and explore what matters.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# File summary pill
col_file1, col_file2, col_file3 = st.columns([2.1, 1, 1])
with col_file1:
    st.markdown(
        f'<div class="glass-note">📁 <b>Active dataset</b><br><span class="small-muted">{filename}</span></div>',
        unsafe_allow_html=True,
    )
with col_file2:
    st.markdown(
        f'<div class="glass-note">📐 <b>Rows</b><br><span class="small-muted">{len(df_original):,}</span></div>',
        unsafe_allow_html=True,
    )
with col_file3:
    st.markdown(
        f'<div class="glass-note">🧩 <b>Columns</b><br><span class="small-muted">{len(df_original.columns):,}</span></div>',
        unsafe_allow_html=True,
    )


# ----------------------------- 1. OVERVIEW -----------------------------
section("Dataset Overview", "📊")
total_rows, total_cols = df_original.shape
missing_original = missing_count(df_original)
duplicates_original = int(df_original.duplicated().sum())

k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    kpi_card("TOTAL ROWS", f"{total_rows:,}", "Records in your dataset", "#087f72")
with k2:
    kpi_card("TOTAL COLUMNS", f"{total_cols:,}", "Available data fields", "#346c83")
with k3:
    kpi_card("MISSING CELLS", f"{missing_original:,}", "Empty values detected", "#e78a67")
with k4:
    kpi_card("DUPLICATE ROWS", f"{duplicates_original:,}", "Repeated full records", "#d7a84b")

tab_preview, tab_columns, tab_stats = st.tabs(
    ["🧾 Data Preview", "🧬 Column Information", "📈 Descriptive Statistics"]
)
with tab_preview:
    st.caption("A quick look at the records in your uploaded dataset.")
    st.dataframe(df_original.head(100), use_container_width=True, height=340)
with tab_columns:
    info_df = pd.DataFrame(
        {
            "Column": df_original.columns,
            "Data Type": [str(dtype) for dtype in df_original.dtypes],
            "Non-Null Values": [int(df_original[col].notna().sum()) for col in df_original.columns],
            "Missing Values": [int(df_original[col].isna().sum()) for col in df_original.columns],
            "Unique Values": [int(df_original[col].nunique(dropna=True)) for col in df_original.columns],
        }
    )
    st.dataframe(info_df, use_container_width=True, height=340)
with tab_stats:
    describe_df = df_original.describe(include="all").transpose().reset_index(names="Column")
    st.dataframe(describe_df, use_container_width=True, height=340)


# ----------------------------- 2. CLEANING -----------------------------
section("Data Cleaning", "🧹")
st.markdown(
    '<div class="small-muted">Choose how to prepare the data. The original dataset stays unchanged for comparison.</div>',
    unsafe_allow_html=True,
)
clean_col1, clean_col2 = st.columns(2)
with clean_col1:
    remove_duplicates = st.checkbox("Remove duplicate rows", value=False)
with clean_col2:
    fill_missing = st.checkbox("Fill missing values", value=False)

df_cleaned = df_original.copy()
duplicates_removed = 0
missing_before_clean = missing_count(df_cleaned)

if remove_duplicates:
    before = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = before - len(df_cleaned)

if fill_missing:
    for col in df_cleaned.columns:
        if df_cleaned[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                median = df_cleaned[col].median()
                df_cleaned[col] = df_cleaned[col].fillna(median)
            else:
                mode = df_cleaned[col].mode(dropna=True)
                replacement = mode.iloc[0] if not mode.empty else "Unknown"
                df_cleaned[col] = df_cleaned[col].fillna(replacement)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Rows after cleaning", f"{len(df_cleaned):,}", delta=f"{len(df_cleaned)-len(df_original):+,} rows")
with c2:
    st.metric("Duplicates removed", f"{duplicates_removed:,}")
with c3:
    st.metric("Missing cells remaining", f"{missing_count(df_cleaned):,}", delta=f"{missing_count(df_cleaned)-missing_before_clean:+,} cells")

with st.expander("🔍 Preview cleaned dataset"):
    st.dataframe(df_cleaned.head(100), use_container_width=True, height=260)


# ----------------------------- 3. FILTERS -----------------------------
section("Interactive Data Explorer", "🧭")
st.caption("Use filters to focus on specific groups or ranges. Charts and insights below follow the filtered data.")
df_filtered = df_cleaned.copy()
filter_cols = st.multiselect(
    "Choose columns to filter",
    options=list(df_cleaned.columns),
    default=[],
    help="Choose one or more columns, then set filter values below.",
)

if filter_cols:
    filter_grid = st.columns(min(3, len(filter_cols)))
    for idx, col in enumerate(filter_cols):
        container = filter_grid[idx % len(filter_grid)]
        with container:
            series = df_cleaned[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1:
                min_value = float(series.min())
                max_value = float(series.max())
                selected_range = st.slider(
                    f"{col} range",
                    min_value=min_value,
                    max_value=max_value,
                    value=(min_value, max_value),
                    key=f"filter_{col}",
                )
                df_filtered = df_filtered[
                    df_filtered[col].between(selected_range[0], selected_range[1]) |
                    df_filtered[col].isna()
                ]
            else:
                values = series.dropna().unique().tolist()
                if len(values) > 100:
                    values = series.value_counts().head(100).index.tolist()
                selected_values = st.multiselect(
                    f"{col} values",
                    options=values,
                    default=values,
                    key=f"filter_cat_{col}",
                )
                if selected_values:
                    df_filtered = df_filtered[df_filtered[col].isin(selected_values) | df_filtered[col].isna()]
                else:
                    df_filtered = df_filtered.iloc[0:0]

f1, f2, f3 = st.columns(3)
with f1:
    st.metric("Filtered rows", f"{len(df_filtered):,}")
with f2:
    st.metric("Rows excluded", f"{len(df_cleaned)-len(df_filtered):,}")
with f3:
    st.metric("Columns available", f"{len(df_filtered.columns):,}")


# ----------------------------- 4. VISUALIZATIONS -----------------------------
section("Interactive Visualizations", "🎨")
if df_filtered.empty:
    st.warning("No records match the current filters. Adjust your filter selections to see charts.")
else:
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    all_cols = df_filtered.columns.tolist()
    non_numeric_cols = df_filtered.select_dtypes(exclude=np.number).columns.tolist()

    chart_col1, chart_col2, chart_col3 = st.columns([1.2, 1, 1])
    with chart_col1:
        chart_type = st.selectbox(
            "Choose a chart type",
            [
                "Correlation Heatmap",
                "Histogram",
                "Bar Chart",
                "Scatter Plot",
                "Line Chart",
                "Box Plot",
                "Pie Chart",
                "Area Chart",
                "Treemap",
            ],
        )
    with chart_col2:
        default_x = numeric_cols[0] if numeric_cols else all_cols[0]
        x_col = st.selectbox("X-axis / Category", all_cols, index=all_cols.index(default_x))
    with chart_col3:
        default_y = numeric_cols[1] if len(numeric_cols) > 1 else (numeric_cols[0] if numeric_cols else all_cols[0])
        y_col = st.selectbox("Y-axis / Value", all_cols, index=all_cols.index(default_y))

    palette = ["#087F72", "#43A58E", "#E78A67", "#D7A84B", "#346C83", "#8BBEAA", "#C76F52", "#6F9E9A"]
    fig = None

    try:
        if chart_type == "Correlation Heatmap":
            if len(numeric_cols) < 2:
                st.info("A correlation heatmap needs at least two numeric columns.")
            else:
                corr = df_filtered[numeric_cols].corr(numeric_only=True)
                fig = px.imshow(
                    corr,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale=["#e0f2e9", "#8bcab2", "#279783", "#123d39"],
                    title="How numeric variables move together",
                )
        elif chart_type == "Histogram":
            if pd.api.types.is_numeric_dtype(df_filtered[x_col]):
                fig = px.histogram(
                    df_filtered, x=x_col, color=non_numeric_cols[0] if non_numeric_cols else None,
                    marginal="box", color_discrete_sequence=palette,
                    title=f"Distribution of {x_col}",
                )
            else:
                counts = df_filtered[x_col].astype(str).value_counts().head(30).reset_index()
                counts.columns = [x_col, "Count"]
                fig = px.bar(counts, x=x_col, y="Count", color="Count", color_continuous_scale="Tealgrn",
                             title=f"Frequency of {x_col}")
        elif chart_type == "Bar Chart":
            if pd.api.types.is_numeric_dtype(df_filtered[x_col]):
                bar_data = df_filtered.groupby(x_col, dropna=False)[y_col].mean(numeric_only=True).reset_index() if pd.api.types.is_numeric_dtype(df_filtered[y_col]) else None
                if bar_data is not None:
                    fig = px.bar(bar_data.head(60), x=x_col, y=y_col, color=y_col, color_continuous_scale=["#e0f2e9", "#8bcab2", "#279783", "#123d39"],
                                 title=f"Average {y_col} by {x_col}")
                else:
                    fig = px.histogram(df_filtered, x=x_col, color_discrete_sequence=palette, title=f"Count by {x_col}")
            elif pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                bar_data = df_filtered.groupby(x_col, dropna=False)[y_col].mean().reset_index().sort_values(y_col, ascending=False).head(40)
                fig = px.bar(bar_data, x=x_col, y=y_col, color=y_col, color_continuous_scale="Tealgrn",
                             title=f"Average {y_col} by {x_col}")
            else:
                bar_data = df_filtered[x_col].astype(str).value_counts().head(30).reset_index()
                bar_data.columns = [x_col, "Count"]
                fig = px.bar(bar_data, x=x_col, y="Count", color="Count", color_continuous_scale=["#e0f2e9", "#8bcab2", "#279783", "#123d39"],
                             title=f"Record count by {x_col}")
        elif chart_type == "Scatter Plot":
            if pd.api.types.is_numeric_dtype(df_filtered[x_col]) and pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                color_col = non_numeric_cols[0] if non_numeric_cols else None
                fig = px.scatter(
                    df_filtered, x=x_col, y=y_col, color=color_col,
                    color_discrete_sequence=palette, opacity=0.75,
                    trendline="ols" if len(df_filtered) > 2 else None,
                    title=f"{y_col} vs {x_col}",
                )
            else:
                st.info("Choose numeric columns for both axes to create a scatter plot.")
        elif chart_type == "Line Chart":
            if pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                line_data = df_filtered[[x_col, y_col]].dropna().copy()
                if not pd.api.types.is_numeric_dtype(line_data[x_col]):
                    line_data = line_data.groupby(x_col, as_index=False)[y_col].mean()
                else:
                    line_data = line_data.sort_values(x_col)
                fig = px.line(line_data.head(1000), x=x_col, y=y_col, markers=True,
                              color_discrete_sequence=palette, title=f"{y_col} across {x_col}")
            else:
                st.info("Choose a numeric column for the Y-axis to create a line chart.")
        elif chart_type == "Box Plot":
            fig = px.box(
                df_filtered, x=x_col if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else None,
                y=y_col if pd.api.types.is_numeric_dtype(df_filtered[y_col]) else None,
                color=x_col if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else None,
                color_discrete_sequence=palette, points="outliers",
                title=f"Spread and outliers: {y_col}",
            )
            if fig is not None and not fig.data:
                fig = None
        elif chart_type == "Pie Chart":
            pie_col = x_col if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else (non_numeric_cols[0] if non_numeric_cols else x_col)
            pie_data = df_filtered[pie_col].astype(str).value_counts().head(12).reset_index()
            pie_data.columns = [pie_col, "Count"]
            fig = px.pie(pie_data, names=pie_col, values="Count", hole=0.42,
                         color_discrete_sequence=palette, title=f"Share by {pie_col}")
        elif chart_type == "Area Chart":
            if pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                area_data = df_filtered[[x_col, y_col]].dropna().copy()
                if not pd.api.types.is_numeric_dtype(area_data[x_col]):
                    area_data = area_data.groupby(x_col, as_index=False)[y_col].mean()
                else:
                    area_data = area_data.sort_values(x_col)
                fig = px.area(area_data.head(1000), x=x_col, y=y_col, color_discrete_sequence=palette,
                              title=f"{y_col} across {x_col}")
            else:
                st.info("Choose a numeric Y-axis for an area chart.")
        elif chart_type == "Treemap":
            category = x_col if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else (non_numeric_cols[0] if non_numeric_cols else None)
            value = y_col if pd.api.types.is_numeric_dtype(df_filtered[y_col]) else None
            if category and value:
                tree_data = df_filtered.groupby(category, dropna=False)[value].sum().reset_index().sort_values(value, ascending=False).head(40)
                fig = px.treemap(tree_data, path=[category], values=value, color=value,
                                 color_continuous_scale=["#d8f3e5", "#83cbb1", "#2c9886", "#123d39"],
                                 title=f"{value} contribution by {category}")
            else:
                st.info("Treemap needs a category column and a numeric value column.")

        if fig is not None:
            # Consistent styling and descriptive axes for every chart type.
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FBFDFB",
                font=dict(family="DM Sans, sans-serif", size=12, color="#405D56"),
                title=dict(font=dict(family="Manrope, sans-serif", size=19, color="#183532"), x=0.02, xanchor="left"),
                margin=dict(l=48, r=28, t=72, b=68),
                height=500,
                colorway=palette,
                legend_title_font_color="#405D56",
                hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#B8D9CC", font=dict(color="#183532", size=12)),
            )
            if chart_type == "Correlation Heatmap":
                fig.update_xaxes(title_text="Numeric features", tickangle=-35, automargin=True)
                fig.update_yaxes(title_text="Numeric features", automargin=True)
            elif chart_type == "Histogram":
                fig.update_xaxes(title_text=x_col, automargin=True, tickangle=-25 if df_filtered[x_col].nunique(dropna=True) > 12 else 0)
                fig.update_yaxes(title_text="Number of records", automargin=True, rangemode="tozero")
            elif chart_type == "Bar Chart":
                if pd.api.types.is_numeric_dtype(df_filtered[x_col]) and pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                    fig.update_xaxes(title_text=x_col, automargin=True)
                    fig.update_yaxes(title_text=f"Average {y_col}", automargin=True, rangemode="tozero")
                elif pd.api.types.is_numeric_dtype(df_filtered[y_col]):
                    fig.update_xaxes(title_text=x_col, automargin=True, tickangle=-30 if df_filtered[x_col].nunique(dropna=True) > 8 else 0)
                    fig.update_yaxes(title_text=f"Average {y_col}", automargin=True, rangemode="tozero")
                else:
                    fig.update_xaxes(title_text=x_col, automargin=True, tickangle=-30 if df_filtered[x_col].nunique(dropna=True) > 8 else 0)
                    fig.update_yaxes(title_text="Number of records", automargin=True, rangemode="tozero")
            elif chart_type == "Scatter Plot":
                fig.update_xaxes(title_text=x_col, automargin=True)
                fig.update_yaxes(title_text=y_col, automargin=True)
            elif chart_type == "Line Chart":
                fig.update_xaxes(title_text=x_col, automargin=True, tickangle=-25 if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else 0)
                fig.update_yaxes(title_text=y_col, automargin=True)
            elif chart_type == "Box Plot":
                x_title = x_col if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else None
                y_title = y_col if pd.api.types.is_numeric_dtype(df_filtered[y_col]) else None
                fig.update_xaxes(title_text=x_title, automargin=True, tickangle=-25 if x_title and df_filtered[x_col].nunique(dropna=True) > 8 else 0)
                fig.update_yaxes(title_text=y_title, automargin=True)
            elif chart_type == "Pie Chart":
                fig.update_layout(margin=dict(l=20, r=20, t=72, b=25))
            elif chart_type == "Area Chart":
                fig.update_xaxes(title_text=x_col, automargin=True, tickangle=-25 if not pd.api.types.is_numeric_dtype(df_filtered[x_col]) else 0)
                fig.update_yaxes(title_text=y_col, automargin=True)
            elif chart_type == "Treemap":
                fig.update_layout(margin=dict(l=18, r=18, t=72, b=18))
            fig.update_xaxes(
                gridcolor="#EEEAFB", zerolinecolor="#DDD7F5", linecolor="#D9D3F0",
                tickfont=dict(size=10, color="#77739B"), title_font=dict(size=12, color="#39345F"),
                automargin=True,
            )
            fig.update_yaxes(
                gridcolor="#EEEAFB", zerolinecolor="#DDD7F5", linecolor="#D9D3F0",
                tickfont=dict(size=10, color="#77739B"), title_font=dict(size=12, color="#39345F"),
                automargin=True,
            )
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type in ["Correlation Heatmap", "Scatter Plot", "Line Chart", "Area Chart", "Treemap"]:
            pass
    except Exception as exc:
        st.warning(f"That chart could not be created with the selected columns. Try another chart or different axes. Details: {exc}")


# ----------------------------- 5. AUTOMATED INSIGHTS -----------------------------
section("Automated Data Insights", "💡")
st.markdown(
    '<div class="small-muted">These findings are calculated from the filtered dataset using statistical rules. They are not generated by a connected AI language model.</div>',
    unsafe_allow_html=True,
)

if df_filtered.empty:
    st.info("There are no filtered records to analyze.")
else:
    st.markdown(
        f'<div class="glass-note">🔎 The current view contains <b>{len(df_filtered):,} rows</b> and <b>{len(df_filtered.columns):,} columns</b>.</div>',
        unsafe_allow_html=True,
    )
    insights = []
    insights.append(f"Missing cells: {missing_count(df_filtered):,}.")
    insights.append(f"Duplicate rows: {int(df_filtered.duplicated().sum()):,}.")

    numeric_df = df_filtered.select_dtypes(include=np.number)
    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().transpose()
        for col in list(numeric_df.columns)[:8]:
            vals = numeric_df[col].dropna()
            if vals.empty:
                continue
            insights.append(
                f"**{col}:** average {vals.mean():,.2f}; median {vals.median():,.2f}; "
                f"minimum {vals.min():,.2f}; maximum {vals.max():,.2f}."
            )
        if len(numeric_df.columns) >= 2 and len(df_filtered) > 1:
            corr = numeric_df.corr(numeric_only=True)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            stack = upper.stack().dropna()
            if not stack.empty:
                pair = stack.abs().idxmax()
                coefficient = float(stack.loc[pair])
                insights.append(
                    f"The strongest absolute correlation is between **{pair[0]}** and **{pair[1]}** "
                    f"(r = {coefficient:.2f}). Correlation does not establish causation."
                )

    categorical_cols = df_filtered.select_dtypes(exclude=np.number).columns.tolist()
    for col in categorical_cols[:6]:
        mode = df_filtered[col].mode(dropna=True)
        if not mode.empty:
            count = int((df_filtered[col] == mode.iloc[0]).sum())
            insights.append(f"**{col}:** most frequent value is `{mode.iloc[0]}` ({count:,} rows).")

    left, right = st.columns([1.2, 1])
    with left:
        for item in insights:
            st.markdown(f"- {item}")
    with right:
        if len(numeric_df.columns) >= 1:
            chosen_num = st.selectbox("Quick distribution view", numeric_df.columns.tolist(), key="quick_distribution")
            dist = px.histogram(
                df_filtered, x=chosen_num, nbins=25, marginal="box",
                color_discrete_sequence=["#087f72"],
                title=f"Distribution: {chosen_num}",
            )
            dist.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.0)",
                plot_bgcolor="rgba(255,255,255,0.25)",
                font=dict(family="DM Sans", color="#34345f"),
                title=dict(font=dict(family="Manrope, sans-serif", size=18, color="#183532"), x=0.02),
                height=380,
                margin=dict(l=45, r=20, t=62, b=58),
            )
            dist.update_xaxes(title_text=chosen_num, gridcolor="#EEEAFB", tickfont=dict(size=10, color="#77739B"), automargin=True)
            dist.update_yaxes(title_text="Number of records", gridcolor="#EEEAFB", tickfont=dict(size=10, color="#77739B"), automargin=True)
            st.plotly_chart(dist, use_container_width=True)
        else:
            st.info("No numeric columns are available for a distribution chart.")


# ----------------------------- 6. EXPORT -----------------------------
section("Export Your Analysis", "📤")
st.markdown(
    """
    <div class="action-card">
      <div class="action-icon">📥</div>
      <div>
        <div class="action-title">Download your results</div>
        <div class="action-description">Choose a format below. Files are generated from the current cleaned dataset and analysis.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
report_text = make_report(df_original, df_cleaned, df_filtered, filename)
cleaned_csv = df_cleaned.to_csv(index=False).encode("utf-8-sig")
report_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>InsightAI Analysis Report</title>
<style>
body{{font-family:Arial,sans-serif;background:#f2f0ff;color:#292651;padding:32px}}
main{{max-width:1000px;margin:auto;background:white;border-radius:20px;padding:28px;box-shadow:0 10px 30px #d8d2f5}}
h1{{color:#6955df}} pre{{white-space:pre-wrap;font-family:Consolas,monospace;line-height:1.5}}
</style></head><body><main><h1>✨ InsightAI Analysis Report</h1><pre>{report_text}</pre></main></body></html>"""

export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    st.download_button(
        "📥  Download cleaned CSV",
        data=cleaned_csv,
        file_name="insightai_cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_col2:
    st.download_button(
        "📝  Download text report",
        data=report_text.encode("utf-8"),
        file_name="insightai_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
with export_col3:
    st.download_button(
        "🌐  Download HTML report",
        data=report_html.encode("utf-8"),
        file_name="insightai_analysis_report.html",
        mime="text/html",
        use_container_width=True,
    )

st.markdown(
    """
    <div class="footer">
        ✨ <b>InsightAI</b> &nbsp;·&nbsp; Built with Python, Pandas, NumPy, Plotly & Streamlit
        <br><span class="small-muted">Colorful data exploration, made approachable.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
