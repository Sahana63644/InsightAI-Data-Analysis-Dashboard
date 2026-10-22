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
        --ink: #20224a;
        --muted: #666b91;
        --purple: #7257e8;
        --blue: #3289e8;
        --teal: #11a99a;
        --pink: #e45ba6;
        --gold: #e6a23c;
        --panel: rgba(255,255,255,0.78);
        --line: rgba(107, 95, 190, 0.18);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, rgba(92, 222, 210, 0.44), transparent 29%),
            radial-gradient(ellipse at 91% 7%, rgba(176, 142, 255, 0.42), transparent 30%),
            radial-gradient(ellipse at 78% 80%, rgba(255, 190, 218, 0.30), transparent 31%),
            linear-gradient(135deg, #edfafa 0%, #f1efff 48%, #fff0f6 100%);
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(239, 244, 255, 0.55);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(165deg, #e0faf5 0%, #e7e4ff 55%, #fce8f4 100%);
        border-right: 1px solid rgba(107, 95, 190, 0.18);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #272451 !important;
        letter-spacing: -0.035em;
    }
    h1 { font-weight: 800 !important; }
    h2, h3 { font-weight: 750 !important; }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: #35375f;
    }

    .hero {
        padding: 30px 32px;
        border-radius: 28px;
        margin: 2px 0 24px 0;
        color: white;
        background: linear-gradient(120deg, #4f46c9 0%, #7257e8 42%, #298bcf 76%, #12a99b 100%);
        box-shadow: 0 18px 44px rgba(86, 75, 190, 0.22);
        position: relative;
        overflow: hidden;
    }
    .hero:after {
        content: "✦";
        position: absolute;
        right: 35px;
        top: 8px;
        font-size: 105px;
        opacity: 0.16;
        color: white;
    }
    .hero h1 { color: white !important; margin: 0 0 8px 0; font-size: clamp(2rem, 3.5vw, 3.15rem); }
    .hero p { color: rgba(255,255,255,0.90) !important; margin: 0; font-size: 1.04rem; }
    .hero-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: #f1fffe;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 6px 11px;
        border-radius: 100px;
        margin-bottom: 13px;
    }

    .section-heading {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 25px 0 12px 0;
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        font-size: 1.45rem;
        color: #292651;
    }
    .section-heading .bubble {
        display: inline-flex;
        width: 37px;
        height: 37px;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        color: white;
        background: linear-gradient(135deg, #7257e8, #3289e8);
        box-shadow: 0 5px 14px rgba(94, 81, 202, 0.20);
    }

    .kpi {
        border-radius: 20px;
        padding: 19px 19px 17px 19px;
        min-height: 133px;
        border: 1px solid rgba(255,255,255,0.92);
        box-shadow: 0 9px 24px rgba(67, 65, 135, 0.09);
        background: rgba(255,255,255,0.76);
        backdrop-filter: blur(14px);
        position: relative;
        overflow: hidden;
    }
    .kpi:after {
        content: "";
        position: absolute;
        width: 85px;
        height: 85px;
        border-radius: 50%;
        right: -24px;
        top: -24px;
        background: var(--accent);
        opacity: 0.13;
    }
    .kpi-label {
        font-size: 0.83rem;
        font-weight: 700;
        color: #62678d;
        margin-bottom: 10px;
    }
    .kpi-value {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.55rem, 2.2vw, 2.2rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.12;
        color: var(--accent);
    }
    .kpi-foot { font-size: 0.75rem; color: #7c80a1; margin-top: 9px; }

    .glass-note {
        padding: 15px 18px;
        border-radius: 16px;
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 7px 20px rgba(67,65,135,0.06);
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(255,255,255,0.9);
        padding: 14px 17px;
        border-radius: 16px;
        box-shadow: 0 7px 20px rgba(67,65,135,0.06);
    }
    [data-testid="stMetricLabel"] { color: #65698e !important; }
    [data-testid="stMetricValue"] { color: #292651 !important; }

    div[data-testid="stTabs"] button {
        border-radius: 12px 12px 0 0;
        font-weight: 700;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5d4ad3;
        border-bottom-color: #7257e8;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 0;
        color: white;
        font-weight: 700;
        background: linear-gradient(100deg, #6955df, #3689e7);
        box-shadow: 0 6px 16px rgba(94, 81, 202, 0.18);
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 9px 20px rgba(94, 81, 202, 0.27);
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.58);
        border: 1.5px dashed rgba(104, 85, 218, 0.42);
        padding: 10px;
        border-radius: 18px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border-radius: 14px;
        overflow: hidden;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        border-radius: 11px;
        background-color: rgba(255,255,255,0.82);
    }
    [data-testid="stExpander"] {
        border-radius: 14px;
        border-color: rgba(107,95,190,0.20);
        background: rgba(255,255,255,0.50);
    }
    hr { border-color: rgba(107,95,190,0.16); }
    .small-muted { color: #737798; font-size: 0.84rem; }
    .footer {
        margin-top: 34px;
        padding: 18px;
        text-align: center;
        color: #696d91;
        font-size: 0.82rem;
        border-top: 1px solid rgba(107,95,190,0.16);
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
            <div style="font-size:2.4rem">✨</div>
            <div style="font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:850;color:#30285f">InsightAI</div>
            <div style="color:#67658f;font-size:.86rem">Your data, made clearer.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 🧭 Dashboard Controls")
    st.caption("Upload a dataset to begin exploring.")
    uploaded_file = st.file_uploader(
        "Upload your dataset",
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
    st.markdown("### 🎨 Dashboard guide")
    st.caption("① Overview  ·  ② Clean  ·  ③ Filter  ·  ④ Visualize  ·  ⑤ Insights  ·  ⑥ Export")


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
        "⬇️ Download sample dataset",
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
    kpi_card("TOTAL ROWS", f"{total_rows:,}", "Records in your dataset", "#7257e8")
with k2:
    kpi_card("TOTAL COLUMNS", f"{total_cols:,}", "Available data fields", "#2388df")
with k3:
    kpi_card("MISSING CELLS", f"{missing_original:,}", "Empty values detected", "#df5b9f")
with k4:
    kpi_card("DUPLICATE ROWS", f"{duplicates_original:,}", "Repeated full records", "#0d9e8f")

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

    palette = ["#7257E8", "#3289E8", "#11A99A", "#E45BA6", "#F0AE42", "#7A9E35", "#EF725B", "#4ECDC4"]
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
                    color_continuous_scale=["#fce7f3", "#c4b5fd", "#6366f1", "#164e9a"],
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
                    fig = px.bar(bar_data.head(60), x=x_col, y=y_col, color=y_col, color_continuous_scale="Purples",
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
                fig = px.bar(bar_data, x=x_col, y="Count", color="Count", color_continuous_scale="Purples",
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
                                 color_continuous_scale=["#99f6e4", "#818cf8", "#c084fc", "#f472b6"],
                                 title=f"{value} contribution by {category}")
            else:
                st.info("Treemap needs a category column and a numeric value column.")

        if fig is not None:
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.0)",
                plot_bgcolor="rgba(255,255,255,0.30)",
                font=dict(family="DM Sans, sans-serif", color="#34345f"),
                title_font=dict(family="Manrope, sans-serif", size=19, color="#292651"),
                margin=dict(l=18, r=18, t=65, b=25),
                height=490,
                colorway=palette,
                legend_title_font_color="#34345f",
            )
            fig.update_xaxes(gridcolor="rgba(105,95,170,0.12)", zerolinecolor="rgba(105,95,170,0.12)")
            fig.update_yaxes(gridcolor="rgba(105,95,170,0.12)", zerolinecolor="rgba(105,95,170,0.12)")
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
                color_discrete_sequence=["#7257e8"],
                title=f"Distribution: {chosen_num}",
            )
            dist.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0.0)",
                plot_bgcolor="rgba(255,255,255,0.25)",
                font=dict(family="DM Sans", color="#34345f"),
                title_font_color="#292651",
                height=360,
                margin=dict(l=12, r=12, t=55, b=15),
            )
            dist.update_xaxes(gridcolor="rgba(105,95,170,0.12)")
            dist.update_yaxes(gridcolor="rgba(105,95,170,0.12)")
            st.plotly_chart(dist, use_container_width=True)
        else:
            st.info("No numeric columns are available for a distribution chart.")


# ----------------------------- 6. EXPORT -----------------------------
section("Export Your Analysis", "📦")
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
        "⬇️ Download cleaned CSV",
        data=cleaned_csv,
        file_name="insightai_cleaned_data.csv",
        mime="text/csv",
        use_container_width=True,
    )
with export_col2:
    st.download_button(
        "📄 Download text report",
        data=report_text.encode("utf-8"),
        file_name="insightai_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
with export_col3:
    st.download_button(
        "🌈 Download HTML report",
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
