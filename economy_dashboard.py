
# Unified Synthetic Economy Dashboard (All-in-one, fully linked)
# --------------------------------------------------------------
# Run locally:
#   pip install -r requirements.txt
#   streamlit run economy_dashboard.py

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Unified Economy Dashboard", page_icon="📊", layout="wide")
st.title("📊 Unified Synthetic Economy Dashboard")
st.caption("One page • Shared controls • Plotly + Altair + scikit-learn • All visuals update together")

# --------------------
# Data generation
# --------------------
@st.cache_data
def generate_synthetic_data(seed: int = 42, months: int = 48) -> pd.DataFrame:
    import math
    rng = np.random.default_rng(seed)

    end = pd.Timestamp.today().normalize().replace(day=1)
    dates = pd.date_range(end=end, periods=months, freq="MS")

    sectors = [
        "Manufacturing", "Technology", "Finance", "Healthcare",
        "Energy", "Retail", "Transportation", "Construction"
    ]
    countries = ["USA","CAN","MEX","BRA","GBR","DEU","FRA","ITA","ESP","CHN","IND","JPN","AUS","ZAF"]

    sector_base = {s: rng.normal(200, 30) for s in sectors}
    sector_trend = {s: rng.uniform(0.001, 0.01) for s in sectors}
    sector_margin_mu = {
        "Manufacturing": 0.12, "Technology": 0.22, "Finance": 0.25, "Healthcare": 0.18,
        "Energy": 0.15, "Retail": 0.10, "Transportation": 0.08, "Construction": 0.09
    }
    country_mult = {c: rng.normal(1.0, 0.12) for c in countries}

    rows = []
    for t_idx, date in enumerate(dates):
        season = 1.0 + 0.1 * math.sin(2*math.pi * (date.month/12.0))
        for s in sectors:
            for c in countries:
                base = sector_base[s] * country_mult[c] * season
                rev = base * ((1 + sector_trend[s]) ** t_idx) * rng.normal(1.0, 0.15)
                employees = max(5, rev / 5.0 + rng.normal(0, 10))
                capex = max(0, rev * rng.uniform(0.05, 0.25))
                margin = np.clip(rng.normal(sector_margin_mu[s], 0.04), 0.01, 0.45)
                sentiment = np.clip((margin - 0.10) * 4 + rng.normal(0, 0.4), -1, 1)
                rows.append({
                    "date": date,
                    "sector": s,
                    "country_iso3": c,
                    "revenue": float(rev),
                    "employees": float(employees),
                    "capex": float(capex),
                    "profit_margin": float(margin),
                    "sentiment": float(sentiment),
                })

    df = pd.DataFrame(rows).sort_values("date")
    df["revenue_yoy"] = (
        df.sort_values(["sector", "country_iso3", "date"])
          .groupby(["sector", "country_iso3"])["revenue"]
          .pct_change(periods=12)
    )
    return df

# --------------------
# Sidebar (shared controls for ALL visuals)
# --------------------
with st.sidebar:
    st.header("⚙️ Global Controls")
    seed = st.number_input("Random seed", 0, 10_000, 42, step=1)
    months = st.slider("Months of history", 24, 120, 48, step=12)

df = generate_synthetic_data(seed=seed, months=months)

with st.sidebar:
    dmin, dmax = df["date"].min(), df["date"].max()
    date_range = st.slider(
        "Date range",
        min_value=dmin.to_pydatetime(),
        max_value=dmax.to_pydatetime(),
        value=(dmin.to_pydatetime(), dmax.to_pydatetime()),
    )
    sectors = sorted(df["sector"].unique().tolist())
    selected_sectors = st.multiselect("Sectors", sectors, default=sectors)
    countries = sorted(df["country_iso3"].unique().tolist())
    selected_countries = st.multiselect("Countries (ISO-3)", countries, default=countries[:8])

    st.divider()
    metric_for_ts = st.selectbox("Time series metric", ["revenue","employees","capex","profit_margin","sentiment"], 0)
    ts_agg = st.radio("Time series aggregation", ["sum","mean"], index=0, horizontal=True)
    split_by_sector = st.checkbox("Color by sector in time series", value=True)

    st.divider()
    k_clusters = st.slider("K for clustering", 2, 8, 4, step=1)
    feature_scale = st.checkbox("Standardize features (clusters)", True)

    st.divider()
    metric_for_map = st.selectbox("Map metric", ["revenue","employees","capex","profit_margin","sentiment","revenue_yoy"], 0)
    metric_for_bar = st.selectbox("Bar metric", ["revenue","employees","capex"], 0)
    top_n = st.slider("Top N sectors for bar", 3, 8, 8, step=1)

# Filtered frame
mask = (
    (df["date"] >= pd.Timestamp(date_range[0])) &
    (df["date"] <= pd.Timestamp(date_range[1])) &
    (df["sector"].isin(selected_sectors)) &
    (df["country_iso3"].isin(selected_countries))
)
dff = df.loc[mask].copy()

# --------------------
# KPIs
# --------------------
def kpis(data: pd.DataFrame):
    latest = data["date"].max()
    prev_m = latest - pd.offsets.MonthBegin(1)

    cur = data[data["date"] == latest]
    prev = data[data["date"] == prev_m]

    def mom(cur_val, prev_val):
        if prev_val is None or pd.isna(prev_val) or prev_val == 0:
            return None
        return (cur_val - prev_val) / prev_val * 100

    rev_cur = cur["revenue"].sum()
    rev_prev = prev["revenue"].sum() if len(prev) else np.nan
    rev_delta = mom(rev_cur, rev_prev)

    emp_cur = cur["employees"].sum()
    emp_prev = prev["employees"].sum() if len(prev) else np.nan
    emp_delta = mom(emp_cur, emp_prev)

    mar_cur = cur["profit_margin"].mean()
    mar_prev = prev["profit_margin"].mean() if len(prev) else np.nan
    mar_delta = None if pd.isna(mar_prev) else (mar_cur - mar_prev) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Monthly Revenue", f"${rev_cur:,.0f}", None if rev_delta is None else f"{rev_delta:+.1f}% MoM")
    c2.metric("Latest Monthly Employees", f"{emp_cur:,.0f}", None if emp_delta is None else f"{emp_delta:+.1f}% MoM")
    c3.metric("Avg Profit Margin", f"{mar_cur:.1%}", None if mar_delta is None else f"{mar_delta:+.1f} pts")

kpis(dff)
st.markdown("---")

# --------------------
# LAYOUT: 2 rows
# Row 1: Time series (wide) + Cluster (wide)
# Row 2: Map (wide) + Bar + Pie
# --------------------

# ---- Time Series (Plotly) ----
def time_series_chart(data: pd.DataFrame) -> None:
    group_cols = ["date"]
    if split_by_sector:
        group_cols.append("sector")
    if ts_agg == "sum":
        ts_df = data.groupby(group_cols, as_index=False)[metric_for_ts].sum()
    else:
        ts_df = data.groupby(group_cols, as_index=False)[metric_for_ts].mean()

    fig = px.line(
        ts_df, x="date", y=metric_for_ts,
        color=("sector" if split_by_sector else None),
        markers=True,
        labels={"date":"Date", metric_for_ts: metric_for_ts.replace("_"," ").title()}
    )
    fig.update_layout(height=420, hovermode="x unified", margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- Clusters (Plotly) ----
def compute_sector_features(data: pd.DataFrame) -> pd.DataFrame:
    g = data.groupby(["sector","date"], as_index=False).agg(
        revenue=("revenue","sum"),
        employees=("employees","sum"),
        capex=("capex","sum"),
        margin=("profit_margin","mean"),
        sentiment=("sentiment","mean"),
    )
    feats = []
    for s, sdf in g.groupby("sector"):
        sdf = sdf.sort_values("date")
        rev0 = sdf["revenue"].iloc[0]
        rev1 = sdf["revenue"].iloc[-1]
        n_months = max(1, (sdf["date"].iloc[-1].to_period("M") - sdf["date"].iloc[0].to_period("M")).n)
        cagr = (rev1 / rev0) ** (12.0 / n_months) - 1.0 if rev0 > 0 else np.nan
        mom = sdf["revenue"].pct_change()
        vol = float(mom.std()) if len(mom) > 1 else 0.0
        feats.append({
            "sector": s,
            "revenue_mean": sdf["revenue"].mean(),
            "employees_mean": sdf["employees"].mean(),
            "capex_ratio": (sdf["capex"].sum() / max(1e-6, sdf["revenue"].sum())),
            "avg_margin": sdf["margin"].mean(),
            "avg_sentiment": sdf["sentiment"].mean(),
            "revenue_cagr": cagr,
            "rev_volatility": vol,
        })
    return pd.DataFrame(feats)

def cluster_chart(data: pd.DataFrame) -> None:
    feat_df = compute_sector_features(data).dropna().reset_index(drop=True)
    if len(feat_df) < 2:
        st.info("Not enough sectors in the current filter to cluster.")
        return

    X = feat_df.drop(columns=["sector"]).values
    if feature_scale:
        X = StandardScaler().fit_transform(X)

    k_use = min(max(2, k_clusters), max(2, len(feat_df)-1)) if len(feat_df) > 2 else 2
    kmeans = KMeans(n_clusters=k_use, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(X)

    XY = PCA(n_components=2, random_state=0).fit_transform(X)
    plot_df = pd.DataFrame({"PC1": XY[:,0], "PC2": XY[:,1], "sector": feat_df["sector"], "cluster": labels.astype(int)})
    fig_scatter = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", text="sector", title="KMeans clusters (PCA view)")
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---- Map (Plotly Choropleth) ----
def map_chart(data: pd.DataFrame) -> None:
    agg_map = data.groupby("country_iso3", as_index=False).agg(
        value=(metric_for_map, "mean" if metric_for_map in ["profit_margin","sentiment","revenue_yoy"] else "sum")
    )
    fig_map = px.choropleth(
        agg_map,
        locations="country_iso3",
        color="value",
        projection="natural earth",
        color_continuous_scale="Viridis",
        title=f"{metric_for_map.replace('_',' ').title()} by Country"
    )
    fig_map.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ---- Bar (Plotly) & Pie (Altair) ----
def bar_and_pie(data: pd.DataFrame) -> None:
    left, right = st.columns([1,1])

    agg_bar = (
        data.groupby("sector", as_index=False)[metric_for_bar]
            .sum()
            .sort_values(metric_for_bar, ascending=False)
            .head(top_n)
    )
    fig_bar = px.bar(
        agg_bar, x="sector", y=metric_for_bar, text_auto=".2s",
        title=f"Top {top_n} Sectors by {metric_for_bar.title()}",
        labels={metric_for_bar: metric_for_bar.title()}
    )
    fig_bar.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
    left.plotly_chart(fig_bar, use_container_width=True)

    agg_pie = data.groupby("sector", as_index=False).agg(total_revenue=("revenue","sum"))
    agg_pie["share"] = agg_pie["total_revenue"] / agg_pie["total_revenue"].sum()
    pie = (
        alt.Chart(agg_pie)
        .mark_arc()
        .encode(
            theta=alt.Theta(field="share", type="quantitative"),
            color=alt.Color(field="sector", type="nominal"),
            tooltip=[alt.Tooltip("sector:N"), alt.Tooltip("total_revenue:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")]
        )
        .properties(height=420, title="Sector Revenue Share")
    )
    right.altair_chart(pie, use_container_width=True)

# --------------------
# RENDER (all linked)
# --------------------
r1c1, r1c2 = st.columns([1,1])
with r1c1:
    st.subheader("📈 Time Series")
    time_series_chart(dff)
with r1c2:
    st.subheader("🧩 Clusters")
    cluster_chart(dff)

st.subheader("🗺️ Map + 🎯 Breakdowns")
r2c1, r2c2 = st.columns([1.2, 1])
with r2c1:
    map_chart(dff)
with r2c2:
    bar_and_pie(dff)

# --------------------
# Data & Export
# --------------------
st.markdown("---")
st.subheader("📎 Data Sample & Download")
st.dataframe(dff.sample(min(100, len(dff))), use_container_width=True, height=420)
st.download_button(
    "Download filtered CSV",
    data=dff.to_csv(index=False).encode("utf-8"),
    file_name="synthetic_economy_filtered.csv",
    mime="text/csv",
    use_container_width=True
)
