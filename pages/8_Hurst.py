# streamlit_app.py
# TEC — The Everything Calculator
# Hurst Exponent (R/S Analysis) — Excel Input
# ------------------------------------------------------------
# Features
# - Upload .xlsx/.xls or .csv
# - Choose sheet + column
# - Clean/convert to numeric, drop NaNs
# - Compute R/S exactly in the style we built:
#   1) mean center
#   2) cumulative sum
#   3) range (max-min)
#   4) std (ddof selectable; default ddof=1)
#   5) R/S = R / S
# - Plot raw (non-log) R/S vs n
# - Plot "entropized" (log-log) + linear fit => H (slope)
# - TEC-style layout: dark UI, cards, clear explanations

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# 0) PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="TEC • Hurst (R/S)",
    page_icon="📈",
    layout="wide",
)


# ----------------------------
# 1) TEC STYLE (dark, clean)
# ----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#0b1220;
  --panel:#0f1a2e;
  --panel2:#0d172a;
  --stroke:rgba(255,255,255,.10);
  --muted:rgba(229,231,235,.65);
  --text:rgba(255,255,255,.92);
  --accent:#6ee7ff;
  --accent2:#a78bfa;
  --good:#34d399;
  --warn:#fbbf24;
  --bad:#fb7185;
}

html, body, [class*="css"] {
  background: var(--bg) !important;
}

.block-container{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1250px;
}

.tec-title{
  font-size: 1.6rem;
  font-weight: 750;
  color: var(--text);
  margin: 0 0 .25rem 0;
  letter-spacing: .2px;
}
.tec-sub{
  color: var(--muted);
  margin: 0 0 1rem 0;
}

.card{
  background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 14px;
}

.card h3{
  margin: 0 0 .4rem 0;
  color: var(--text);
  font-size: 1.05rem;
  font-weight: 700;
}
.card p, .card li{
  color: var(--muted);
  line-height: 1.35rem;
  font-size: .95rem;
}

.kpi{
  display:flex;
  gap:12px;
  flex-wrap: wrap;
}
.kpi-box{
  flex: 1;
  min-width: 180px;
  background: rgba(255,255,255,.03);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 12px 12px;
}
.kpi-label{
  color: var(--muted);
  font-size: .82rem;
}
.kpi-value{
  color: var(--text);
  font-size: 1.25rem;
  font-weight: 750;
  margin-top: 2px;
}

hr{
  border: none;
  border-top: 1px solid var(--stroke);
  margin: 1rem 0;
}
.small-note{
  color: var(--muted);
  font-size: .86rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="tec-title">📈 Hurst Exponent — R/S Analysis</div>
<div class="tec-sub">
Upload an Excel/CSV, pick a numeric column, compute R/S across scales <b>n</b>, then estimate <b>H</b> from the log–log slope.
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 2) CORE MATH (R/S)
# ----------------------------
def rs_stat(serie: np.ndarray, ddof: int = 1) -> float:
    """
    R/S for ONE window (exactly the method we built):
    - mean center
    - cumulative sum
    - range = max-min
    - std = np.std(serie, ddof=ddof)
    - return R/S
    """
    x = np.asarray(serie, dtype=float)
    if x.size < 2:
        return np.nan

    mu = np.mean(x)
    y = np.cumsum(x - mu)  # accumulated centered series
    R = float(np.max(y) - np.min(y))

    S = float(np.std(x, ddof=ddof))
    if S == 0.0 or not np.isfinite(S):
        return np.nan

    return R / S


def rs_by_blocks(serie: np.ndarray, n: int, ddof: int = 1) -> float:
    """
    Compute R/S(n) by splitting series into non-overlapping blocks of length n,
    computing R/S per block, and averaging.

    This is more stable than prefix-only and still uses the exact same core rs_stat.
    """
    x = np.asarray(serie, dtype=float)
    if n <= 1 or n > x.size:
        return np.nan

    k = x.size // n
    if k < 1:
        return np.nan

    blocks = x[: k * n].reshape(k, n)
    vals = np.array([rs_stat(blocks[i], ddof=ddof) for i in range(k)], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return np.nan
    return float(np.mean(vals))


def choose_ns(n_points: int, min_n: int, max_n: int, how: str) -> np.ndarray:
    min_n = int(max(2, min_n))
    max_n = int(max(min_n + 1, max_n))
    n_points = int(max(8, n_points))

    if how == "log":
        ns = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), n_points)).astype(int))
    else:
        ns = np.unique(np.linspace(min_n, max_n, n_points).round().astype(int))

    ns = ns[(ns >= 2) & (ns <= max_n)]
    return ns


def fit_hurst(ns: np.ndarray, rs_vals: np.ndarray, log_base: str = "e") -> Tuple[float, float]:
    """
    Fit log(R/S) = a + H log(n)
    Returns (H, C) where C = base^a (or exp(a) if base=e)
    """
    ns = np.asarray(ns, dtype=float)
    rs_vals = np.asarray(rs_vals, dtype=float)

    mask = (ns > 0) & (rs_vals > 0) & np.isfinite(rs_vals)
    x = ns[mask]
    y = rs_vals[mask]

    if x.size < 2:
        return np.nan, np.nan

    if log_base == "2":
        lx = np.log2(x)
        ly = np.log2(y)
        H, a = np.polyfit(lx, ly, 1)
        C = 2 ** a
    elif log_base == "10":
        lx = np.log10(x)
        ly = np.log10(y)
        H, a = np.polyfit(lx, ly, 1)
        C = 10 ** a
    else:
        lx = np.log(x)
        ly = np.log(y)
        H, a = np.polyfit(lx, ly, 1)
        C = math.exp(a)

    return float(H), float(C)


# ----------------------------
# 3) INPUT (Excel/CSV)
# ----------------------------
with st.container():
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown('<div class="card"><h3>1) Data input</h3>', unsafe_allow_html=True)
        st.write("Upload your file and select the numeric column you want to analyze.")

        uploaded = st.file_uploader("Upload (.xlsx / .xls / .csv)", type=["xlsx", "xls", "csv"])

        df = None
        sheet_name = None

        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    xls = pd.ExcelFile(uploaded)
                    if len(xls.sheet_names) > 1:
                        sheet_name = st.selectbox("Sheet", xls.sheet_names, index=0)
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                    else:
                        sheet_name = xls.sheet_names[0]
                        df = pd.read_excel(xls, sheet_name=sheet_name)
            except Exception as e:
                st.error(f"Could not read the file. Details: {e}")

        if df is not None:
            st.caption(f"Loaded: {uploaded.name}" + (f" • sheet: {sheet_name}" if sheet_name else ""))

            col1, col2 = st.columns([0.62, 0.38], gap="small")
            with col1:
                col_choice = st.selectbox("Numeric column", list(df.columns), index=0)
            with col2:
                ddof = st.selectbox("Std ddof", [1, 0], index=0, help="ddof=1 (sample) is the usual choice; ddof=0 is population std.")

            # Convert to numeric + drop NaNs
            series_raw = pd.to_numeric(df[col_choice], errors="coerce").dropna()
            serie = series_raw.to_numpy(dtype=float)

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('<div class="small-note">Cleaning rules: non-numeric → NaN → dropped.</div>', unsafe_allow_html=True)
            st.write("Rows kept:", int(serie.size))

            if serie.size > 0:
                st.write("Preview:", serie[:10])

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h3>2) R/S scale settings</h3>', unsafe_allow_html=True)
        st.write("Choose the range of window sizes **n** used to compute R/S(n).")

        method = st.selectbox(
            "R/S aggregation",
            ["blocks_mean", "prefix"],
            index=0,
            help=(
                "blocks_mean: split into non-overlapping blocks of size n, compute R/S per block, then average.\n"
                "prefix: compute R/S on the first n points only (more didactic, less stable)."
            ),
        )

        how = st.selectbox("n spacing", ["log", "linear"], index=0)
        log_base = st.selectbox("log base for the fit", ["e", "10", "2"], index=0)

        min_n = st.number_input("min n", min_value=2, value=16, step=1)
        max_n = st.number_input("max n", min_value=3, value=1024, step=1)
        n_points = st.slider("how many n values", min_value=8, max_value=60, value=25)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            """
<div class="small-note">
<b>Tip:</b> For stable H estimates, prefer log-spaced n and keep max n well below the series length.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 4) COMPUTE
# ----------------------------
if uploaded is None or df is None:
    st.info("Upload a file to start.")
    st.stop()

if "serie" not in locals() or serie.size < 50:
    st.warning("Need at least ~50 numeric points to produce meaningful R/S scaling.")
    st.stop()

# Ensure max_n doesn't exceed size too much
max_reasonable = max(3, int(serie.size // 2))
if max_n > max_reasonable:
    st.warning(f"max n capped to {max_reasonable} (≈ half of your series length).")
    max_n = max_reasonable

ns = choose_ns(n_points=n_points, min_n=int(min_n), max_n=int(max_n), how=how)

rs_vals = []
for n in ns:
    if method == "blocks_mean":
        rs_vals.append(rs_by_blocks(serie, int(n), ddof=int(ddof)))
    else:
        # prefix mode
        rs_vals.append(rs_stat(serie[: int(n)], ddof=int(ddof)))

rs_vals = np.array(rs_vals, dtype=float)

H, C = fit_hurst(ns, rs_vals, log_base=log_base)

# KPI panel
st.markdown('<div class="card"><h3>Results</h3>', unsafe_allow_html=True)
st.markdown(
    f"""
<div class="kpi">
  <div class="kpi-box">
    <div class="kpi-label">H (slope in log–log)</div>
    <div class="kpi-value">{H:.6f}</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">C (scale constant)</div>
    <div class="kpi-value">{C:.6g}</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Valid R/S points</div>
    <div class="kpi-value">{int(np.sum(np.isfinite(rs_vals) & (rs_vals>0)))}/{len(ns)}</div>
  </div>
  <div class="kpi-box">
    <div class="kpi-label">Series length</div>
    <div class="kpi-value">{serie.size}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    r"""
<div class="small-note" style="margin-top:.6rem;">
Model: \( \log(R/S) = \log(C) + H\log(n) \). In log–log space, \(H\) is the line slope.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ----------------------------
# 5) PLOTS
# ----------------------------
valid_mask = np.isfinite(rs_vals) & (rs_vals > 0) & np.isfinite(ns) & (ns > 0)
ns_v = ns[valid_mask]
rs_v = rs_vals[valid_mask]

c1, c2 = st.columns(2, gap="large")

# --- Plot 1: raw (non-log) R/S vs n
with c1:
    st.markdown('<div class="card"><h3>Raw scale plot (no log)</h3>', unsafe_allow_html=True)
    st.markdown(
        """
<p>
This is the direct curve \(R/S(n)\) versus window size \(n\). It usually looks like a smooth, increasing curve.
</p>
""",
        unsafe_allow_html=True,
    )

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=ns_v,
            y=rs_v,
            mode="markers+lines",
            name="R/S(n)",
        )
    )
    fig1.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="n (window size)", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        yaxis=dict(title="R/S(n)", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color="rgba(255,255,255,.88)"),
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Plot 2: "entropized" (log-log) + fit
with c2:
    st.markdown('<div class="card"><h3>Entropized plot (log–log) + fit</h3>', unsafe_allow_html=True)
    st.markdown(
        r"""
<p>
In log–log, a power law becomes a line:
\[
\log(R/S) = \log(C) + H\log(n)
\]
So the slope is \(H\).
</p>
""",
        unsafe_allow_html=True,
    )

    # log transform for plot (use natural logs for fit line visualization; slope is invariant to base)
    lx = np.log(ns_v)
    ly = np.log(rs_v)

    # Fit in natural log space for plotting a line (same H)
    H_plot, a_plot = np.polyfit(lx, ly, 1)
    ly_hat = a_plot + H_plot * lx

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lx, y=ly, mode="markers", name="log(R/S)"))
    fig2.add_trace(go.Scatter(x=lx, y=ly_hat, mode="lines", name=f"fit slope ≈ {H_plot:.4f}"))

    fig2.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="log(n)", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        yaxis=dict(title="log(R/S(n))", gridcolor="rgba(255,255,255,.08)", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(color="rgba(255,255,255,.88)"),
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""
<div class="small-note">
Fit (natural log for visualization): slope \(\\approx {H_plot:.6f}\\). Your chosen base for reporting: <b>{log_base}</b>.
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 6) TABLE + EXPORT
# ----------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card"><h3>R/S table</h3>', unsafe_allow_html=True)

out_df = pd.DataFrame({"n": ns, "R/S(n)": rs_vals})
st.dataframe(out_df, use_container_width=True, height=260)

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button("Download R/S table as CSV", data=csv_bytes, file_name="rs_table.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 7) NOTES (TEC-style)
# ----------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card"><h3>How to interpret H</h3>', unsafe_allow_html=True)
st.markdown(
    r"""
<ul>
  <li><b>H ≈ 0.5</b>: compatible with an uncorrelated random process (no long-term memory).</li>
  <li><b>H > 0.5</b>: persistent behavior (moves tend to reinforce).</li>
  <li><b>H < 0.5</b>: anti-persistent behavior (moves tend to revert).</li>
</ul>
<div class="small-note">
Practical note: very small \(n\) and very large \(n\) can add noise/bias. Try log-spaced \(n\) and keep \(n\) comfortably below series length.
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)
