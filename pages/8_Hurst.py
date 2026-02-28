# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Excel Input
# ------------------------------------------------------------
# What this app does:
# 1) Upload Excel/CSV, select numeric column -> series
# 2) Choose a block size n
# 3) Split into blocks, compute per-block metrics table:
#    mean, std, mean(cumsum), min/max(cumsum), R, R/S
# 4) Choose a list of n values (linear or log spaced)
# 5) Compute R/S(n) as the average R/S across blocks for each n
# 6) Plot:
#    - Raw: R/S vs n
#    - Log-log: log(R/S) vs log(n) + fitted line (slope = H)

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# 0) PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="TEC • Hurst (R/S)", page_icon="📈", layout="wide")


# ----------------------------
# 1) TEC-STYLE (clean, dark)
# ----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#0b1220;
  --panel:rgba(255,255,255,.04);
  --stroke:rgba(255,255,255,.10);
  --text:rgba(255,255,255,.92);
  --muted:rgba(229,231,235,.66);
}
html, body, [class*="css"]{ background:var(--bg) !important; }
.block-container{ max-width:1250px; padding-top:1.2rem; padding-bottom:2rem; }
.card{
  background:var(--panel);
  border:1px solid var(--stroke);
  border-radius:18px;
  padding:14px 14px;
}
.title{ font-size:1.55rem; font-weight:800; color:var(--text); margin:0 0 .25rem 0; }
.sub{ color:var(--muted); margin:0 0 1rem 0; line-height:1.35rem; }
.small{ color:var(--muted); font-size:.88rem; }
hr{ border:none; border-top:1px solid var(--stroke); margin:1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="title">📈 Hurst Exponent — R/S Analysis</div>
<div class="sub">
Upload an Excel/CSV, compute <b>R/S</b> per block, then estimate <b>H</b> from the slope in the log–log plot.
</div>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 2) CORE BUILDING BLOCKS
# ----------------------------
def dividir_em_blocos(lista, tamanho):
    """Split into complete blocks (pure slicing)."""
    return [
        lista[i : i + tamanho]
        for i in range(0, len(lista), tamanho)
        if len(lista[i : i + tamanho]) == tamanho
    ]


def metricas_do_bloco(bloco, ddof=1):
    """
    Metrics for one block, matching the method you built:
    - mean
    - std (ddof)
    - centered cumulative sum
    - mean(cumsum)
    - min/max(cumsum)
    - R = max-min
    - RS = R/std
    """
    x = np.array(bloco, dtype=float)

    mu = float(np.mean(x))
    S = float(np.std(x, ddof=ddof))

    centered = x - mu
    Y = np.cumsum(centered)

    Y_mean = float(np.mean(Y))
    Y_min = float(np.min(Y))
    Y_max = float(np.max(Y))
    R = float(Y_max - Y_min)

    RS = float(R / S) if S != 0 else np.nan

    return {
        "mean": mu,
        "std": S,
        "mean_cumsum": Y_mean,
        "min_cumsum": Y_min,
        "max_cumsum": Y_max,
        "R": R,
        "RS": RS,
    }


def tabela_metricas_blocos(serie: List[float], n: int, ddof: int = 1) -> pd.DataFrame:
    blocos = dividir_em_blocos(serie, n)
    rows = []
    for idx, bloco in enumerate(blocos, start=1):
        m = metricas_do_bloco(bloco, ddof=ddof)
        m["block"] = idx
        m["n"] = n
        rows.append(m)
    return pd.DataFrame(rows)


def rs_medio_por_n(serie: List[float], n: int, ddof: int = 1) -> float:
    """
    R/S(n) = mean of RS across all complete blocks of size n.
    """
    blocos = dividir_em_blocos(serie, n)
    if len(blocos) == 0:
        return np.nan

    rs_vals = []
    for bloco in blocos:
        rs = metricas_do_bloco(bloco, ddof=ddof)["RS"]
        if np.isfinite(rs) and rs > 0:
            rs_vals.append(rs)

    if len(rs_vals) == 0:
        return np.nan
    return float(np.mean(rs_vals))


def choose_ns(min_n: int, max_n: int, points: int, spacing: str) -> List[int]:
    min_n = int(max(2, min_n))
    max_n = int(max(min_n + 1, max_n))
    points = int(max(8, points))

    if spacing == "log":
        ns = np.unique(
            np.round(np.logspace(np.log10(min_n), np.log10(max_n), points)).astype(int)
        )
    else:
        ns = np.unique(np.linspace(min_n, max_n, points).round().astype(int))

    ns = ns[(ns >= 2) & (ns <= max_n)]
    return ns.tolist()


def fit_hurst(ns: np.ndarray, rs_vals: np.ndarray) -> Tuple[float, float]:
    """
    Fit log(R/S) = a + H log(n) using natural log.
    Returns (H, C) where C = exp(a).
    """
    ns = np.asarray(ns, dtype=float)
    rs_vals = np.asarray(rs_vals, dtype=float)

    mask = (ns > 0) & (rs_vals > 0) & np.isfinite(rs_vals)
    x = ns[mask]
    y = rs_vals[mask]

    if x.size < 2:
        return np.nan, np.nan

    lx = np.log(x)
    ly = np.log(y)

    H, a = np.polyfit(lx, ly, 1)
    C = float(np.exp(a))
    return float(H), C


# ----------------------------
# 3) INPUT (Excel/CSV)
# ----------------------------
with st.container():
    c1, c2 = st.columns([1.15, 0.85], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1) Upload and select column")

        uploaded = st.file_uploader("Upload (.xlsx / .xls / .csv)", type=["xlsx", "xls", "csv"])

        df = None
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    xls = pd.ExcelFile(uploaded)
                    if len(xls.sheet_names) > 1:
                        sheet = st.selectbox("Sheet", xls.sheet_names, index=0)
                        df = pd.read_excel(xls, sheet_name=sheet)
                    else:
                        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            except Exception as e:
                st.error(f"Could not read file: {e}")

        if df is not None:
            col = st.selectbox("Numeric column", list(df.columns), index=0)
            ddof = st.selectbox("Std ddof", [1, 0], index=0)

            serie = (
                pd.to_numeric(df[col], errors="coerce")
                .dropna()
                .astype(float)
                .tolist()
            )

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown('<div class="small">Cleaning: non-numeric → NaN → dropped.</div>', unsafe_allow_html=True)
            st.write("Series length:", len(serie))
            st.write("Preview:", serie[:10])

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2) Block + scale settings")

        n_block = st.number_input("Block size for the per-block metrics table (n)", min_value=2, value=100, step=1)

        spacing = st.selectbox("n spacing for R/S(n)", ["log", "linear"], index=0)
        points = st.slider("How many n values", min_value=8, max_value=60, value=25)

        min_n = st.number_input("min n for scaling", min_value=2, value=16, step=1)
        max_n = st.number_input("max n for scaling", min_value=3, value=512, step=1)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            '<div class="small">Tip: keep max n well below the series length. Log spacing usually stabilizes H.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


if uploaded is None or df is None:
    st.info("Upload a file to start.")
    st.stop()

if len(serie) < 50:
    st.warning("Need at least ~50 numeric points to compute meaningful scaling.")
    st.stop()

# Cap max_n to something reasonable
max_reasonable = max(3, len(serie) // 2)
if max_n > max_reasonable:
    st.warning(f"max n capped to {max_reasonable} (≈ half of your series length).")
    max_n = max_reasonable

if n_block > len(serie):
    st.warning("Block size for the table is bigger than the series length. Reduce n.")
    st.stop()


# ----------------------------
# 4) TABLE: per-block metrics for a single n_block
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"3) Per-block metrics table (block size n = {int(n_block)})")

df_blocks = tabela_metricas_blocos(serie, int(n_block), ddof=int(ddof))

if df_blocks.empty:
    st.warning("No complete blocks with this n. Try a smaller n.")
    st.stop()

st.dataframe(df_blocks, use_container_width=True, height=320)

st.markdown(
    '<div class="small">Each row is one block. RS = R / std, where R is computed from the centered cumulative sum.</div>',
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ----------------------------
# 5) SCALING: compute R/S(n) for multiple n, estimate H
# ----------------------------
ns = choose_ns(int(min_n), int(max_n), int(points), spacing)

rs_vals = np.array([rs_medio_por_n(serie, n, ddof=int(ddof)) for n in ns], dtype=float)
ns_arr = np.array(ns, dtype=float)

H, C = fit_hurst(ns_arr, rs_vals)

valid = np.isfinite(rs_vals) & (rs_vals > 0)
valid_count = int(np.sum(valid))

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("4) Scaling results")

st.write(f"Valid R/S points: {valid_count}/{len(ns)}")
st.write(f"H (slope) ≈ {H:.6f}")
st.write(f"C (scale) ≈ {C:.6g}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ----------------------------
# 6) PLOTS
# ----------------------------
ns_v = ns_arr[valid]
rs_v = rs_vals[valid]

cA, cB = st.columns(2, gap="large")

# Plot 1: raw R/S vs n
with cA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("5) Raw plot: R/S(n) vs n")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ns_v, y=rs_v, mode="markers+lines", name="R/S(n)"))

    fig1.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="n (window size)", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(title="R/S(n)", gridcolor="rgba(255,255,255,.10)"),
        font=dict(color="rgba(255,255,255,.90)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="small">This curve typically looks smooth and increasing.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# Plot 2: log-log (entropized)
with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("6) Log–log plot: log(R/S) vs log(n)")

    lx = np.log(ns_v)
    ly = np.log(rs_v)

    # Fit line in log space for visualization
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
        xaxis=dict(title="log(n)", gridcolor="rgba(255,255,255,.10)"),
        yaxis=dict(title="log(R/S(n))", gridcolor="rgba(255,255,255,.10)"),
        font=dict(color="rgba(255,255,255,.90)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f'<div class="small">In log–log, power laws become lines. Slope = H ≈ {H_plot:.6f}.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# 7) OPTIONAL: scaling table export
# ----------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("7) Scaling table (n, R/S(n))")

df_scaling = pd.DataFrame({"n": ns_arr, "RS_mean": rs_vals})
st.dataframe(df_scaling, use_container_width=True, height=260)

csv_bytes = df_scaling.to_csv(index=False).encode("utf-8")
st.download_button("Download scaling table CSV", data=csv_bytes, file_name="rs_scaling_table.csv", mime="text/csv")

st.markdown("</div>", unsafe_allow_html=True)
