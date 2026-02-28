# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Excel Input
# ------------------------------------------------------------
# - Versão Restaurada (Simplicidade e Tabela de Blocos)
# - Correção de CSS para evitar clipping no topo
# - Suporte a CSV/Excel com auto-detecção

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
# 1) TEC-STYLE (CSS REVISADO)
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

/* Esconde o header nativo para evitar o erro de corte */
header[data-testid="stHeader"] { visibility: hidden; height: 0px; }

html, body, [class*="css"]{ background:var(--bg) !important; }

.block-container{ 
    max-width:1250px; 
    padding-top:4rem !important; /* Espaço extra no topo */
    padding-bottom:2rem; 
}

.card{
  background:var(--panel);
  border:1px solid var(--stroke);
  border-radius:18px;
  padding:1.5rem;
  margin-bottom: 1rem;
}

.title{ 
    font-size:1.8rem; 
    font-weight:800; 
    color:var(--text); 
    margin:0 0 .5rem 0; 
    line-height: 1.2;
}

.sub{ color:var(--muted); margin:0 0 1.5rem 0; line-height:1.4rem; }
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
# 2) CORE FUNCTIONS
# ----------------------------
def dividir_em_blocos(lista, tamanho):
    return [lista[i : i + tamanho] for i in range(0, len(lista), tamanho) if len(lista[i : i + tamanho]) == tamanho]

def metricas_do_bloco(bloco, ddof=1):
    x = np.array(bloco, dtype=float)
    mu = float(np.mean(x))
    S = float(np.std(x, ddof=ddof))
    centered = x - mu
    Y = np.cumsum(centered)
    R = float(np.max(Y) - np.min(Y))
    RS = float(R / S) if S != 0 else np.nan
    return {
        "mean": mu, "std": S, "min_cumsum": np.min(Y),
        "max_cumsum": np.max(Y), "R": R, "RS": RS
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
    blocos = dividir_em_blocos(serie, n)
    if not blocos: return np.nan
    rs_vals = [metricas_do_bloco(b, ddof=ddof)["RS"] for b in blocos]
    rs_vals = [r for r in rs_vals if np.isfinite(r) and r > 0]
    return float(np.mean(rs_vals)) if rs_vals else np.nan

def choose_ns(min_n: int, max_n: int, points: int, spacing: str) -> List[int]:
    if spacing == "log":
        ns = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), points)).astype(int))
    else:
        ns = np.unique(np.linspace(min_n, max_n, points).round().astype(int))
    return ns[(ns >= 2)].tolist()

def fit_hurst(ns: np.ndarray, rs_vals: np.ndarray) -> Tuple[float, float]:
    mask = (ns > 0) & (rs_vals > 0) & np.isfinite(rs_vals)
    x, y = np.log10(ns[mask]), np.log10(rs_vals[mask])
    if x.size < 2: return np.nan, np.nan
    H, a = np.polyfit(x, y, 1)
    return float(H), float(10**a)

# ----------------------------
# 3) INPUT
# ----------------------------
with st.container():
    c1, c2 = st.columns([1.2, 0.8], gap="large")

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("1) Data Upload")
        uploaded = st.file_uploader("Upload (.xlsx / .csv)", type=["xlsx", "csv"])
        
        df = None
        if uploaded:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded, sep=None, engine='python')
            else:
                df = pd.read_excel(uploaded)
            
            col = st.selectbox("Numeric column", list(df.select_dtypes(include=[np.number]).columns))
            serie = df[col].dropna().astype(float).tolist()
            st.write(f"Series length: {len(serie)}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("2) Settings")
        n_block = st.number_input("Block size for table (n)", 2, 5000, 100)
        min_n = st.number_input("Min n scaling", 4, 100, 8)
        max_n = st.number_input("Max n scaling", 100, 2000, 512)
        spacing = st.selectbox("Spacing", ["log", "linear"])
        st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.info("Please upload a file to proceed.")
    st.stop()

# ----------------------------
# 4) TABELA DE BLOCOS
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"3) Metrics per Block (n = {n_table if 'n_table' in locals() else n_block})")
df_blocks = tabela_metricas_blocos(serie, int(n_block))
st.dataframe(df_blocks, use_container_width=True, height=250, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# 5) SCALING & REGRESSION
# ----------------------------
ns = choose_ns(int(min_n), int(max_n), 25, spacing)
rs_vals = np.array([rs_medio_por_n(serie, n) for n in ns])
H, C = fit_hurst(np.array(ns), rs_vals)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write(f"**Estimated Hurst Exponent (H):** {H:.6f}")
st.write(f"**Interpretation:** {'Trending' if H > 0.55 else 'Mean Reverting' if H < 0.45 else 'Random Walk'}")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# 6) PLOTS
# ----------------------------
cA, cB = st.columns(2)

valid = np.isfinite(rs_vals) & (rs_vals > 0)
ns_v, rs_v = np.array(ns)[valid], rs_vals[valid]

with cA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig1 = go.Figure(go.Scatter(x=ns_v, y=rs_v, mode="lines+markers", line=dict(color="#1E90FF")))
    fig1.update_layout(title="R/S(n) vs n", template="plotly_dark", height=400, 
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    lx, ly = np.log10(ns_v), np.log10(rs_v)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lx, y=ly, mode="markers", name="Log Data", marker=dict(color="#FF4B4B")))
    fig2.add_trace(go.Scatter(x=lx, y=H*lx + np.log10(C), mode="lines", name="Fit", line=dict(color="white", dash="dot")))
    fig2.update_layout(title="Log-Log Plot", template="plotly_dark", height=400,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:gray; font-size:12px;'>TEC — Hurst Engine</div>", unsafe_allow_html=True)
