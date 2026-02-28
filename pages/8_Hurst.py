# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Excel Input
# ------------------------------------------------------------
# Autor: Gemini para The Everything Calculator
# ------------------------------------------------------------

from __future__ import annotations
import time
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
# 1) TEC-STYLE (Clean Dark Theme)
# ----------------------------
st.markdown(
    """
<style>
:root{
  --bg:#0e1117;
  --panel:rgba(255,255,255,.045);
  --stroke:rgba(255,255,255,.08);
  --text:rgba(255,255,255,.92);
  --muted:rgba(229,231,235,.60);
  --accent:#FF4B4B;
  --accent2:#1E90FF;
}
.main { background-color: var(--bg); }
.block-container{ max-width:1250px; padding-top:1.5rem; }
.card{
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 1.2rem;
  margin-bottom: 1rem;
}
.title{ font-size:1.8rem; font-weight:800; color:var(--text); margin-bottom: 0.2rem; }
.sub{ color:var(--muted); margin-bottom: 1.5rem; font-size:0.95rem; }
.small-muted{ color:var(--muted); font-size:0.85rem; }
.metric-row { display: flex; gap: 20px; margin-bottom: 10px; }
hr { border:none; border-top:1px solid var(--stroke); margin: 1.5rem 0; }

/* Custom metrics design */
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="title">📈 Hurst Exponent Engine</div>
<div class="sub">Análise de Rescaled Range (R/S) para detecção de persistência e memória em séries temporais.</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 2) CORE ENGINE
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
        "mean": mu,
        "std": S,
        "min_cumsum": np.min(Y),
        "max_cumsum": np.max(Y),
        "R": R,
        "RS": RS,
    }

def rs_medio_por_n(serie: List[float], n: int, ddof: int = 1) -> float:
    blocos = dividir_em_blocos(serie, n)
    if not blocos: return np.nan
    
    rs_vals = [metricas_do_bloco(b, ddof=ddof)["RS"] for b in blocos]
    rs_vals = [r for r in rs_vals if np.isfinite(r) and r > 0]
    
    return float(np.mean(rs_vals)) if rs_vals else np.nan

# ----------------------------
# 3) SIDEBAR CONTROLS
# ----------------------------
st.sidebar.markdown("### 🎛️ Configurações")
uploaded = st.sidebar.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"): df = pd.read_csv(uploaded)
        else: df = pd.read_excel(uploaded)
        
        col_target = st.sidebar.selectbox("Coluna Alvo", df.select_dtypes(include=[np.number]).columns)
        serie = df[col_target].dropna().astype(float).tolist()
    except Exception as e:
        st.sidebar.error(f"Erro no arquivo: {e}")
        st.stop()
else:
    # Demo data
    st.sidebar.info("Utilizando Random Walk para demonstração.")
    serie = np.cumsum(np.random.randn(1000) + 0.02).tolist()

st.sidebar.markdown("---")
n_table = st.sidebar.number_input("Tamanho do bloco p/ Tabela (n)", 2, 5000, 100)
min_n = st.sidebar.slider("Min n (Scaling)", 4, 50, 8)
max_n = st.sidebar.slider("Max n (Scaling)", 100, len(serie)//2, min(512, len(serie)//2))
points = st.sidebar.slider("Pontos na Regressão", 10, 60, 25)
spacing = st.sidebar.radio("Espaçamento n", ["log", "linear"])

# ----------------------------
# 4) COMPUTATION
# ----------------------------
t0 = time.time()

# Tabela detalhada
df_blocks = pd.DataFrame([
    {**metricas_do_bloco(b), "block": i+1} 
    for i, b in enumerate(dividir_em_blocos(serie, n_table))
])

# Scaling para Hurst
if spacing == "log":
    ns = np.unique(np.geomspace(min_n, max_n, points).astype(int))
else:
    ns = np.unique(np.linspace(min_n, max_n, points).astype(int))

rs_results = []
for n in ns:
    val = rs_medio_por_n(serie, n)
    if np.isfinite(val): rs_results.append((n, val))

ns_arr, rs_arr = np.array([x[0] for x in rs_results]), np.array([x[1] for x in rs_results])
H, a = np.polyfit(np.log(ns_arr), np.log(rs_arr), 1)

dt = time.time() - t0

# ----------------------------
# 5) MAIN UI LAYOUT
# ----------------------------
# Metrics Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Hurst Exponent (H)", f"{H:.4f}")
m2.metric("Comportamento", "Persistente" if H > 0.55 else "Anti-persistente" if H < 0.45 else "Random Walk")
m3.metric("Janelas Analisadas", len(ns_arr))
m4.metric("Latency", f"{dt*1000:.1f}ms")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"📋 Métricas por Bloco (n = {n_table})")
st.dataframe(df_blocks, use_container_width=True, height=250, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

tab_viz, tab_data = st.tabs(["📊 Visualização Fractal", "📄 Dados de Escalonamento"])

with tab_viz:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ns_arr, y=rs_arr, mode='lines+markers', name="R/S(n)", marker=dict(color='#1E90FF')))
        fig1.update_layout(title="R/S vs n (Escala Linear)", template="plotly_dark", height=400, 
                          margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Log-Log Plot
        lx, ly = np.log(ns_arr), np.log(rs_arr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=lx, y=ly, mode='markers', name="Logs Reais", marker=dict(color='#FF4B4B')))
        fig2.add_trace(go.Scatter(x=lx, y=a + H*lx, mode='lines', name=f"Fit (Slope={H:.3f})", line=dict(dash='dot', color='white')))
        fig2.update_layout(title="Log-Log Analysis (Slope = H)", template="plotly_dark", height=400,
                          margin=dict(l=20,r=20,t=40,b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df_scaling = pd.DataFrame({"n": ns_arr, "RS_mean": rs_arr, "log_n": np.log(ns_arr), "log_RS": np.log(rs_arr)})
    st.dataframe(df_scaling, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 6) FOOTER
# ----------------------------
st.markdown("<div style='text-align:center; color:rgba(255,255,255,0.2); padding:20px;'>The Everything Calculator — Hurst Engine v2.0</div>", unsafe_allow_html=True)
