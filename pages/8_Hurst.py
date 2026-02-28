# streamlit_app.py
# TEC — Hurst Exponent (R/S) Analysis — Professional Edition
# ------------------------------------------------------------
# - Engine: Hurst / Rescaled Range (R/S) 
# - UI: Design Dark Institucional (T.E.C. Style)
# - Features: Upload Dinâmico, Log-Log Fit, Diagnósticos de Memória Longa

from __future__ import annotations

import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# 0) PAGE CONFIG & THEME
# ----------------------------
st.set_page_config(page_title="TEC • Hurst Exponent", page_icon="📈", layout="wide")

st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --panel: rgba(255,255,255,0.04);
  --border: rgba(255,255,255,0.08);
  --accent: #FF4B4B;
  --text-muted: rgba(229,231,235,0.60);
}

.main { background-color: var(--bg); }
.stMetric {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 15px;
}

.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 20px;
  margin-bottom: 20px;
}

.hr { border-top: 1px solid var(--border); margin: 1.5rem 0; }
.footer { text-align:center; color: var(--text-muted); font-size: 0.85rem; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 1) CORE ENGINE (MATEMÁTICA)
# ----------------------------

def compute_rs_metrics(block: np.ndarray, ddof: int = 1) -> Dict:
    """Calcula a estatística R/S para um único bloco de dados."""
    n = len(block)
    mu = np.mean(block)
    std = np.std(block, ddof=ddof)
    
    # Normalização e Soma Cumulativa Centrada
    centered_sum = np.cumsum(block - mu)
    R = np.max(centered_sum) - np.min(centered_sum)
    
    rs = R / std if std > 0 else 0
    return {"R": R, "S": std, "RS": rs}

@st.cache_data(show_spinner=False)
def hurst_analysis_full(series: np.ndarray, ns: List[int], ddof: int = 1) -> pd.DataFrame:
    """Executa a análise R/S para múltiplos tamanhos de janela."""
    results = []
    for n in ns:
        num_blocks = len(series) // n
        if num_blocks == 0: continue
        
        # Slicing eficiente de blocos
        blocks = series[:num_blocks * n].reshape((num_blocks, n))
        rs_values = []
        
        for i in range(num_blocks):
            m = compute_rs_metrics(blocks[i], ddof)
            if m["RS"] > 0:
                rs_values.append(m["RS"])
        
        if rs_values:
            results.append({
                "n": n,
                "RS_mean": np.mean(rs_values),
                "log_n": np.log(n),
                "log_rs": np.log(np.mean(rs_values))
            })
            
    return pd.DataFrame(results)

# ----------------------------
# 2) UI / SIDEBAR CONTROLS
# ----------------------------
st.title("📈 Hurst Exponent Analysis")
st.caption("Análise de Faixa Reescalada (R/S) para Identificação de Memória Longa")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configurações")
    uploaded = st.file_uploader("Data Source (.csv, .xlsx)", type=["csv", "xlsx"])
    
    if uploaded:
        ext = uploaded.name.split('.')[-1]
        df_raw = pd.read_csv(uploaded) if ext == "csv" else pd.read_excel(uploaded)
        col_target = st.selectbox("Coluna Alvo", df_raw.columns)
        series = pd.to_numeric(df_raw[col_target], errors='coerce').dropna().values
    else:
        st.info("Aguardando upload... Usando dados sintéticos (Random Walk).")
        series = np.cumsum(np.random.randn(2000))
        
    st.markdown("---")
    st.subheader("Parâmetros de Escala")
    min_n = st.number_input("Janela Mínima (n)", value=10, min_value=4)
    max_n = st.number_input("Janela Máxima (n)", value=len(series)//4, min_value=20)
    num_points = st.slider("Pontos de Amostragem", 10, 100, 30)
    spacing = st.radio("Espaçamento de n", ["Logarítmico", "Linear"])
    ddof_val = st.selectbox("Graus de Liberdade (std)", [1, 0], index=0)

# ----------------------------
# 3) PROCESSAMENTO
# ----------------------------
if spacing == "Logarítmico":
    ns = np.unique(np.geomspace(min_n, max_n, num_points).astype(int)).tolist()
else:
    ns = np.unique(np.linspace(min_n, max_n, num_points).astype(int)).tolist()

# Execução da análise
t0 = time.time()
df_results = hurst_analysis_full(series, ns, ddof_val)
t_exec = time.time() - t0

# Regressão Linear para H
if not df_results.empty:
    poly = np.polyfit(df_results["log_n"], df_results["log_rs"], 1)
    H_exponent = poly[0]
    intercept = poly[1]
    df_results["fit"] = np.exp(intercept) * (df_results["n"] ** H_exponent)
else:
    H_exponent = 0

# ----------------------------
# 4) DASHBOARD LAYOUT
# ----------------------------
m1, m2, m3, m4 = st.columns(4)

# Lógica de interpretação do Hurst
def interpret_hurst(h):
    if h > 0.55: return "Persistente (Trend)"
    if h < 0.45: return "Anti-persistente"
    return "Random Walk"

m1.metric("Hurst Exponent (H)", f"{H_exponent:.4f}")
m2.metric("Regime", interpret_hurst(H_exponent))
m3.metric("Pontos Analisados", len(series))
m4.metric("Tempo de Processamento", f"{t_exec:.3f}s")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

tab_viz, tab_data, tab_theory = st.tabs(["📊 Visualização", "📋 Dados Brutos", "📚 Teoria"])

with tab_viz:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Escalonamento R/S")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=df_results["n"], y=df_results["RS_mean"], mode='markers+lines', name="R/S observado", marker=dict(color='#FF4B4B')))
        fig_raw.update_layout(template="plotly_dark", xaxis_title="n", yaxis_title="R/S(n)", margin=dict(l=0,r=0,b=0))
        st.plotly_chart(fig_raw, use_container_width=True)

    with c2:
        st.subheader("Regressão Log-Log")
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(x=df_results["log_n"], y=df_results["log_rs"], mode='markers', name="Dados", marker=dict(color='#1E90FF')))
        fig_log.add_trace(go.Scatter(x=df_results["log_n"], y=np.polyval(poly, df_results["log_n"]), mode='lines', name=f"H = {H_exponent:.4f}", line=dict(dash='dash')))
        fig_log.update_layout(template="plotly_dark", xaxis_title="log(n)", yaxis_title="log(R/S)", margin=dict(l=0,r=0,b=0))
        st.plotly_chart(fig_log, use_container_width=True)

with tab_data:
    st.dataframe(df_results, use_container_width=True)

with tab_theory:
    st.markdown("""
    ### O que é o Expoente de Hurst?
    O expoente de Hurst ($H$) é uma medida de **memória de longo prazo** em séries temporais. 
    Ele quantifica a tendência relativa de uma série temporal de regredir à média ou de se agrupar em uma direção.
    """)
    st.latex(r"E[R(n)/S(n)] = C \cdot n^H")
    st.markdown("""
    - **$H < 0.5$**: Série Anti-persistente (significa que um aumento será seguido por uma queda).
    - **$H = 0.5$**: Random Walk (Movimento Browniano clássico).
    - **$H > 0.5$**: Série Persistente (um aumento no passado indica probabilidade de aumento no futuro).
    """)

st.markdown("<div class='footer'>The Everything Calculator - Unconventional Analysis Group (U.A.G.) • 2026</div>", unsafe_allow_html=True)
