# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Excel Input (Versão Corrigida)
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
# 1) TEC-STYLE (CSS CORRIGIDO PARA EVITAR CORTES)
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
}

/* Ajuste crítico para evitar corte no topo das páginas Streamlit */
.block-container { 
    max-width: 1250px; 
    padding-top: 4rem !important; 
    padding-bottom: 2rem; 
}

.card {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 1.2rem;
  margin-bottom: 1rem;
}

/* Correção de line-height para o título não cortar o topo das letras */
.title { 
    font-size: 2.2rem; 
    font-weight: 800; 
    color: var(--text); 
    margin-bottom: 0.5rem;
    line-height: 1.3; 
    display: flex;
    align-items: center;
    gap: 15px;
}

.sub { 
    color: var(--muted); 
    margin-bottom: 2rem; 
    font-size: 1rem; 
    line-height: 1.4;
}

hr { border:none; border-top:1px solid var(--stroke); margin: 1.5rem 0; }

/* Estilização das Métricas nativas */
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 2) HEADER
# ----------------------------
st.markdown(
    """
<div class="title">
    <span>📈</span> Hurst Exponent Engine
</div>
<div class="sub">
    Análise de Rescaled Range (R/S) para detecção de persistência e memória em séries temporais.
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 3) CORE ENGINE (MATH)
# ----------------------------
def dividir_em_blocos(lista, tamanho):
    """Divide a série em blocos contíguos de tamanho fixo."""
    return [lista[i : i + tamanho] for i in range(0, len(lista), tamanho) if len(lista[i : i + tamanho]) == tamanho]

def metricas_do_bloco(bloco, ddof=1):
    """Calcula R/S para um único bloco de dados."""
    x = np.array(bloco, dtype=float)
    mu = np.mean(x)
    std = np.std(x, ddof=ddof)
    
    # Séries acumulada de desvios (centrada na média)
    Y = np.cumsum(x - mu)
    
    R = np.max(Y) - np.min(Y)
    RS = R / std if std > 0 else np.nan

    return {
        "mean": mu,
        "std": std,
        "min_cumsum": np.min(Y),
        "max_cumsum": np.max(Y),
        "R": R,
        "RS": RS,
    }

def rs_medio_por_n(serie: List[float], n: int, ddof: int = 1) -> float:
    """Calcula a média de R/S para todos os blocos de tamanho n."""
    blocos = dividir_em_blocos(serie, n)
    if not blocos: return np.nan
    
    rs_vals = [metricas_do_bloco(b, ddof=ddof)["RS"] for b in blocos]
    rs_vals = [r for r in rs_vals if np.isfinite(r) and r > 0]
    
    return float(np.mean(rs_vals)) if rs_vals else np.nan

# ----------------------------
# 4) SIDEBAR & INPUT
# ----------------------------
st.sidebar.markdown("### 🎛️ Dados e Escala")
uploaded = st.sidebar.file_uploader("Upload Excel ou CSV", type=["xlsx", "xls", "csv"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"): df = pd.read_csv(uploaded)
        else: df = pd.read_excel(uploaded)
        
        # Selecionar apenas colunas numéricas
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        col_target = st.sidebar.selectbox("Coluna para Análise", num_cols)
        serie = df[col_target].dropna().astype(float).tolist()
    except Exception as e:
        st.sidebar.error(f"Erro ao ler arquivo: {e}")
        st.stop()
else:
    st.sidebar.warning("Aguardando arquivo. Usando dados sintéticos.")
    serie = np.cumsum(np.random.randn(1000) + 0.01).tolist()

st.sidebar.markdown("---")
n_table = st.sidebar.number_input("Tamanho do bloco (Tabela)", 2, 5000, 100)
min_n = st.sidebar.slider("Min n (Escalonamento)", 4, 100, 8)
max_n = st.sidebar.slider("Max n (Escalonamento)", 100, len(serie)//2, min(1024, len(serie)//2))
spacing = st.sidebar.radio("Distribuição de n", ["log", "linear"])

# ----------------------------
# 5) PROCESSAMENTO
# ----------------------------
t0 = time.time()

# 1. Gerar Tabela de Blocos
df_blocks = pd.DataFrame([
    {**metricas_do_bloco(b), "block": i+1} 
    for i, b in enumerate(dividir_em_blocos(serie, n_table))
])

# 2. Cálculo do Expoente de Hurst (Regressão Log-Log)
if spacing == "log":
    ns = np.unique(np.geomspace(min_n, max_n, 25).astype(int))
else:
    ns = np.unique(np.linspace(min_n, max_n, 25).astype(int))

rs_list = []
for n in ns:
    val = rs_medio_por_n(serie, n)
    if np.isfinite(val): rs_list.append((n, val))

ns_arr = np.array([x[0] for x in rs_list])
rs_arr = np.array([x[1] for x in rs_list])

# Ajuste linear: log(R/S) = H*log(n) + log(C)
H, intercept = np.polyfit(np.log(ns_arr), np.log(rs_arr), 1)

dt = time.time() - t0

# ----------------------------
# 6) DASHBOARD
# ----------------------------

# Linha de Métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hurst Exponent (H)", f"{H:.4f}")
c2.metric("Memória", "Persistente" if H > 0.55 else "Anti-persistente" if H < 0.45 else "Aleatória")
c3.metric("Amostras (n)", len(ns_arr))
c4.metric("Processamento", f"{dt*1000:.1f}ms")

# Tabela Detalhada
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"📋 Detalhamento dos Blocos (n = {n_table})")
st.dataframe(df_blocks, use_container_width=True, height=250, hide_index=True)
st.markdown('</div>', unsafe_allow_html=True)

# Gráficos
tab_viz, tab_data = st.tabs(["📊 Gráficos de Escalonamento", "📄 Dados Brutos"])

with tab_viz:
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=ns_arr, y=rs_arr, mode='lines+markers', line=dict(color='#1E90FF')))
        fig1.update_layout(title="R/S vs n (Escala Linear)", template="plotly_dark", height=400,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=20))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        lx, ly = np.log(ns_arr), np.log(rs_arr)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=lx, y=ly, mode='markers', name="Dados", marker=dict(color='#FF4B4B')))
        fig2.add_trace(go.Scatter(x=lx, y=H*lx + intercept, mode='lines', name=f"H={H:.3f}", line=dict(color='white', dash='dot')))
        fig2.update_layout(title="Log-Log Plot (Hurst Slope)", template="plotly_dark", height=400,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Dados utilizados para a regressão linear:")
    df_fit = pd.DataFrame({"n": ns_arr, "R/S Médio": rs_arr, "Log(n)": np.log(ns_arr), "Log(R/S)": np.log(rs_arr)})
    st.dataframe(df_fit, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:rgba(255,255,255,0.2); padding:20px;'>The Everything Calculator — Hurst Engine v2.1</div>", unsafe_allow_html=True)
