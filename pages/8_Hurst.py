# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Professional Edition (FIXED TOP)
# ------------------------------------------------------------

from __future__ import annotations
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA
# ----------------------------
st.set_page_config(page_title="TEC • Hurst Pro", page_icon="📈", layout="wide")

# ----------------------------
# 1) ESTILO (CSS) - BLINDAGEM CONTRA CORTES
# ----------------------------
st.markdown(
    """
<style>
/* 1. Esconde a barra nativa do Streamlit que causa o corte */
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0px;
}

/* 2. Define o fundo e o espaçamento real do container */
.main { background-color: #0e1117; }
.block-container { 
    max-width: 1250px; 
    padding-top: 2rem !important; /* Reduzido pois agora o spacer faz o trabalho */
    padding-bottom: 2rem; 
}

/* 3. Spacer de segurança no topo */
.top-spacer { height: 40px; }

/* 4. Título com line-height e overflow corrigido */
.title-container {
    margin-bottom: 2rem;
    padding-top: 10px;
}

.title { 
    font-size: 2.5rem; 
    font-weight: 800; 
    color: rgba(255,255,255,.95); 
    line-height: 1.1; 
    display: flex;
    align-items: center;
    gap: 15px;
    overflow: visible !important;
}

.sub { 
    color: rgba(229,231,235,.60); 
    margin-top: 8px;
    font-size: 1.1rem; 
}

/* 5. Design dos Cards */
.card {
  background: rgba(255,255,255,.045);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 1.2rem;
}

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 14px;
}

hr { border:none; border-top:1px solid rgba(255,255,255,.08); margin: 2rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 2) ELEMENTO ESPAÇADOR + HEADER
# ----------------------------
# O spacer "empurra" o conteúdo para baixo fisicamente, evitando o erro de corte
st.markdown('<div class="top-spacer"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="title-container">
    <div class="title">📈 Hurst Exponent Engine</div>
    <div class="sub">Análise Profissional de Rescaled Range (R/S) para detecção de persistência.</div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 3) ENGINE (CÁLCULO R/S)
# ----------------------------
def get_hurst_rs(serie: np.ndarray, ns: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Cálculo do R/S com base em Mandelbrot/Hurst."""
    rs_values = []
    for n in ns:
        n_chunks = len(serie) // n
        if n_chunks == 0: continue
        
        chunk_rs = []
        for i in range(n_chunks):
            chunk = serie[i*n : (i+1)*n]
            mu = np.mean(chunk)
            std = np.std(chunk, ddof=1)
            if std > 0:
                y = np.cumsum(chunk - mu)
                r = np.max(y) - np.min(y)
                chunk_rs.append(r / std)
        
        if chunk_rs:
            rs_values.append(np.mean(chunk_rs))
        else:
            rs_values.append(np.nan)
            
    rs_values = np.array(rs_values)
    mask = np.isfinite(np.log10(rs_values))
    # Usando log10 para consistência visual no gráfico
    h, intercept = np.polyfit(np.log10(ns[mask]), np.log10(rs_values[mask]), 1)
    return rs_values, h, intercept

# ----------------------------
# 4) INPUT DE DADOS (CSV/EXCEL)
# ----------------------------
st.sidebar.header("📁 Importação")
uploaded = st.sidebar.file_uploader("Upload arquivo", type=["csv", "xlsx"])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            # Engine python + sep=None detecta vírgula ou ponto-e-vírgula do arquivo carregado
            df = pd.read_csv(uploaded, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded)
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_col = st.sidebar.selectbox("Coluna", num_cols)
        data = df[target_col].dropna().values
    except Exception as e:
        st.sidebar.error(f"Erro no arquivo: {e}")
        st.stop()
else:
    # Fallback para simulação
    st.sidebar.info("Aguardando arquivo. Simulando dados.")
    data = np.cumsum(np.random.randn(1000) + 0.01)

# ----------------------------
# 5) PARÂMETROS E PROCESSAMENTO
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações")
min_n = st.sidebar.number_input("Janela Mín (n)", 8, 100, 8)
max_n = st.sidebar.number_input("Janela Máx (n)", 100, len(data)//2, 512)
points = st.sidebar.slider("Resolução", 10, 50, 25)
window_roll = st.sidebar.slider("Janela Móvel (Hurst)", 100, len(data)//2, 300)

t0 = time.time()
ns = np.unique(np.geomspace(min_n, max_n, points).astype(int))
rs_vals, H, intercept = get_hurst_rs(data, ns)
dt = time.time() - t0

# ----------------------------
# 6) MÉTRICAS PRINCIPAIS
# ----------------------------
c1, c2, c3, c4 = st.columns(4)

if H > 0.55: msg = "PERSISTENTE"
elif H < 0.45: msg = "ANTI-PERSISTENTE"
else: msg = "ALEATÓRIO"

c1.metric("Expoente H", f"{H:.4f}")
c2.metric("Regime", msg)
c3.metric("Dim. Fractal (D)", f"{2-H:.2f}")
c4.metric("Processamento", f"{dt*1000:.0f}ms")

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# 7) VISUALIZAÇÃO (ABAS)
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Análise R/S", "📈 Hurst Dinâmico", "📄 Dados"])

with tab1:
    g1, g2 = st.columns(2)
    
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Plot Log-Log (Base 10 para melhor leitura)
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(
            x=np.log10(ns), y=np.log10(rs_vals),
            mode='markers', name='Obs', marker=dict(color='#FF4B4B', size=10)
        ))
        lx = np.log10(ns)
        ly = H * lx + intercept
        fig_log.add_trace(go.Scatter(x=lx, y=ly, mode='lines', name=f'Fit (H={H:.3f})', line=dict(color='white', dash='dot')))
        
        fig_log.update_layout(
            title="Escalonamento Log10 (R/S Analysis)",
            xaxis_title="log10(Janela n)", yaxis_title="log10(R/S)",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450
        )
        st.plotly_chart(fig_log, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Série Temporal Original
        fig_series = go.Figure()
        fig_series.add_trace(go.Scatter(y=data, mode='lines', line=dict(color='#1E90FF', width=1.5)))
        fig_series.update_layout(
            title="Série de Dados Analisada",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=450
        )
        st.plotly_chart(fig_series, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Cálculo Rolling Hurst
    rolling_h = []
    ns_fixo = np.unique(np.geomspace(8, window_roll//2.5, 8).astype(int))
    for i in range(len(data) - window_roll):
        _, h_val, _ = get_hurst_rs(data[i:i+window_roll], ns_fixo)
        rolling_h.append(h_val)
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(y=rolling_h, mode='lines', line=dict(color='#00CC96'), fill='tozeroy'))
    fig_roll.add_hline(y=0.5, line_dash="dash", line_color="white", annotation_text="Passeio Aleatório")
    fig_roll.update_layout(
        title=f"Variação do Hurst no Tempo (Janela = {window_roll})",
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    df_raw = pd.DataFrame({"Janela (n)": ns, "R/S Médio": rs_vals})
    st.dataframe(df_raw, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='text-align:center; color:rgba(255,255,255,0.2); font-size:12px; padding:20px;'>TEC Professional Edition — Hurst Analytics</div>", unsafe_allow_html=True)
