# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Professional Edition
# ------------------------------------------------------------
# - Design Institucional TEC (Dark Mode & No-Clipping)
# - Suporte Robusto a Excel/CSV (Auto-delimiter)
# - Análise de Rescaled Range (R/S)
# - Hurst Móvel (Rolling Memory Analysis)
# - Diagnóstico de Fractalidade e Normalidade

from __future__ import annotations
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA)
# ----------------------------
st.set_page_config(page_title="TEC • Hurst Pro", page_icon="📈", layout="wide")

# ----------------------------
# 1) ESTILO (CSS) - PADRÃO TEC SEM CORTES
# ----------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --panel: rgba(255,255,255,.045);
  --stroke: rgba(255,255,255,.08);
  --text: rgba(255,255,255,.92);
  --muted: rgba(229,231,235,.60);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}

/* Ajuste de margem superior para evitar que o Streamlit corte o título */
.block-container { 
    max-width: 1250px; 
    padding-top: 5rem !important; 
    padding-bottom: 2rem; 
}

.card {
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 1.5rem;
  margin-bottom: 1.2rem;
}

.title { 
    font-size: 2.4rem; 
    font-weight: 800; 
    color: var(--text); 
    margin-bottom: 0.5rem;
    line-height: 1.2; 
    display: flex;
    align-items: center;
    gap: 15px;
}

.sub { 
    color: var(--muted); 
    margin-bottom: 2.5rem; 
    font-size: 1.1rem; 
}

/* Customização dos Cards de Métricas */
div[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 15px;
}

hr { border:none; border-top:1px solid var(--stroke); margin: 2rem 0; }

.footer { text-align:center; color: var(--muted); margin-top: 40px; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 2) ENGINE MATEMÁTICO (CORE)
# ----------------------------
def get_hurst_rs(serie: np.ndarray, ns: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Calcula o Hurst usando Rescaled Range Analysis."""
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
    mask = np.isfinite(np.log(rs_values))
    # Regressão linear no espaço log-log
    h, intercept = np.polyfit(np.log(ns[mask]), np.log(rs_values[mask]), 1)
    return rs_values, h, intercept

def get_rolling_hurst(serie: np.ndarray, window: int) -> np.ndarray:
    """Calcula o Hurst de forma móvel ao longo da série."""
    rolling_h = []
    # Usamos uma grade fixa de 'n' para velocidade
    ns_fixo = np.unique(np.geomspace(8, window//2.5, 8).astype(int))
    
    for i in range(len(serie) - window):
        chunk = serie[i : i+window]
        _, h_val, _ = get_hurst_rs(chunk, ns_fixo)
        rolling_h.append(h_val)
    return np.array(rolling_h)

# ----------------------------
# 3) CABEÇALHO
# ----------------------------
st.markdown('<div class="title"><span>📈</span> Hurst Exponent Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Análise Profissional de Rescaled Range (R/S) para Séries Temporais.</div>', unsafe_allow_html=True)

# ----------------------------
# 4) SIDEBAR (CONTROLES)
# ----------------------------
st.sidebar.header("📂 Configurações de Dados")
uploaded = st.sidebar.file_uploader("Upload CSV ou Excel", type=["csv", "xlsx"])

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            # O separador None permite que o pandas detecte vírgula ou ponto-e-vírgula sozinho
            df = pd.read_csv(uploaded, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded)
        
        target_col = st.sidebar.selectbox("Selecione a Coluna", df.select_dtypes(include=[np.number]).columns)
        data = df[target_col].dropna().values
        st.sidebar.success(f"Carregado: {len(data)} registros")
    except Exception as e:
        st.sidebar.error(f"Erro ao processar arquivo: {e}")
        st.stop()
else:
    # Dados de Demonstração (Random Walk com Tendência)
    st.sidebar.info("Aguardando arquivo. Usando simulação padrão.")
    data = np.cumsum(np.random.randn(1000) + 0.01)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parâmetros da Análise")
min_n = st.sidebar.number_input("Janela Mínima (n)", 4, 100, 8)
max_n = st.sidebar.number_input("Janela Máxima (n)", 100, len(data)//2, 512)
points = st.sidebar.slider("Resolução (Pontos de n)", 10, 60, 25)
window_roll = st.sidebar.slider("Janela Móvel (Rolling)", 100, len(data)//2, 300)

# ----------------------------
# 5) CÁLCULOS PRINCIPAIS
# ----------------------------
t0 = time.time()
ns = np.unique(np.geomspace(min_n, max_n, points).astype(int))
rs_vals, H, intercept = get_hurst_rs(data, ns)

# Diagnóstico de Fractalidade
# Dimensão Fractal D = 2 - H
fractal_dim = 2 - H

dt = time.time() - t0

# ----------------------------
# 6) LAYOUT DE MÉTRICAS (TOP CARDS)
# ----------------------------
m1, m2, m3, m4 = st.columns(4)

# Lógica de Classificação
if H > 0.55:
    status, color = "PERSISTENTE", "normal"
elif H < 0.45:
    status, color = "ANTI-PERSISTENTE", "normal"
else:
    status, color = "RANDOM WALK", "off"

m1.metric("Hurst Exponent (H)", f"{H:.4f}")
m2.metric("Regime de Memória", status)
m3.metric("Dimensão Fractal (D)", f"{fractal_dim:.2f}")
m4.metric("Latência Engine", f"{dt*1000:.1f}ms")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ----------------------------
# 7) ABAS DE VISUALIZAÇÃO
# ----------------------------
tab_main, tab_rolling, tab_data = st.tabs(["📊 Análise R/S", "📈 Dinâmica Temporal", "📄 Diagnóstico Bruto"])

with tab_main:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Log-Log Plot
        fig_log = go.Figure()
        fig_log.add_trace(go.Scatter(
            x=np.log10(ns), y=np.log10(rs_vals),
            mode='markers', name='Observado',
            marker=dict(color='#FF4B4B', size=10, opacity=0.8)
        ))
        
        # Linha de Tendência
        fit_x = np.log10(ns)
        fit_y = H * np.log(ns) + intercept # Convertendo para base 10 visualmente
        # (Nota: polyfit foi em log natural, ajustamos para a visualização log10 se preferir, 
        # mas aqui manteremos a consistência visual da inclinação H)
        fit_y_vis = (H * np.log(ns) + intercept) / np.log(10) 
        
        fig_log.add_trace(go.Scatter(
            x=fit_x, y=fit_y_vis,
            mode='lines', name=f'Slope H = {H:.3f}',
            line=dict(color='white', width=2, dash='dot')
        ))
        
        fig_log.update_layout(
            title="Diagnóstico Log-Log (R/S Scaling)",
            xaxis_title="log10(Janela n)", yaxis_title="log10(R/S médio)",
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, margin=dict(t=60, b=20)
        )
        st.plotly_chart(fig_log, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        # Distribuição de Retornos
        returns = np.diff(data)
        fig_dist = px.histogram(
            returns, nbins=50, 
            title="Distribuição de Retornos (Normalidade vs Fractalidade)",
            color_discrete_sequence=['#1E90FF']
        )
        fig_dist.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, margin=dict(t=60, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab_rolling:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hurst Móvel (Rolling Analysis)")
    
    with st.spinner("Calculando evolução temporal..."):
        roll_h_vals = get_rolling_hurst(data, window_roll)
    
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        y=roll_h_vals, mode='lines', 
        line=dict(color='#00CC96', width=2),
        fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)',
        name="Hurst Móvel"
    ))
    
    # Linha guia de 0.5
    fig_roll.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.4)", annotation_text="Passeio Aleatório")
    
    fig_roll.update_layout(
        title=f"Evolução da Memória (Janela Móvel = {window_roll} pts)",
        xaxis_title="Tempo / Index", yaxis_title="Valor de H",
        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.info("💡 Interpretação: Quando o Hurst Móvel sobe consistentemente acima de 0.5, o ativo está ganhando força de tendência (persistência). Quedas abaixo de 0.5 sugerem exaustão e reversão.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dados Brutos do Escalonamento")
    df_diag = pd.DataFrame({
        "Janela (n)": ns,
        "R/S Médio": rs_vals,
        "Log(n)": np.log(ns),
        "Log(RS)": np.log(rs_vals)
    })
    st.dataframe(df_diag, use_container_width=True, hide_index=True)
    
    # Botão de download
    csv = df_diag.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar Tabela de Diagnóstico", data=csv, file_name="hurst_rs_scaling.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 8) RODAPÉ
# ----------------------------
st.markdown("<div class='footer'>The Everything Calculator — Hurst Professional Engine • Fellipe Almässy</div>", unsafe_allow_html=True)
