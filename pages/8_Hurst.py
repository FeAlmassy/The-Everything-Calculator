# streamlit_app.py
# TEC — Hurst Exponent (R/S) — Versão Restaurada e Blindada
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time

# ----------------------------
# 0) CONFIGURAÇÃO (Obrigatório ser o 1º comando)
# ----------------------------
st.set_page_config(page_title="TEC • Hurst (R/S)", page_icon="📈", layout="wide")

# ----------------------------
# 1) CSS DE BLINDAGEM (Evita o erro de texto cortado)
# ----------------------------
st.markdown(
    """
<style>
/* Esconde o cabeçalho nativo que sobrepõe o título */
header[data-testid="stHeader"] { visibility: hidden; height: 0px; }

/* Garante fundo escuro e empurra o conteúdo para baixo */
.main { background-color: #0b1220; }
.block-container { 
    max-width: 1250px; 
    padding-top: 5rem !important; 
}

/* Estilo dos Cards e Textos */
.card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.title { 
    font-size: 2.2rem; 
    font-weight: 800; 
    color: rgba(255,255,255,.92); 
    line-height: 1.2;
    margin-bottom: 0.5rem;
}

.sub { color: rgba(229,231,235,.66); margin-bottom: 2rem; }

div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Título Principal
st.markdown('<div class="title">📈 Hurst Exponent — R/S Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Análise de Rescaled Range (R/S) com Tabela de Métricas por Bloco.</div>', unsafe_allow_html=True)

# ----------------------------
# 2) FUNÇÕES DO MOTOR
# ----------------------------
def metricas_do_bloco(bloco, ddof=1):
    x = np.array(bloco, dtype=float)
    mu, std = np.mean(x), np.std(x, ddof=ddof)
    Y = np.cumsum(x - mu)
    R = np.max(Y) - np.min(Y)
    return {"mean": mu, "std": std, "R": R, "RS": R/std if std > 0 else np.nan, "min_Y": np.min(Y), "max_Y": np.max(Y)}

def rs_medio_por_n(serie, n):
    blocos = [serie[i:i+n] for i in range(0, len(serie), n) if len(serie[i:i+n]) == n]
    rs_vals = [metricas_do_bloco(b)["RS"] for b in blocos]
    return np.mean([r for r in rs_vals if np.isfinite(r)]) if rs_vals else np.nan

# ----------------------------
# 3) INTERFACE LATERAL E UPLOAD
# ----------------------------
st.sidebar.header("📁 Dados")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded, sep=None, engine='python')
    else:
        df = pd.read_excel(uploaded)
    col = st.sidebar.selectbox("Coluna Numérica", df.select_dtypes(include=[np.number]).columns)
    serie = df[col].dropna().values
else:
    st.sidebar.info("Aguardando arquivo...")
    st.stop()

# Configurações
n_block = st.sidebar.number_input("n para Tabela", 2, 5000, 100)
min_n, max_n = st.sidebar.slider("Faixa de n (Scaling)", 8, 100, 8), st.sidebar.slider("Máx n", 100, len(serie)//2, 512)

# ----------------------------
# 4) TABELA DE BLOCOS (REQUISITO DA VERSÃO 2)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(f"📋 Métricas por Bloco (n = {n_block})")
blocos_lista = [serie[i:i+n_block] for i in range(0, len(serie), n_block) if len(serie[i:i+n_block]) == n_block]
df_blocks = pd.DataFrame([metricas_do_bloco(b) for b in blocos_lista])
st.dataframe(df_blocks, use_container_width=True, height=250)
st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# 5) CÁLCULO HURST E GRÁFICOS
# ----------------------------
ns = np.unique(np.geomspace(min_n, max_n, 20).astype(int))
rs_vals = np.array([rs_medio_por_n(serie, n) for n in ns])
valid = np.isfinite(np.log10(rs_vals))
H, inter = np.polyfit(np.log10(ns[valid]), np.log10(rs_vals[valid]), 1)

m1, m2 = st.columns(2)
m1.metric("Hurst Exponent (H)", f"{H:.4f}")
m2.metric("Regime", "Persistente" if H > 0.55 else "Reversão" if H < 0.45 else "Aleatório")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig1 = go.Figure(go.Scatter(x=ns, y=rs_vals, mode='lines+markers', line=dict(color='#1E90FF')))
    fig1.update_layout(title="R/S vs n", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    lx, ly = np.log10(ns[valid]), np.log10(rs_vals[valid])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=lx, y=ly, mode='markers', name='Dados', marker=dict(color='#FF4B4B')))
    fig2.add_trace(go.Scatter(x=lx, y=H*lx + inter, mode='lines', name=f'H={H:.3f}', line=dict(color='white', dash='dot')))
    fig2.update_layout(title="Log-Log Plot", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
