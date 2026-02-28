# streamlit_app.py
# Engine de Análise de Hurst — Versão Profissional (TEC)
# ------------------------------------------------------------
# - Algoritmo Rescaled Range (R/S) de Hurst
# - Suporte a upload de Excel/CSV
# - Visualização Log-Log com Regressão Linear
# - Diagnósticos de Persistência e Memória
# - Estilo Institucional TEC

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from typing import Tuple, List

# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA
# ----------------------------
st.set_page_config(page_title="TEC | Hurst Exponent", layout="wide")

# ----------------------------
# 1) ESTILO (CSS) - PADRÃO TEC
# ----------------------------
st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.08);
  --muted: rgba(229,231,235,0.60);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}
.main { background-color: var(--bg); }
section[data-testid="stSidebar"] { background-color: #0b1020; border-right: 1px solid var(--border); }
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px;
}
.hr { border-top: 1px solid var(--border); margin: 1rem 0; }
.badge {
  display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
  color: rgba(229,231,235,0.80); font-size: 0.82rem;
}
.footer { text-align:center; color: var(--muted); margin-top: 20px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2) CORE ENGINE: HURST R/S
# ----------------------------
def get_hurst_exponent(series: np.ndarray, max_window: int = None) -> Tuple[float, List, List]:
    """Calcula o expoente de Hurst usando Rescaled Range Analysis."""
    series = np.array(series)
    N = len(series)
    
    # Gera janelas (powers of 2)
    min_window = 8
    if max_window is None:
        max_window = N // 2
        
    # Criar escalas logarítmicas
    lags = np.unique(np.floor(np.geomspace(min_window, max_window, num=20)).astype(int))
    
    RS_values = []
    
    for lag in lags:
        # Divide a série em blocos de tamanho 'lag'
        n_chunks = N // lag
        rs_list = []
        
        for i in range(n_chunks):
            chunk = series[i*lag : (i+1)*lag]
            # 1. Calcular média e desvio
            mean = np.mean(chunk)
            std = np.std(chunk)
            if std == 0: continue
            
            # 2. Séries acumulada de desvios
            z = np.cumsum(chunk - mean)
            
            # 3. Range (Amplitude)
            r = np.max(z) - np.min(z)
            
            # 4. Rescaled Range
            rs_list.append(r / std)
            
        RS_values.append(np.mean(rs_list))
    
    # Regressão Linear no espaço log-log
    log_lags = np.log(lags)
    log_rs = np.log(RS_values)
    
    p = np.polyfit(log_lags, log_rs, 1)
    return float(p[0]), lags, RS_values

# ----------------------------
# 3) UI - HEADER E TEORIA
# ----------------------------
st.title("Hurst Exponent Engine")
st.caption("Análise de Memória de Longo Prazo e Fractalidade")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

with st.expander("📚 Teoria: O que o Expoente de Hurst nos diz?"):
    st.markdown("""
    O expoente de Hurst ($H$) é uma medida de **autocorrelação de longo prazo**.
    * **$H < 0.5$ (Anti-persistente):** A série tende a reverter à média. Se subiu, a probabilidade é que desça.
    * **$H = 0.5$ (Random Walk):** Passeio aleatório (Movimento Browniano). Sem memória.
    * **$H > 0.5$ (Persistente):** Presença de tendência. Se subiu, tende a continuar subindo.
    """)
    st.latex(r"E[R(n)/S(n)] = C \cdot n^H")

# ----------------------------
# 4) SIDEBAR - CONTROLES
# ----------------------------
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Excel ou CSV", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)
        
        col_target = st.sidebar.selectbox("Selecione a Coluna", df_input.columns)
        data_series = df_input[col_target].dropna().values
        
        st.sidebar.success(f"Loaded: {len(data_series)} pontos")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar: {e}")
        st.stop()
else:
    # Gerador de dados sintéticos para demo
    st.sidebar.info("Aguardando arquivo. Usando Random Walk para demonstração.")
    data_series = np.cumsum(np.random.randn(2000))

# ----------------------------
# 5) PROCESSAMENTO
# ----------------------------
t0 = time.time()
H, lags, rs_vals = get_hurst_exponent(data_series)
dt = time.time() - t0

# ----------------------------
# 6) DASHBOARD - MÉTRICAS
# ----------------------------
m1, m2, m3, m4 = st.columns(4)

# Lógica de interpretação
if H < 0.45: status, color = "REVERSÃO", "normal"
elif H > 0.55: status, color = "TENDÊNCIA", "normal"
else: status, color = "ALEATÓRIO", "off"

m1.metric("Expoente de Hurst (H)", f"{H:.4f}")
m2.metric("Comportamento", status)
m3.metric("Pontos Analisados", f"{len(data_series)}")
m4.metric("Tempo de Proc.", f"{dt:.3f}s")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ----------------------------
# 7) VISUALIZAÇÃO (ABAS)
# ----------------------------
tab_plot, tab_data = st.tabs(["Análise Gráfica", "Dados de Regressão"])

with tab_plot:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Gráfico Log-Log
        fig = go.Figure()
        
        # Pontos R/S
        fig.add_trace(go.Scatter(
            x=np.log10(lags), y=np.log10(rs_vals),
            mode='markers', name='R/S Observado',
            marker=dict(color='#1E90FF', size=10, opacity=0.7)
        ))
        
        # Linha de Regressão
        m, b = np.polyfit(np.log10(lags), np.log10(rs_vals), 1)
        fit_y = m * np.log10(lags) + b
        
        fig.add_trace(go.Scatter(
            x=np.log10(lags), y=fit_y,
            mode='lines', name=f'Fit (H={H:.3f})',
            line=dict(color='#FF4B4B', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title="Diagnóstico Log-Log (R/S Analysis)",
            xaxis_title="log(Janela)", yaxis_title="log(R/S)",
            template="plotly_dark", height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Série Temporal")
        fig_series = go.Figure()
        fig_series.add_trace(go.Scatter(y=data_series, line=dict(color='#FF4B4B', width=1.5)))
        fig_series.update_layout(
            template="plotly_dark", height=300, 
            showlegend=False, margin=dict(l=0,r=0,t=0,b=0)
        )
        st.plotly_chart(fig_series, use_container_width=True)
        
        st.info(f"O valor de H={H:.2f} indica que a série possui {'memória positiva' if H > 0.5 else 'memória negativa (reversão)' if H < 0.5 else 'ausência de memória'}.")

with tab_data:
    df_diag = pd.DataFrame({
        "Janela (n)": lags,
        "R/S Average": rs_vals,
        "log(n)": np.log10(lags),
        "log(R/S)": np.log10(rs_vals)
    })
    st.dataframe(df_diag, use_container_width=True, hide_index=True)

# ----------------------------
# 8) RODAPÉ
# ----------------------------
st.markdown(f"<div class='footer'>The Everything Calculator - Fellipe Almässy • {time.strftime('%Y')}</div>", unsafe_allow_html=True)
