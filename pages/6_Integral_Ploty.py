import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.integrate import quad

# 1. ESTÉTICA DA PÁGINA
st.set_page_config(page_title="T.E.C. - Advanced Visuals", layout="wide")

# Custom CSS para esconder menus desnecessários e focar no conteúdo
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    stMetric { background-color: #161b22; border-radius: 10px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧮 Integração Numérica Reativa")
st.markdown("---")

# 2. SIDEBAR COM CONTROLES FLUIDOS
st.sidebar.header("Configurações do Modelo")
funcao_input = st.sidebar.text_input("Função f(x):", value="x**2 * sin(x)")
col1, col2 = st.sidebar.columns(2)
a = col1.number_input("Limite a", value=0.0)
b = col2.number_input("Limite b", value=6.28) # 2pi aproximado

# O slider de retângulos é o que dá o movimento ao interagir
num_ret = st.sidebar.select_slider("Refinamento da Malha (dx)", options=[10, 20, 30, 50, 100, 200])

# 3. MATEMÁTICA (U.A.G. SITREP)
try:
    x_sym = sp.Symbol('x')
    expressao = sp.sympify(funcao_input)
    f_num = sp.lambdify(x_sym, expressao, 'numpy')

    # Cálculos
    dx = (b - a) / num_ret
    x_ret = np.linspace(a, b - dx, num_ret)
    y_ret = f_num(x_ret)
    res_riemann = np.sum(y_ret * dx)
    res_real, _ = quad(f_num, a, b)
    erro = abs(res_real - res_riemann)

    # 4. MÉTRICAS COM INDICADOR DE DELTA
    m1, m2, m3 = st.columns(3)
    m1.metric("Aproximação de Riemann", f"{res_riemann:.4f}", f"dx: {dx:.4f}")
    m2.metric("Valor Exato (SciPy)", f"{res_real:.4f}")
    m3.metric("Erro Absoluto", f"{erro:.6e}", delta_color="inverse")

    # 5. O GRÁFICO PLOTLY (O IMPACTO VISUAL)
    x_curva = np.linspace(a - 1, b + 1, 1000)
    y_curva = f_num(x_curva)

    fig = go.Figure()

    # Área de preenchimento suave por trás
    fig.add_trace(go.Scatter(
        x=x_curva, y=y_curva, 
        fill='tozeroy', 
        name='Área Real',
        fillcolor='rgba(255, 75, 75, 0.1)',
        line=dict(color='rgba(255, 255, 255, 0)')
    ))

    # Curva Principal
    fig.add_trace(go.Scatter(
        x=x_curva, y=y_curva, 
        name='f(x)',
        line=dict(color='#FF4B4B', width=4)
    ))

    # Retângulos de Riemann (Onde o movimento acontece)
    fig.add_trace(go.Bar(
        x=x_ret, y=y_ret,
        width=dx, offset=0,
        name='Partição Riemann',
        marker=dict(color='#1E90FF', opacity=0.6, line=dict(color='white', width=0.5))
    ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro na expressão matemática: {e}")

st.markdown("<div style='text-align: center; color: #444;'>Unconventional Analysis Group - SITREP Method</div>", unsafe_allow_html=True)