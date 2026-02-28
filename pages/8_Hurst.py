import streamlit as st
import numpy as np
import sympy as sp
import plotly.graph_objects as go
from scipy.integrate import quad

# --- CONFIGURAÇÃO T.E.C. STYLE ---
st.set_page_config(page_title="TEC • Parametric Integrator", layout="wide")

st.markdown("""
<style>
    :root { --accent: #FF4B4B; --bg: #0e1117; }
    .stApp { background-color: var(--bg); }
    .main-title { font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧩 Construtor de Modelos Paramétricos</p>', unsafe_allow_html=True)
st.caption("Ajuste coeficientes reais para modelagem de fenômenos sem erros de sintaxe.")

with st.sidebar:
    st.header("⚙️ Configuração do Modelo")
    modelo = st.selectbox("Escolha o Modelo Matemático", 
                        ["Polinomial (Quadrático)", "Oscilatório (Senoide)", "Crescimento Exponencial"])
    
    st.divider()
    
    if modelo == "Polinomial (Quadrático)":
        st.write("f(x) = ax² + bx + c")
        a_val = st.slider("Coeficiente a", -5.0, 5.0, 1.0)
        b_val = st.slider("Coeficiente b", -5.0, 5.0, 0.0)
        c_val = st.slider("Constante c", -10.0, 10.0, 0.0)
        expr_str = f"{a_val}*x**2 + {b_val}*x + {c_val}"
        
    elif modelo == "Oscilatório (Senoide)":
        st.write("f(x) = A * sin(w * x + phi)")
        amp = st.slider("Amplitude (A)", 0.1, 10.0, 1.0)
        freq = st.slider("Frequência (w)", 0.1, 5.0, 1.0)
        fase = st.slider("Fase (phi)", 0.0, 6.28, 0.0)
        expr_str = f"{amp}*sin({freq}*x + {fase})"
        
    elif modelo == "Crescimento Exponencial":
        st.write("f(x) = A * e^(k * x)")
        base = st.slider("Escala (A)", 0.1, 5.0, 1.0)
        taxa = st.slider("Taxa (k)", -1.0, 1.0, 0.2)
        expr_str = f"{base}*exp({taxa}*x)"

    st.divider()
    limites = st.slider("Intervalo de Integração [a, b]", -20.0, 20.0, (0.0, 5.0))
    n_part = st.number_input("Partições (n)", value=100)

# --- ENGINE MATEMÁTICA ---
x_sym = sp.Symbol('x')
expr = sp.sympify(expr_str)
f_num = sp.lambdify(x_sym, expr, modules=['numpy'])

# Cálculo da Integral (SciPy para referência rápida)
area, _ = quad(f_num, limites[0], limites[1])

# --- VISUALIZAÇÃO ---
st.latex(rf"f(x) = {sp.latex(expr)}")

col1, col2 = st.columns([3, 1])

with col1:
    x_plot = np.linspace(limites[0]-2, limites[1]+2, 500)
    y_plot = f_num(x_plot)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_plot, y=y_plot, name="f(x)", line=dict(color="#FF4B4B", width=3)))
    
    # Preenchimento da Integral
    x_fill = np.linspace(limites[0], limites[1], 100)
    fig.add_trace(go.Scatter(x=x_fill, y=f_num(x_fill), fill='tozeroy', name="Área Integrada", fillcolor='rgba(255, 75, 75, 0.2)', line=dict(width=0)))
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric("Integral Calculada", f"{area:.4f}")
    st.write(f"Intervalo: [{limites[0]}, {limites[1]}]")
    st.info("Esta abordagem foca em parâmetros de negócio/engenharia.")
