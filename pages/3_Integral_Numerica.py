import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad

# IDENTIDADE VISUAL
st.set_page_config(page_title="T.E.C. - Cálculo Numérico", layout="wide")

# CABECALHO 
st.caption("Unconventional Analysis Group U.A.G")
st.title("Integrais Definidas")

# --- Fundamentação Teórica (Padrão U.A.G.) ---
st.markdown("### 🏛️ Fundamentação Teórica (U.A.G.)")

st.write("""
A **Integral Definida** representa o acúmulo de quantidades infinitesimais em um 
determinado intervalo. Geometricamente, ela é a **área líquida** entre a função 
$f(x)$ e o eixo horizontal ($x$).
""")

with st.popover("Visualizar Processo de Soma de Riemann"):
    st.write("1. **Partição:** O intervalo $[a, b]$ é dividido em pedaços de largura **dx**.")
    st.write("2. **Amostragem:** Medimos a altura da curva no início de cada intervalo.")
    st.write("3. **Somatória:** O resultado final aproxima o valor real da integral.")
    st.latex(r"\int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1} f(x_i) \cdot \Delta x")

st.markdown("---")

# SIDEBAR
st.sidebar.header("Parâmetros do Cálculo")
funcao_input = st.sidebar.text_input("Digite a função f(x):", value="x**3")

col_side_a, col_side_b = st.sidebar.columns(2)
a = col_side_a.number_input("Limite inferior (a)", value=-2.0)
b = col_side_b.number_input("Limite superior (b)", value=2.0)

num_ret = st.sidebar.number_input("Quantidade de Retângulos", value=30, step=10)

# NOVO: Botão de Arredondamento na Sidebar
arredondar = st.sidebar.checkbox("Arredondar resultados", value=False)

# MATEMATICA
try:
    x_sym = sp.Symbol('x')
    expressao = sp.sympify(funcao_input)
    f_num = sp.lambdify(x_sym, expressao, 'numpy')

    # 1. Cálculo por Aproximação (Soma de Riemann)
    dx = (b - a) / num_ret
    x_ret = np.linspace(a, b - dx, int(num_ret))
    y_ret = f_num(x_ret)
    resultado_riemann = np.sum(y_ret * dx)

    # 2. Cálculo Real (SciPy)
    resultado_real, _ = quad(f_num, a, b)

    # --- Visualização de Resultados ---
    m1, m2, m3 = st.columns(3)
    
    # Lógica de arredondamento aplicada às métricas
    def formatar(valor):
        return f"{round(valor)}" if arredondar else f"{valor:.6f}"

    with m1:
        st.write("Aproximação (Riemann)")
        st.subheader(formatar(resultado_riemann))
    
    with m2:
        st.write("Resultado Real (SciPy)")
        st.subheader(formatar(resultado_real))
        
    with m3:
        st.write("Tamanho dx")
        st.subheader(f"{dx:.4f}")

    # --- Visualização Gráfica ---
    st.markdown("### Visualização Gráfica")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    x_curva = np.linspace(a - 0.5, b + 0.5, 500)
    y_curva = f_num(x_curva)
    ax.plot(x_curva, y_curva, color='#FF4B4B', linewidth=2.5)

    ax.bar(x_ret, y_ret, width=dx, align='edge', color='#1E90FF', alpha=0.4, edgecolor='#1E90FF')

    ax.axhline(0, color='white', linewidth=1)
    ax.axvline(0, color='white', linewidth=1)
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"Erro: {e}")

st.divider()
st.markdown("<div style='text-align: center; color: gray;'>T.E.C. - Unconventional Analysis Group</div>", unsafe_allow_html=True)