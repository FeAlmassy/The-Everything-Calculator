import streamlit as st
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# IDENTIDADE VISUAL
st.set_page_config(page_title="T.E.C. - Cálculo Numérico", layout="wide")

# CABECALHO 
st.caption("Unconventional Analysis Group U.A.G")
st.title("Integrais Definidas")

st.write("""
### 🏛️ Fundamentação Teórica (U.A.G.)

A **Integral Definida** representa o acúmulo de quantidades infinitesimais em um 
determinado intervalo. Geometricamente, ela é a **área líquida** entre a função 
$f(x)$ e o eixo horizontal ($x$).

Neste módulo, utilizamos as **Somas de Riemann**, que seguem este processo:

1. **Partição:** O intervalo total de $[a, b]$ é dividido em pedaços de largura 
igual, chamados de **dx**.

2. **Amostragem:** Para cada pedaço, medimos a altura da curva no ponto inicial 
do intervalo para formar um retângulo.

3. **Acúmulo:** Calculamos a área de cada retângulo individual e as guardamos 
na nossa **lista_areas**.

4. **Somatória:** O resultado final é a soma de todos esses valores, aproximando 
o valor real da integral.
""")

st.latex(r"\int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1} f(x_i) \cdot \Delta x")
st.divider()

# SIDEBAR
st.sidebar.header("Parâmetros do Cálculo")
funcao_input = st.sidebar.text_input("Digite a função f(x):", value="x**3")

col_side_a, col_side_b = st.sidebar.columns(2)
a = col_side_a.number_input("Limite inferior", value=-2.0)
b = col_side_b.number_input("Limite superior", value=2.0)

# SIDEBAR NUM RET
num_ret = st.sidebar.select_slider(
    "Quantidade de Retângulos",
    options=[10, 20, 30, 50, 100, 1000, 10000, 50000, 100000],
    value=30
)

# MATEMATICA
try:
    x_sym = sp.Symbol('x')
    expressao = sp.sympify(funcao_input)
    f_num = sp.lambdify(x_sym, expressao, 'numpy')

    tamanho_eixo_x = b - a
    dx = tamanho_eixo_x / num_ret
    lista_areas = []


    for passo in range(num_ret):
        x_atual = a + dx * passo
        area = dx * f_num(x_atual)
        lista_areas.append(area)

    resultado_final = sum(lista_areas)

    # --- Visualização de Resultados ---
    m1, m2, m3 = st.columns(3)
    m1.metric("Area total", f"{resultado_final:.6f}")
    m2.metric("Numero de retangulos", f"{num_ret}")
    m3.metric("tamanho dx", f"{dx:.4}")

    st.subheader("Visualizacao Grafica")
    c1, c2, c3 = st.columns([1, 3, 1])
    
    with c2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        cor_grade = '#808080'
        
        x_curva = np.linspace(a - 1, b + 1, 400)
        y_curva = f_num(x_curva)
        
        # Desenhar retângulos (apenas se num_ret for visualmente viável)
        if num_ret <= 1000:
            for i in range(num_ret):
                x_ret = a + i * dx
                # Usando f_num diretamente para a altura
                ax.add_patch(plt.Rectangle((x_ret, 0), dx, f_num(x_ret), 
                                           edgecolor='#1E90FF', facecolor='#1E90FF', alpha=0.3))
        else:
            # Preenchimento para muitos retângulos
            ax.fill_between(x_curva, y_curva, where=(x_curva>=a) & (x_curva<=b), color='#1E90FF', alpha=0.2)

        ax.plot(x_curva, y_curva, color='#FF4B4B', label=f'f(x) = {funcao_input}', linewidth=2)
        ax.axhline(0, color=cor_grade, linewidth=1.2)
        ax.axvline(0, color=cor_grade, linewidth=1.2)
        ax.grid(True, linestyle=':', alpha=0.3, color=cor_grade)
        ax.tick_params(colors=cor_grade)
        
        st.pyplot(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erro: {e}")

st.divider()
st.markdown("<div style='text-align: center; color: gray;'>T.E.C. - The Everything </div>", unsafe_allow_html=True)