import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Configuração de Layout padrão TEC
st.set_page_config(layout="wide")

st.title("Função Gamma")

# --- Fundamentação Teórica (Padrão U.A.G.) ---
st.markdown("### Fundamentação Teórica")

st.markdown("""
A **Função Gamma**, representada pela letra grega $\Gamma(n)$, é uma das ferramentas mais poderosas da análise matemática clássica, servindo como a **extensão natural da função fatorial** para o campo dos números reais e complexos. 

**O que é e quem a criou?** Diferente do fatorial comum ($n!$), que funciona apenas com números inteiros, a Função Gamma permite calcular o "fatorial" de números fracionários. Ela foi introduzida por **Leonhard Euler** em 1729.

**Para que serve?** Utilizada em Estatística (Distribuição Gamma), Física Quântica e Engenharia para modelar comportamentos assintóticos e tempos de espera.
""")

# --- Botão Pop-up (Tabela de Valores) ---
with st.popover("Demonstracao Simbolica"):
    st.markdown("### Relação com o Fatorial")
    st.write("Abaixo, a progressão da Função Gamma para inteiros:")
    st.latex(r"\Gamma(1) = 0! = 1")
    st.latex(r"\Gamma(2) = 1! = 1")
    st.latex(r"\Gamma(3) = 2! = 2 \cdot 1 = 2")
    st.latex(r"\Gamma(4) = 3! = 3 \cdot 2 \cdot 1 = 6")
    st.latex(r"\Gamma(5) = 4! = 4 \cdot 3 \cdot 2 \cdot 1 = 24")
    st.divider()
    st.markdown("**Generalização (Propriedade Recursiva):**")
    st.latex(r"\Gamma(n) = (n-1) \cdot \Gamma(n-1)")
    st.info("Esta relação permite que a função 'salte' de um inteiro para o próximo, definindo a estrutura do fatorial.")

st.latex(r"\Gamma(n) = \int_{0}^{\infty} x^{n-1} e^{-x} \, dx")

st.markdown("---")

# --- Sidebar para Parâmetros ---
st.sidebar.header("Parâmetros do Cálculo")
n = st.sidebar.number_input("Digite o valor de n :", value=4.0, step=0.1)
b_grafico = st.sidebar.number_input("Limite de visualização no gráfico:", value=20.0, step=1.0)

# --- Lógica de Cálculo (SciPy) ---
def gamma_func(x, n):
    return (x**(n-1)) * np.exp(-x)

resultado_real, erro = quad(gamma_func, 0, np.inf, args=(n))

# --- Exibição de Métricas ---
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Resultado Real $\Gamma(n)$")
    st.subheader(f"{resultado_real:.10f}")
with col2:
    st.write("Erro de Cálculo (Biblioteca)")
    st.subheader(f"{erro:.2e}")
with col3:
    st.write("Fatorial Equivalente")
    if n.is_integer() and n > 0:
        st.subheader(f"{int(n-1)}!")
    else:
        st.subheader("n/a")

# --- Visualização Gráfica ---
st.markdown("### Visualizacao Grafica")

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax.grid(True, linestyle='--', alpha=0.2)

x_plot = np.linspace(0.0001, b_grafico, 500)
y_plot = gamma_func(x_plot, n)

ax.plot(x_plot, y_plot, color='#ff4b4b', linewidth=2.5, label=f'f(x)')
ax.fill_between(x_plot, y_plot, color='#1f77b4', alpha=0.3)
ax.set_xlim(0, b_grafico)

st.pyplot(fig)
