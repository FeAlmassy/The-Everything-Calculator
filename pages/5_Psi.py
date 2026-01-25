import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import polygamma # Biblioteca específica para a função Psi

# Configuração de Layout padrão TEC
st.set_page_config(layout="wide")

st.title("Função Psi")

# --- Fundamentação Teórica (Padrão U.A.G.) ---
st.markdown("### 🏛️ Fundamentação Teórica (U.A.G.)")

st.markdown("""
A **Função Psi**, denotada por $\psi(n)$, é definida como a derivada logarítmica da Função Gamma. Enquanto a Gamma cresce de forma explosiva (fatorial), a Função Psi descreve a "velocidade" desse crescimento em uma escala logarítmica.

**O que é e quem a criou?** Ela foi estudada profundamente por **James Stirling** e **Bernhard Riemann**. Ela resolve o problema de entender como o "fatorial" se comporta quando analisamos sua inclinação. Matematicamente, ela é o passo seguinte após a Gamma no estudo de funções especiais.

**Para que serve?** 1. **Somas de Séries:** É usada para converter somas infinitas complexas em valores exatos.
2. **Estatística:** Fundamental para estimar parâmetros de distribuições de probabilidade.
3. **Física de Partículas:** Aparece em cálculos de renormalização.
""")

# --- Botão Pop-up (Demonstrações e Valores) ---
with st.popover("Visualizar Relação com Gamma e Valores"):
    st.markdown("### Demonstração Matemática")
    st.write("A definição formal da Função Psi é:")
    st.latex(r"\psi(n) = \frac{d}{dn} \ln(\Gamma(n)) = \frac{\Gamma'(n)}{\Gamma(n)}")
    st.divider()
    st.markdown("**Valores Notáveis:**")
    st.latex(r"\psi(1) = -\gamma \approx -0.57721 \text{ (Constante de Euler-Mascheroni)}")
    st.latex(r"\psi(2) = 1 - \gamma")
    st.latex(r"\psi(n+1) = \psi(n) + \frac{1}{n}")
    st.info("A última fórmula mostra que a Psi é a versão contínua das somas harmônicas!")

st.markdown("---")

# --- Sidebar para Parâmetros ---
st.sidebar.header("Parâmetros da Psi")
n_val = st.sidebar.number_input("Valor de n:", value=1.0, step=0.1)
x_max = st.sidebar.number_input("Limite de visualização (eixo x):", value=5.0, step=1.0)

# --- Cálculo ---
# A função polygamma(0, x) é a função Digamma (Psi)
resultado_psi = polygamma(0, n_val)

# --- Métricas ---
col1, col2 = st.columns(2)
with col1:
    st.write("Valor de $\psi(n)$")
    st.subheader(f"{resultado_psi:.10f}")
with col2:
    st.write("Comportamento")
    if n_val < 0.5:
        st.warning("Próximo a uma Assíntota Vertical")
    else:
        st.success("Zona de Crescimento Logarítmico")

# --- Visualização Gráfica ---
st.markdown("### Visualizacao Grafica")

fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.tick_params(colors='white')
ax.grid(True, linestyle='--', alpha=0.2)

# Geramos os dados para o gráfico (evitando x <= 0 onde a Psi explode)
x_plot = np.linspace(0.1, x_max, 500)
y_plot = polygamma(0, x_plot)

# Desenha a curva (Verde/Ciano para diferenciar da Gamma) e linha de zero
ax.plot(x_plot, y_plot, color='#00ffcc', linewidth=2.5, label=r'$\psi(x)$')
ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
ax.axvline(n_val, color='#ff4b4b', linestyle=':', label=f'n = {n_val}')

ax.set_xlim(0, x_max)
ax.legend()

st.pyplot(fig)