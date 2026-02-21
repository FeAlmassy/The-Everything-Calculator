import streamlit as st

# 1) CONFIGURAÇÃO (Deve ser a primeira linha)
st.set_page_config(page_title="TEC - Home", layout="wide", page_icon="🧮")

# 2) ESTILO (Mantendo sua identidade visual)
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.08);
  --muted: rgba(229,231,235,0.60);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}

.main { background-color: var(--bg); }

.hero-section {
    padding: 3rem 1rem;
    background: linear-gradient(135deg, rgba(255,75,75,0.05) 0%, rgba(30,144,255,0.05) 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    text-align: center;
}

.feature-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 20px;
    height: 100%;
    transition: transform 0.3s ease;
}

.feature-card:hover {
    border-color: var(--accent);
    transform: translateY(-5px);
}

.title-text {
    font-size: 3.5rem;
    font-weight: 800;
    background: -webkit-linear-gradient(#eee, #333);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}

.footer { text-align:center; color: var(--muted); margin-top: 3rem; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# 3) HERO SECTION (BOAS-VINDAS)
st.markdown(
    """
    <div class="hero-section">
        <h1 class="title-text">THE EVERYTHING CALCULATOR</h1>
        <p style="color: var(--muted); font-size: 1.2rem; max-width: 800px; margin: 0 auto;">
            Uma plataforma de computação científica de alto nível, projetada para unir 
            rigor matemático, análise numérica e visualização de dados em tempo real.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4) O QUE É O TEC? (COLUNAS)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎯 O Intuito")
    st.write(
        """
        O **TEC** não é apenas uma calculadora comum. Ele foi desenvolvido para profissionais e 
        estudantes que precisam de **evidência numérica**. 
        
        Enquanto calculadoras padrão entregam apenas o resultado final, o TEC expõe as entranhas 
        do cálculo: ordens de erro, diagnósticos de convergência e comportamento assintótico.
        """
    )

with col2:
    st.markdown("### 🛠️ Core Tecnológico")
    st.markdown(
        """
        - **Engine:** SymPy para manipulação simbólica robusta.
        - **Performance:** Computação vetorizada com NumPy.
        - **Visualização:** Gráficos interativos em Plotly (60fps).
        - **Verificação:** Validação cruzada com bibliotecas padrão (SciPy).
        """
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# 5) O QUE DÁ PRA FAZER? (GRID DE CARDS)
st.subheader("Explore as Ferramentas")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📐 Cálculo Numérico</h4>
            <p class="small-muted">Integração por Riemann, Simpson e Trapézios com análise de erro log-log.</p>
        </div>
        """, unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📈 Análise de Funções</h4>
            <p class="small-muted">Visualização de funções complexas, detecção de descontinuidades e limites.</p>
        </div>
        """, unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🧪 Em breve...</h4>
            <p class="small-muted">Álgebra Linear, Equações Diferenciais e Otimização de Sistemas.</p>
        </div>
        """, unsafe_allow_html=True
    )

# 6) CHAMADA PARA AÇÃO
st.markdown("<br>", unsafe_allow_html=True)
st.info("💡 **Dica:** Utilize o menu lateral para navegar entre os módulos disponíveis.")

# 7) RODAPÉ
st.markdown("<div class='footer'>TEC Engine v1.0 • Desenvolvido por Fellipe Almässy</div>", unsafe_allow_html=True)
