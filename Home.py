import streamlit as st

# ------------------------------------------------------------
# 1) CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="TEC - The Everything Calculator",
    page_icon="🧮",
    layout="wide"
)

# ------------------------------------------------------------
# 2) ESTILO (CSS) - IDENTIDADE VISUAL INSTITUCIONAL
# ------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.08);
  --muted: rgba(229,231,235,0.60);
  --muted2: rgba(229,231,235,0.40);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}

/* Estilização Geral */
.main { background-color: var(--bg); }

/* Hero Section */
.hero-section {
    padding: 3.5rem 2rem;
    background: linear-gradient(135deg, rgba(255,75,75,0.08) 0%, rgba(30,144,255,0.08) 100%);
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 2.5rem;
    text-align: center;
}

.title-text {
    font-size: 3.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #FFFFFF, #888888);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Feature Cards */
.feature-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    height: 100%;
    transition: all 0.3s ease;
}

.feature-card:hover {
    border-color: var(--accent);
    transform: translateY(-5px);
    background: rgba(255,255,255,0.06);
}

.card-title {
    color: var(--accent);
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 10px;
}

/* Contact Card */
.contact-card {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-top: 10px;
}

.hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

.footer { 
    text-align: center; 
    color: var(--muted2); 
    margin-top: 4rem; 
    padding-bottom: 2rem;
    font-size: 0.85rem; 
}

a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 3) HERO SECTION
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero-section">
        <h1 class="title-text">THE EVERYTHING CALCULATOR</h1>
        <p style="color: var(--muted); font-size: 1.25rem; max-width: 850px; margin: 0 auto;">
            Ambiente avançado de computação científica para análise numérica, 
            simulação de dados e rigor matemático aplicado.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 4) CONTEÚDO PRINCIPAL: O QUE É O TEC?
# ------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 🎯 O Intuito")
    st.markdown(
        """
        O **TEC** foi concebido como uma ferramenta de diagnóstico profundo. 
        Diferente de calculadoras convencionais que operam como "caixas-pretas", 
        este motor expõe a matemática por trás dos resultados.
        
        O objetivo é fornecer transparência absoluta em métodos de aproximação, 
        permitindo ao usuário não apenas encontrar um valor, mas compreender a 
        **convergência**, a **estabilidade** e o **erro** inerente ao processo.
        """
    )

with col_right:
    st.markdown("### 🛠️ Core Tecnológico")
    st.markdown(
        """
        * **Precisão Simbólica:** Integração com SymPy para manipulação exata de expressões.
        * **Análise Numérica:** Implementação de algoritmos clássicos (Riemann, Simpson, Trapézios).
        * **Visualização Dinâmica:** Renderização via Plotly para inspeção de curvas e partições.
        * **Benchmark:** Comparação em tempo real com referências de alto desempenho (SciPy quad).
        """
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 5) GRID DE FUNCIONALIDADES
# ------------------------------------------------------------
st.subheader("Módulos Disponíveis")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-title">📐 Integrais Definidas</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Cálculo de áreas sob curvas com múltiplos métodos, análise de erro log-log e visualização de partições.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-title">📈 Análise de Erro</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Estimativa da ordem de convergência (slope) observada versus a teoria assintótica esperada.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-title">🧪 Futuros Módulos</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Expansão prevista para Álgebra Linear Computacional, EDOs e Processamento de Sinais.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 6) CONTATO E RODAPÉ
# ------------------------------------------------------------
inf1, inf2 = st.columns([2, 1])

with inf1:
    st.markdown("### 🔍 Sobre o Desenvolvimento")
    st.write(
        """
        Este projeto é mantido sob uma filosofia de código limpo e transparência científica. 
        Cada módulo é testado para garantir que a interface Streamlit responda com a menor latência 
        possível, mesmo em cálculos de alta densidade de partições.
        """
    )

with inf2:
    st.markdown("### ✉️ Contato")
    st.markdown(
        f"""
        <div class="contact-card">
            <p style="margin-bottom: 8px;"><strong>Fellipe Almässy</strong></p>
            <p style="margin-bottom: 8px; font-size: 0.9rem;">📧 <a href="mailto:fealmassy@gmail.com">fealmassy@gmail.com</a></p>
            <p style="margin-bottom: 8px; font-size: 0.9rem;">📱 (11) 91258-3939</p>
            <p style="margin-bottom: 0px; font-size: 0.8rem; color: var(--muted2);">São Paulo, Brasil</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class='footer'>
        <strong>TEC Engine v1.0</strong> — The Everything Calculator<br>
        Fellipe Almässy • São Paulo, SP • 2026
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# 7) CTA LATERAL
# ------------------------------------------------------------
st.sidebar.success("Selecione um módulo acima para começar.")
st.sidebar.markdown("---")
st.sidebar.caption("Status do Sistema: Operacional")
