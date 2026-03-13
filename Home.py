import streamlit as st

# ------------------------------------------------------------
# 1) NAVEGAÇÃO — define as seções e páginas do TEC
# ------------------------------------------------------------
pg = st.navigation({
    "Cálculo": [
        st.Page("pages/Calculo/integral_3d.py", title="Integral 3D", icon="📐"),
        st.Page("pages/Calculo/limite.py", title="Limites", icon="🎯"),
    ],
    "Álgebra Linear": [
        st.Page("pages/Algebra_Linear/Vetores.py", title="Vetores", icon="📊"),
    ],
})

pg.run()

# ------------------------------------------------------------
# 2) CONFIGURAÇÃO DA PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="TEC - The Everything Calculator",
    page_icon="🧮",
    layout="wide"
)

# ------------------------------------------------------------
# 3) ESTILO (CSS) - IDENTIDADE VISUAL PREMIUM
# ------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.1);
  --muted: rgba(229,231,235,0.70);
  --muted2: rgba(229,231,235,0.40);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}

/* Fundo da aplicação */
.stApp { background-color: var(--bg); }

/* Hero Section */
.hero-section {
    padding: 4rem 2rem;
    background: radial-gradient(circle at top left, rgba(255,75,75,0.1), transparent),
                radial-gradient(circle at bottom right, rgba(30,144,255,0.1), transparent);
    border-radius: 24px;
    border: 1px solid var(--border);
    margin-bottom: 3rem;
    text-align: center;
}

.title-text {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 0.5rem;
    color: #FFFFFF;
}

/* Feature Cards */
.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.feature-card:hover {
    border-color: var(--accent);
    transform: translateY(-8px);
    background: rgba(255,255,255,0.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.card-icon { font-size: 2rem; margin-bottom: 15px; }
.card-title { color: #FFFFFF; font-size: 1.3rem; font-weight: 700; margin-bottom: 12px; }

/* Contact & Info Cards */
.info-box {
    background: rgba(30,144,255,0.05);
    border-left: 4px solid var(--accent2);
    padding: 20px;
    border-radius: 0 12px 12px 0;
}

.contact-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    padding: 25px;
    border-radius: 16px;
    border: 1px solid var(--border);
}

.hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 3rem 0;
}

.footer { 
    text-align: center; 
    color: var(--muted2); 
    margin-top: 5rem; 
    padding-bottom: 3rem;
    font-size: 0.9rem; 
}

code { color: var(--accent) !important; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 5px;
    background: rgba(255,255,255,0.1);
    font-size: 0.75rem;
    margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 4) HERO SECTION
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero-section">
        <div class="badge">ENGINE v1.1.0 - LIVE</div>
        <h1 class="title-text">THE EVERYTHING CALCULATOR</h1>
        <p style="color: var(--muted); font-size: 1.3rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Uma infraestrutura de computação numérica avançada focada em 
            precisão simbólica, análise de convergência e visualização técnica.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# 5) GRID DE FUNCIONALIDADES
# ------------------------------------------------------------
st.markdown("### 🛠️ Módulos de Matemática")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">📐</div>
            <div class="card-title">Cálculo</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Limites, integrais e análise de convergência com visualização de partições e diagnósticos de erro em tempo real.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">📉</div>
            <div class="card-title">Processos Estocásticos</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Movimento browniano, simulação Monte Carlo e análise do expoente de Hurst para séries temporais.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="feature-card">
            <div class="card-icon">💹</div>
            <div class="card-title">Finanças Quantitativas</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Black-Scholes, precificação de opções, VaR e Greeks. Em desenvolvimento.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 6) ROADMAP E SINTAXE
# ------------------------------------------------------------
col_road, col_syntax = st.columns([1, 1], gap="large")

with col_road:
    st.markdown("### 🚀 Roadmap de Desenvolvimento")
    st.markdown(
        """
        - ✅ **v1.0:** Módulo de Integração Numérica
        - ✅ **v1.1:** Módulo de Limites + Navegação por seções
        - 🔄 **v1.2:** Álgebra Linear
        - 📅 **v1.3:** Finanças Quantitativas (Black-Scholes, VaR)
        - 📅 **v1.4:** Otimização e Fronteira de Markowitz
        """
    )

with col_syntax:
    st.markdown("### ⌨️ Guia de Sintaxe (SymPy)")
    st.markdown("O TEC utiliza o padrão Python/SymPy para interpretação de funções:")
    st.code("""
# Potência: x**2 (não use ^)
# Constantes: pi, E
# Funções: exp(x), log(x), sin(x), cos(x)
# Raiz Quadrada: sqrt(x)
# Valor Absoluto: Abs(x)
    """, language="python")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# 7) CONTATO E INFORMAÇÕES PESSOAIS
# ------------------------------------------------------------
inf_left, inf_right = st.columns([1.5, 1])

with inf_left:
    st.markdown("### 🔍 Sobre o Projeto")
    st.write(
        """
        O **The Everything Calculator (TEC)** nasceu da necessidade de uma ferramenta que não apenas calculasse, 
        mas que ensinasse sobre os modelos. O projeto foca em fornecer uma interface 
        limpa e simples para problemas complexos de engenharia e matemática.
        """
    )
    st.markdown(
        """
        <div class="info-box">
            <strong>Aviso de Rigor:</strong> Os métodos numéricos aqui implementados assumem 
            continuidade. Para funções com singularidades, consulte os avisos de convergência 
            no módulo de diagnósticos.
        </div>
        """, unsafe_allow_html=True
    )

with inf_right:
    st.markdown("### ✉️ Contato Direto")
    st.markdown(
        f"""
        <div class="contact-card">
            <p style="margin-bottom: 10px; font-size: 1.1rem;"><strong>Fellipe Almässy</strong></p>
            <p style="margin-bottom: 8px; font-size: 0.95rem; color: var(--muted);">📧 <a href="mailto:fealmassy@gmail.com" style="color: var(--accent2); text-decoration:none;">fealmassy@gmail.com</a></p>
            <p style="margin-bottom: 8px; font-size: 0.95rem; color: var(--muted);">📱 +55 (11) 91258-3939</p>
            <p style="margin-bottom: 0px; font-size: 0.85rem; color: var(--muted2);">📍 São Paulo - SP, Brasil</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# 8) RODAPÉ
# ------------------------------------------------------------
st.markdown(
    """
    <div class='footer'>
        <strong>TEC Engine v1.1.0</strong> — The Everything Calculator<br>
        Fellipe Almässy • 2026 • 
    </div>
    """,
    unsafe_allow_html=True
)

