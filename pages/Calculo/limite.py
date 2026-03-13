from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA
# ----------------------------
st.set_page_config(page_title="Limites", layout="wide")

# ----------------------------
# 1) ESTILO (CSS) — idêntico ao integral_3d
# ----------------------------
st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.08);
  --muted: rgba(229,231,235,0.60);
  --muted2: rgba(229,231,235,0.40);
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

.hr { border: none; border-top: 1px solid var(--border); margin: 0.75rem 0 1.0rem 0; }
.small-muted { color: var(--muted); font-size: 0.92rem; }
.badge {
  display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
  color: rgba(229,231,235,0.80); font-size: 0.82rem;
}
.footer { text-align:center; color: var(--muted2); margin-top: 14px; font-size: 0.85rem; }
.function-display { text-align: center; padding: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# 2) ENGINE
# ----------------------------
x_sym = sp.Symbol("x", real=True)

@st.cache_resource(show_spinner=False)
def parse_funcao(expressao: str):
    locals_map = {
        "x": x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    try:
        expr = sp.sympify(expressao, locals=locals_map)
    except (sp.SympifyError, SyntaxError):
        raise ValueError(f"Não consegui interpretar `{expressao}`. Use sintaxe SymPy: ex. sin(x)/x, x**2 - 1.")
    f = sp.lambdify(x_sym, expr, modules=["numpy"])
    return f, expr


def calcular_limite(f, ponto: float, delta: float, tolerancia: float) -> dict:
    lim_esq = float(f(ponto - delta))
    lim_dir = float(f(ponto + delta))
    existe   = abs(lim_esq - lim_dir) <= tolerancia
    return {
        "existe":    existe,
        "lim_esq":   lim_esq,
        "lim_dir":   lim_dir,
        "lim_final": (lim_esq + lim_dir) / 2 if existe else None,
        "delta_usado": delta,
    }


def limite_simbolico(expr, ponto: float):
    try:
        return sp.limit(expr, x_sym, ponto)
    except Exception:
        return None


# ----------------------------
# 3) PAINEL TEÓRICO
# ----------------------------
def theory_panel():
    st.markdown("### Teoria Explicada")
    st.markdown(
        "<span class='badge'>Definição de Limite</span> "
        "<span class='badge'>Limites Laterais</span> "
        "<span class='badge'>Método Numérico</span>",
        unsafe_allow_html=True,
    )
    with st.expander("Abrir teoria completa (definição, laterais, método numérico)", expanded=False):
        st.markdown("### Definição Formal (ε-δ)")
        st.markdown("Dizemos que o limite de f(x) quando x tende a x₀ é L se:")
        st.latex(r"\forall\,\varepsilon > 0,\;\exists\,\delta > 0 \;\text{ tal que }\; 0 < |x - x_0| < \delta \;\Rightarrow\; |f(x) - L| < \varepsilon")
        st.markdown("O ponto x₀ em si **não precisa estar no domínio** de f — o limite descreve o comportamento de aproximação, não o valor em x₀.")

        st.markdown("---")
        st.markdown("### Limites Laterais")
        st.markdown("O limite existe se e somente se os dois limites laterais existem **e são iguais**:")
        st.latex(r"\lim_{x \to x_0} f(x) = L \iff \lim_{x \to x_0^-} f(x) = \lim_{x \to x_0^+} f(x) = L")
        st.markdown("Caso clássico onde o limite **não existe**: função sinal em x=0, onde o limite pela esquerda é -1 e pela direita é +1.")

        st.markdown("---")
        st.markdown("### Método Numérico Usado Aqui")
        st.markdown("Este módulo estima o limite avaliando f em dois pontos próximos de x₀:")
        st.latex(r"\text{lim esq} \approx f(x_0 - \delta), \qquad \text{lim dir} \approx f(x_0 + \delta)")
        st.markdown("Se a diferença entre os dois for menor que a tolerância definida, o limite é estimado como a média:")
        st.latex(r"L \approx \frac{f(x_0 - \delta) + f(x_0 + \delta)}{2}")
        st.markdown("""
**Limitações do método:**
- δ muito grande → estimativa imprecisa (não captura o comportamento local)
- δ muito pequeno → erros de ponto flutuante
- Funções com oscilação rápida perto de x₀ podem enganar o método

**O SymPy calcula o limite simbólico exato** quando possível — compare os dois resultados para validar.
        """)


# ----------------------------
# 4) GRÁFICO
# ----------------------------
def gerar_grafico(f, expr, ponto: float, resultado: dict) -> go.Figure:
    margem = max(2.0, abs(ponto) * 0.8) if ponto != 0 else 3.0
    xs = np.linspace(ponto - margem, ponto + margem, 800)

    try:
        ys = np.array(f(xs), dtype=float)
    except Exception:
        ys = np.array([float(f(xi)) for xi in xs], dtype=float)

    if np.any(np.isfinite(ys)):
        lim_y = np.nanpercentile(np.abs(ys[np.isfinite(ys)]), 99) * 3
        ys = np.where(np.isfinite(ys) & (np.abs(ys) < lim_y), ys, np.nan)

    fig = go.Figure()

    # Glow
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip", showlegend=False,
    ))

    # Curva principal
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        name="f(x)",
        line=dict(color="#FF4B4B", width=3),
        hovertemplate="x=%{x:.4f}<br>f(x)=%{y:.4f}<extra></extra>",
    ))

    # Linha vertical em x₀
    fig.add_vline(x=ponto, line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5))

    delta = resultado.get("delta_usado", 0.001)

    # Limite pela esquerda
    fig.add_trace(go.Scatter(
        x=[ponto - delta], y=[resultado["lim_esq"]],
        mode="markers",
        name=f"lim esq = {resultado['lim_esq']:.4f}",
        marker=dict(color="#1E90FF", size=11, symbol="circle"),
    ))

    # Limite pela direita
    fig.add_trace(go.Scatter(
        x=[ponto + delta], y=[resultado["lim_dir"]],
        mode="markers",
        name=f"lim dir = {resultado['lim_dir']:.4f}",
        marker=dict(color="#FFB347", size=11, symbol="circle"),
    ))

    # Limite final
    if resultado["existe"]:
        fig.add_trace(go.Scatter(
            x=[ponto], y=[resultado["lim_final"]],
            mode="markers",
            name=f"limite ≈ {resultado['lim_final']:.4f}",
            marker=dict(color="#00FF99", size=14, symbol="star"),
        ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=450),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


# ----------------------------
# 5) CABEÇALHO
# ----------------------------
st.title("Limites")
st.caption("Estimativa numérica e cálculo simbólico exato via SymPy")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

theory_panel()

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ----------------------------
# 6) SIDEBAR
# ----------------------------
st.sidebar.header("Controles")

exemplos = {
    "sin(x)/x  →  x=0":      ("sin(x)/x",        0.0),
    "(x²-4)/(x-2)  →  x=2":  ("(x**2-4)/(x-2)",  2.0),
    "exp(-x²)  →  x=0":      ("exp(-x**2)",       0.0),
    "|x|/x  →  x=0":         ("Abs(x)/x",         0.0),
    "1/x  →  x=0":           ("1/x",              0.0),
}

exemplo_escolhido = st.sidebar.selectbox("Exemplos rápidos", list(exemplos.keys()))
default_expr, default_ponto = exemplos[exemplo_escolhido]

expressao = st.sidebar.text_input("f(x) (Sintaxe SymPy)", value=default_expr)
ponto     = st.sidebar.number_input("Ponto x₀", value=default_ponto, step=0.1, format="%.4f")

st.sidebar.markdown("---")
delta      = st.sidebar.number_input("δ (delta)", value=0.001, min_value=1e-10, max_value=1.0, format="%.6f",
                                      help="Distância ao redor de x₀ usada para estimar os limites laterais.")
tolerancia = st.sidebar.number_input("Tolerância", value=0.01, min_value=1e-10, max_value=10.0, format="%.6f",
                                      help="Diferença máxima entre limites laterais para aceitar que o limite existe.")

st.sidebar.markdown("---")
st.sidebar.caption("Nota: funções com descontinuidades severas podem enganar o método numérico. Compare sempre com o resultado simbólico.")

calcular = st.sidebar.button("Calcular limite", type="primary", use_container_width=True)

# ----------------------------
# 7) PARSE
# ----------------------------
try:
    f, expr_sym = parse_funcao(expressao)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ----------------------------
# 8) DISPLAY DA FUNÇÃO (sempre visível)
# ----------------------------
st.markdown("<div class='function-display'>", unsafe_allow_html=True)
st.latex(r"f(x) = " + sp.latex(expr_sym))
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# 9) CÁLCULO E RESULTADO
# ----------------------------
if calcular:
    try:
        resultado = calcular_limite(f, ponto, delta=delta, tolerancia=tolerancia)
    except Exception as e:
        st.error(f"Erro ao avaliar a função: {e}")
        st.stop()

    lim_sim = limite_simbolico(expr_sym, ponto)

    # Métricas
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Resultado numérico",
        f"{resultado['lim_final']:.6f}" if resultado["existe"] else "Não existe",
    )
    m2.metric("Limite pela esquerda", f"{resultado['lim_esq']:.6f}")
    m3.metric(
        "Limite pela direita",
        f"{resultado['lim_dir']:.6f}",
        delta=f"Δ = {abs(resultado['lim_esq'] - resultado['lim_dir']):.2e}",
        delta_color="normal" if resultado["existe"] else "inverse",
    )
    m4.metric(
        "Resultado simbólico (SymPy)",
        str(lim_sim) if lim_sim is not None else "n/a",
    )

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Banner
    if resultado["existe"]:
        st.success("✓ O limite existe")
        st.latex(rf"\lim_{{x \to {ponto}}} f(x) \approx {resultado['lim_final']:.6f}")
        if lim_sim is not None:
            st.latex(rf"\lim_{{x \to {ponto}}} f(x) = {sp.latex(lim_sim)} \quad \text{{(SymPy, exato)}}")
    else:
        st.error("✗ O limite não existe — os limites laterais divergem além da tolerância definida.")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Gráfico
    st.markdown("<div class='small-muted'>f(x) em torno de x₀:</div>", unsafe_allow_html=True)
    try:
        fig = gerar_grafico(f, expr_sym, ponto, resultado)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Não foi possível gerar o gráfico: {e}")

# ----------------------------
# 10) RODAPÉ
# ----------------------------
st.markdown("<div class='footer'>The Everything Calculator — Fellipe Almässy • </div>", unsafe_allow_html=True)
