# streamlit_app.py
# Mecanismo de Integração Quantitativa
# Fellipe Almässy — Portfolio Project
# ─────────────────────────────────────────────────────────────────────────────
# Arquitetura:
#   • Parsing simbólico via SymPy + avaliação numérica com NumPy (lambdify)
#   • 5 métodos de quadratura: Riemann L/R, Ponto Médio, Trapezoidal, Simpson
#   • Referência de alta precisão via SciPy quad
#   • Diagnósticos de convergência + estimativa de ordem observada (regressão log-log)
#   • Cache agressivo (@st.cache_data / @st.cache_resource) para zero re-computes
#   • Design system próprio via CSS injetado

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st
from scipy.integrate import quad

# ─── 0 · PAGE CONFIG (deve ser a PRIMEIRA chamada Streamlit) ─────────────────
st.set_page_config(
    page_title="Integrais Definidas · Fellipe Almässy",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 1 · DESIGN SYSTEM ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Tokens ── */
:root {
  --bg:         #080b12;
  --surface:    #0f1421;
  --surface2:   #161c2d;
  --border:     rgba(255,255,255,0.06);
  --border2:    rgba(255,255,255,0.10);
  --text:       #e8edf5;
  --muted:      rgba(200,210,230,0.55);
  --muted2:     rgba(200,210,230,0.30);
  --red:        #e8524a;
  --red-glow:   rgba(232,82,74,0.18);
  --blue:       #3d8ef0;
  --blue-glow:  rgba(61,142,240,0.15);
  --green:      #3ecf8e;
  --gold:       #f0b429;
}

/* ── Reset & base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Hide default Streamlit decorations ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem !important; }

/* ── Inputs ── */
input, textarea, select,
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
  background: var(--surface2) !important;
  border-color: var(--border2) !important;
  color: var(--text) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.88rem !important;
  border-radius: 8px !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 14px !important;
  padding: 1.1rem 1.3rem !important;
  transition: border-color 0.2s;
}
div[data-testid="stMetric"]:hover { border-color: rgba(255,255,255,0.18) !important; }
div[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 0.78rem !important; letter-spacing: 0.04em !important; text-transform: uppercase; }
div[data-testid="stMetricValue"]  { font-family: 'JetBrains Mono', monospace !important; font-size: 1.25rem !important; color: var(--text) !important; }
div[data-testid="stMetricDelta"]  { font-size: 0.78rem !important; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border2) !important;
}
button[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  border-radius: 7px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  padding: 0.4rem 1.1rem !important;
  transition: all 0.2s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] iframe { background: var(--surface) !important; }

/* ── Expander ── */
details {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 12px !important;
  padding: 0.2rem 0.5rem !important;
}
summary {
  color: var(--text) !important;
  font-weight: 500 !important;
  cursor: pointer;
  padding: 0.6rem 0.5rem;
}

/* ── Custom components ── */
.page-header {
  padding: 2.5rem 0 0.5rem;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 2rem;
}
.page-header .eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--red);
  text-transform: uppercase;
  letter-spacing: 0.18em;
  margin-bottom: 0.6rem;
}
.page-header h1 {
  font-family: 'Syne', sans-serif !important;
  font-size: clamp(1.8rem, 3vw, 2.6rem) !important;
  font-weight: 800 !important;
  line-height: 1.15 !important;
  letter-spacing: -0.02em;
  margin: 0 0 0.5rem !important;
  color: var(--text) !important;
}
.page-header .subtitle {
  color: var(--muted);
  font-size: 0.97rem;
  line-height: 1.6;
  max-width: 620px;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 0.3rem;
  letter-spacing: -0.01em;
}
.section-sub {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 1rem;
}

.fn-display {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-left: 3px solid var(--red);
  border-radius: 0 12px 12px 0;
  padding: 1.2rem 1.8rem;
  margin: 1.2rem 0 1.8rem;
  text-align: center;
}

.chip {
  display: inline-block;
  padding: 0.18rem 0.6rem;
  border-radius: 999px;
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  margin: 0 0.15rem 0.3rem;
}
.chip.red   { border-color: rgba(232,82,74,0.4);  color: var(--red);   background: var(--red-glow); }
.chip.blue  { border-color: rgba(61,142,240,0.4); color: var(--blue);  background: var(--blue-glow); }
.chip.green { border-color: rgba(62,207,142,0.3); color: var(--green); background: rgba(62,207,142,0.08); }

.theory-card {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
}
.theory-card h3 {
  font-family: 'Syne', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  margin: 0 0 0.5rem;
  color: var(--text);
}
.theory-card .order-badge {
  float: right;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--muted);
}
.theory-card p { color: var(--muted); font-size: 0.9rem; line-height: 1.65; margin: 0 0 0.5rem; }
.theory-card p:last-child { margin-bottom: 0; }
.theory-card .highlight {
  background: var(--surface2);
  border-left: 2px solid var(--gold);
  padding: 0.5rem 0.8rem;
  border-radius: 0 6px 6px 0;
  font-size: 0.85rem;
  color: rgba(200,210,230,0.75);
  margin-top: 0.6rem;
}

.rule-line {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.8rem 0;
}

.footer-bar {
  text-align: center;
  padding: 2rem 0 1rem;
  color: var(--muted2);
  font-size: 0.82rem;
  border-top: 1px solid var(--border);
  margin-top: 3rem;
  font-family: 'JetBrains Mono', monospace;
}

/* Sidebar label style */
.sidebar-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--muted2);
  margin: 1.2rem 0 0.3rem;
  padding-left: 0.1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── 2 · MÉTODOS NUMÉRICOS ────────────────────────────────────────────────────
def riemann_esquerda(f, a, b, n):
    h = (b - a) / n
    return float(np.sum(f(np.linspace(a, b - h, n))) * h)

def riemann_direita(f, a, b, n):
    h = (b - a) / n
    return float(np.sum(f(np.linspace(a + h, b, n))) * h)

def riemann_ponto_medio(f, a, b, n):
    h = (b - a) / n
    return float(np.sum(f(np.linspace(a + h/2, b - h/2, n))) * h)

def trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n + 1); y = f(x); h = (b - a) / n
    return float(h * (np.sum(y) - 0.5 * (y[0] + y[-1])))

def simpson(f, a, b, n):
    if n % 2: n += 1
    x = np.linspace(a, b, n + 1); y = f(x); h = (b - a) / n
    return float((h / 3) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2])))

METODOS: Dict[str, Callable] = {
    "Riemann Esquerda":  riemann_esquerda,
    "Riemann Direita":   riemann_direita,
    "Ponto Médio":       riemann_ponto_medio,
    "Trapezoidal":       trapezoidal,
    "Simpson":           simpson,
}

ORDEM_TEORICA = {
    "Riemann Esquerda": 1, "Riemann Direita": 1,
    "Ponto Médio": 2, "Trapezoidal": 2, "Simpson": 4,
}

# ─── 3 · CACHE LAYER ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def parse_function(expr_str: str):
    x = sp.Symbol("x", real=True)
    ns = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
          "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
          "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi, "e": sp.E}
    expr = sp.sympify(expr_str, locals=ns)
    return expr, sp.lambdify(x, expr, modules=["numpy"])

@st.cache_data(show_spinner=False)
def safe_eval_curve(expr_str: str, a: float, b: float, pts: int = 1600):
    _, f = parse_function(expr_str)
    pad = 0.15 * (b - a)
    xc = np.linspace(a - pad, b + pad, pts)
    try:
        yc = np.array(f(xc), dtype=float)
    except Exception:
        yc = np.array([float(f(xi)) for xi in xc], dtype=float)
    yc[~np.isfinite(yc)] = np.nan
    return xc, yc

@st.cache_data(show_spinner=False)
def compute_quad(expr_str: str, a: float, b: float):
    try:
        _, f = parse_function(expr_str)
        v, e = quad(lambda t: float(f(t)), a, b, limit=300)
        return float(v), float(e)
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def convergence_series(expr_str: str, a: float, b: float, metodo: str, n_max: int, step: int):
    _, f = parse_function(expr_str)
    ref, _ = compute_quad(expr_str, a, b)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)
    ns = np.arange(10, n_max + 1, step, dtype=int)
    fn = METODOS[metodo]
    errs = np.array([abs(ref - fn(f, a, b, int(n))) for n in ns])
    return ns, errs

def observed_order(ns, errs, a, b):
    if len(ns) < 5: return None
    h = (b - a) / ns.astype(float)
    lh = np.log(h); le = np.log(np.maximum(errs, 1e-300))
    m = np.isfinite(lh) & np.isfinite(le)
    if m.sum() < 5: return None
    return float(np.polyfit(lh[m], le[m], 1)[0])


# ─── 4 · TEORIA ──────────────────────────────────────────────────────────────
def render_theory():
    st.markdown("""
<div class="section-title">Como funciona cada método</div>
<div class="section-sub">
  Todos os métodos partem da mesma ideia: dividir o intervalo em <em>n</em> pedaços de largura
  <strong>h = (b−a)/n</strong> e substituir f(x) por algo mais simples dentro de cada pedaço.
  A diferença está em <em>qual</em> aproximação local é usada — e isso determina com que velocidade o erro cai.
</div>
""", unsafe_allow_html=True)

    methods_data = [
        {
            "name": "Somas de Riemann — Esquerda & Direita",
            "order": "O(h¹)",
            "color": "red",
            "body": (
                "A aproximação mais intuitiva: dentro de cada subintervalo, a função é tratada como "
                "uma <strong>constante</strong> — seja o valor no extremo esquerdo, seja no direito. "
                "O resultado é uma coleção de retângulos.<br><br>"
                "O problema: se f(x) tem inclinação, os retângulos sistematicamente sub- ou superestimam. "
                "Dobrar n corta o erro pela metade — convergência linear, ordem 1."
            ),
            "tip": "💡 f crescente → Esquerda subestima, Direita superestima. f decrescente → invertido.",
        },
        {
            "name": "Regra do Ponto Médio",
            "order": "O(h²)",
            "color": "blue",
            "body": (
                "Em vez de usar a borda do subintervalo, usamos o <strong>centro</strong>. "
                "Parece um detalhe pequeno, mas tem um efeito enorme: o erro de subestimação "
                "de um lado cancela o de superestimação do outro dentro de cada intervalo.<br><br>"
                "Resultado: o erro cai com <em>h²</em>. Dobrar n reduz o erro por fator 4. "
                "Surpreendentemente, é melhor que o Trapezoidal para a maioria das funções suaves."
            ),
            "tip": "💡 A cancelação de erros de primeira ordem é a razão pela qual O(h²) emerge.",
        },
        {
            "name": "Regra Trapezoidal",
            "order": "O(h²)",
            "color": "blue",
            "body": (
                "Em vez de uma constante, usamos uma <strong>reta</strong> conectando os dois extremos "
                "de cada subintervalo — formando trapézios. A curvatura de f(x) não é capturada, "
                "mas a tendência linear sim.<br><br>"
                "O erro de truncamento local é proporcional a f''(x), e cancela parcialmente ao somar "
                "todos os subintervalos → ordem global 2."
            ),
            "tip": "💡 Para funções periódicas integradas em um período completo, o Trapezoidal é especialmente preciso.",
        },
        {
            "name": "Regra de Simpson",
            "order": "O(h⁴)",
            "color": "green",
            "body": (
                "Em vez de retas, usamos <strong>parábolas</strong> — cada uma passando por três pontos "
                "consecutivos. A combinação 1-4-2-4-...-4-1 é exatamente o coeficiente que emerge "
                "da integração de uma parábola de Lagrange.<br><br>"
                "O erro cai com h⁴: dobrar n reduz o erro por fator 16. Para funções suaves, "
                "Simpson com poucos subintervalos já supera Riemann com milhares."
            ),
            "tip": "💡 n deve ser par. O erro depende de f⁽⁴⁾(x) — só degrada para funções com curvatura variando muito rápido.",
        },
    ]

    for m in methods_data:
        st.markdown(f"""
<div class="theory-card">
  <span class="order-badge">{m["order"]}</span>
  <h3>{m["name"]}</h3>
  <p>{m["body"]}</p>
  <div class="highlight">{m["tip"]}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='rule-line'></div>", unsafe_allow_html=True)
    st.markdown("""
<div class="section-title">Diagnóstico Log-Log</div>
<div class="section-sub" style="margin-bottom:0.8rem">
  Como comparar a velocidade de convergência de dois métodos com um único número?
</div>
""", unsafe_allow_html=True)
    st.markdown("""
<div class="theory-card">
  <h3>Estimativa da ordem observada</h3>
  <p>Se o erro se comporta como <strong>Erro ≈ C · hᵖ</strong>, então aplicando logaritmo:</p>
  <p style="text-align:center; font-family:'JetBrains Mono',monospace; font-size:0.92rem; padding:0.4rem 0;">
    log(Erro) ≈ p · log(h) + log(C)
  </p>
  <p>
    Ou seja, num gráfico log-log de Erro vs h, a <strong>inclinação da reta</strong> é exatamente a ordem p.
    Este app calcula p numericamente via regressão linear nos pares (log h, log erro) — se o valor
    observado concordar com o teórico, o método está se comportando como esperado para aquela função.
  </p>
  <div class="highlight">
    💡 Discordância entre p observado e teórico é um sinal: a função pode ter singularidades, 
    descontinuidades, ou a convergência assintótica ainda não foi atingida para os n usados.
  </div>
</div>
""", unsafe_allow_html=True)


# ─── 5 · GRÁFICOS ────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c8d2e6"),
    margin=dict(l=8, r=8, t=36, b=8),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
)

def main_plot(expr_str, expr, f_num, a, b, n, show_bars):
    h = (b - a) / n
    xc, yc = safe_eval_curve(expr_str, a, b)
    fig = go.Figure()

    mask = (xc >= a) & (xc <= b)
    fig.add_trace(go.Scatter(
        x=xc[mask], y=yc[mask], fill="tozeroy",
        fillcolor="rgba(232,82,74,0.08)", line=dict(color="transparent"),
        hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter(
        x=xc, y=yc, mode="lines", name="f(x)",
        line=dict(color="#e8524a", width=2.5),
        hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.6f}<extra></extra>"))

    if show_bars:
        xl = np.linspace(a, b - h, n)
        yl = np.array(f_num(xl), dtype=float)
        yl[~np.isfinite(yl)] = np.nan
        fig.add_trace(go.Bar(
            x=xl, y=yl, width=h, name="Partição",
            marker=dict(color="#3d8ef0", opacity=0.45,
                        line=dict(color="rgba(255,255,255,0.20)", width=0.5))))

    fig.add_vline(x=a, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.25)",
                  annotation_text=f"a={a:.3g}", annotation_font_size=11,
                  annotation_font_color="rgba(200,210,230,0.5)")
    fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.25)",
                  annotation_text=f"b={b:.3g}", annotation_font_size=11,
                  annotation_font_color="rgba(200,210,230,0.5)")

    fig.update_layout(**PLOTLY_BASE, title=dict(
        text=f"∫ f(x) dx  ·  n = {n} partições",
        font=dict(family="Syne, sans-serif", size=14, color="#c8d2e6")))
    return fig


def convergence_plot(ns, errs, metodo, loglog, p_obs, p_teo, a, b):
    eps = 1e-300
    y = np.maximum(errs, eps) if loglog else errs
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ns, y=y, mode="lines",
        name=f"Erro — {metodo}",
        line=dict(color="#3d8ef0", width=2.2),
        hovertemplate="n=%{x}<br>erro=%{y:.3e}<extra></extra>"))

    # Linha de referência com inclinação teórica
    if loglog and len(ns) > 5:
        h_ref = (b - a) / ns.astype(float)
        y0 = float(np.maximum(errs[0], eps))
        y_ref = y0 * (h_ref / h_ref[0]) ** p_teo
        fig.add_trace(go.Scatter(
            x=ns, y=np.maximum(y_ref, eps), mode="lines",
            name=f"Referência teórica O(h^{p_teo})",
            line=dict(color="rgba(240,180,41,0.55)", width=1.5, dash="dash"),
            hoverinfo="skip"))

    fig.update_layout(**PLOTLY_BASE,
        title=dict(
            text=f"Convergência · Ordem obs. ≈ {p_obs:.2f}  |  Teórica = {p_teo}",
            font=dict(family="Syne, sans-serif", size=13, color="#c8d2e6")),
        xaxis_title="n  (partições)",
        yaxis_title="Erro Absoluto  |  |Iₙ − quad|",
        legend=dict(orientation="h", y=1.12, x=0))
    if loglog:
        fig.update_layout(xaxis_type="log", yaxis_type="log")
    return fig


# ─── 6 · SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:0.5rem 0 1.2rem">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(200,210,230,0.4);
              text-transform:uppercase;letter-spacing:0.16em;margin-bottom:0.3rem">Portfolio</div>
  <div style="font-family:'Syne',sans-serif;font-size:1.05rem;font-weight:800;color:#e8edf5">
    Fellipe Almässy
  </div>
  <div style="font-size:0.8rem;color:rgba(200,210,230,0.45);margin-top:0.15rem">
    Integrais Definidas
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Função</div>', unsafe_allow_html=True)
    exemplos = {
        "Gaussiana  e^(−x²)":     "exp(-x**2)",
        "Oscilatória suave":       "x**2 * sin(x)",
        "Alta frequência":         "sin(50*x) / (1 + x**2)",
        "Valor absoluto  |x|":     "Abs(x)",
        "Polinômio cúbico":        "x**3 - 3*x + 1",
        "Seno puro":               "sin(x)",
    }
    ex = st.selectbox("Exemplo rápido", list(exemplos.keys()), index=0)
    expr_str = st.text_input("f(x)", value=exemplos[ex],
                              help="Sintaxe SymPy: use sin, cos, exp, log, sqrt, Abs, pi")

    st.markdown('<div class="sidebar-label">Intervalo de integração</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    a = ca.number_input("a", value=-2.0, format="%.4f")
    b = cb.number_input("b", value=2.0, format="%.4f")

    st.markdown('<div class="sidebar-label">Método</div>', unsafe_allow_html=True)
    metodo = st.selectbox("Método principal", list(METODOS.keys()), index=2)

    st.markdown('<div class="sidebar-label">Partições</div>', unsafe_allow_html=True)
    n = st.slider("n", 10, 6000, 500, step=10,
                  help="Mais partições = menor erro, mas mais cálculos.")
    show_bars = st.checkbox("Mostrar barras da partição", value=True)

    st.markdown('<div class="sidebar-label">Diagnósticos de convergência</div>', unsafe_allow_html=True)
    show_conv = st.checkbox("Ativar diagnósticos", value=True)
    n_max  = st.slider("N máximo", 200, 15000, 5000, step=100)
    step   = st.slider("Passo de amostragem", 10, 500, 50, step=10)
    loglog = st.checkbox("Escala log-log", value=True)

    st.markdown("""
<div style="margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);
            font-size:0.78rem;color:rgba(200,210,230,0.35);font-family:'JetBrains Mono',monospace">
  SymPy · NumPy · SciPy · Plotly
</div>""", unsafe_allow_html=True)


# ─── 7 · VALIDAÇÃO ───────────────────────────────────────────────────────────
if a == b:
    st.error("a e b não podem ser iguais.")
    st.stop()
if a > b:
    a, b = b, a
    st.sidebar.warning("a > b: limites invertidos automaticamente.")

try:
    expr, f_num = parse_function(expr_str)
    _ = np.array(f_num(np.array([a, (a+b)/2, b], dtype=float)), dtype=float)
except Exception as e:
    st.error(f"Função inválida — {e}")
    st.stop()


# ─── 8 · REFERÊNCIA ──────────────────────────────────────────────────────────
ref_val, ref_err = compute_quad(expr_str, a, b)


# ─── 9 · CÁLCULO PARA N ATUAL ────────────────────────────────────────────────
rows = []
for nome, fn in METODOS.items():
    t0 = time.perf_counter()
    val = fn(f_num, a, b, n)
    dt = time.perf_counter() - t0
    err = abs(ref_val - val) if ref_val is not None else float("nan")
    rows.append([nome, val, err, dt * 1000, ORDEM_TEORICA.get(nome)])

df = pd.DataFrame(rows, columns=["Método", "Resultado", "Erro Absoluto", "Tempo (ms)", "Ordem p"])
df_sorted = df.sort_values("Erro Absoluto", ascending=True, na_position="last")
primary = float(df[df["Método"] == metodo]["Resultado"].iloc[0])
h = (b - a) / n


# ─── 10 · LAYOUT PRINCIPAL ───────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
  <div class="eyebrow">∫ Métodos de Quadratura Numérica</div>
  <h1>Integrais Definidas</h1>
  <div class="subtitle">
    Aproximação numérica de integrais com 5 métodos clássicos de quadratura,
    análise de convergência e estimativa da ordem observada por regressão log-log.
  </div>
</div>
""", unsafe_allow_html=True)

# Função em destaque
st.markdown('<div class="fn-display">', unsafe_allow_html=True)
st.latex(rf"\displaystyle \int_{{{a:.4g}}}^{{{b:.4g}}} {sp.latex(expr)} \; dx")
st.markdown('</div>', unsafe_allow_html=True)

# Métricas
c1, c2, c3, c4, c5 = st.columns([1.3, 1.2, 1.1, 0.9, 0.9])

c1.metric(f"Resultado — {metodo}", f"{primary:.8f}", f"h = {h:.5g}")
if ref_val is not None:
    c2.metric("Referência SciPy quad", f"{ref_val:.8f}", f"±{ref_err:.1e}")
    c3.metric("Erro Absoluto", f"{abs(ref_val - primary):.4e}", delta_color="inverse")
else:
    c2.metric("SciPy quad", "indisponível", "—")
    c3.metric("Erro Absoluto", "—", "—")
c4.metric("Partições n", f"{n:,}", "O(n) operações")
c5.metric("Ordem teórica p", str(ORDEM_TEORICA.get(metodo, "—")), metodo)

# Tabela comparativa
st.markdown("<div class='rule-line'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="section-title">Comparativo entre todos os métodos</div>
<div class="section-sub">
  Para o mesmo n, como os 5 métodos se comparam? Ordenado por menor erro absoluto.
</div>
""", unsafe_allow_html=True)

st.dataframe(
    df_sorted.style
        .format({"Resultado": "{:.8f}", "Erro Absoluto": "{:.4e}", "Tempo (ms)": "{:.3f}"})
        .background_gradient(subset=["Erro Absoluto"], cmap="RdYlGn_r"),
    use_container_width=True, hide_index=True)

st.markdown("<div class='rule-line'></div>", unsafe_allow_html=True)


# ─── 11 · ABAS ───────────────────────────────────────────────────────────────
tab_viz, tab_conv, tab_teoria = st.tabs([
    "  Visualização  ",
    "  Convergência  ",
    "  Teoria Explicada  ",
])

with tab_viz:
    st.markdown(f"""
<div class="section-sub" style="margin-bottom:0.8rem">
  Gráfico de f(x) no intervalo [a, b] com a área integrada destacada.
  {'As barras azuis mostram os retângulos da partição esquerda.' if show_bars else ''}
</div>""", unsafe_allow_html=True)
    fig = main_plot(expr_str, expr, f_num, a, b, n, show_bars)
    st.plotly_chart(fig, use_container_width=True)

with tab_conv:
    if not show_conv:
        st.info("Ative 'Diagnósticos' na barra lateral para ver a análise de convergência.")
    elif ref_val is None:
        st.warning("A referência SciPy quad não está disponível para esta função. Tente uma função mais suave.")
    else:
        t0 = time.perf_counter()
        ns, errs = convergence_series(expr_str, a, b, metodo, n_max, step)
        dt_conv = time.perf_counter() - t0

        if ns.size == 0:
            st.error("Não foi possível calcular a série de convergência.")
        else:
            p_obs = observed_order(ns, errs, a, b)
            p_teo = ORDEM_TEORICA.get(metodo, 1)
            p_obs_val = p_obs if p_obs is not None else 0.0

            fig_c = convergence_plot(ns, errs, metodo, loglog, p_obs_val, p_teo, a, b)
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("""
<div class="section-sub" style="margin-top:-0.5rem">
  A inclinação da curva no gráfico log-log é a ordem observada. Compare com a ordem teórica esperada para o método.
</div>""", unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns(3)
            if p_obs is not None:
                delta_p = p_obs - p_teo
                cc1.metric("Ordem observada", f"{p_obs:.3f}",
                           f"{'↑' if delta_p > 0 else '↓'} {abs(delta_p):.2f} vs teórico")
            else:
                cc1.metric("Ordem observada", "n/a", "dados insuficientes")
            cc2.metric("Ordem teórica", str(p_teo), f"{metodo}")
            cc3.metric("Runtime", f"{dt_conv*1000:.0f} ms", f"{len(ns)} amostras")

            with st.expander("Ver tabela bruta de convergência"):
                st.dataframe(pd.DataFrame({"n": ns, "erro_abs": errs})
                             .style.format({"erro_abs": "{:.6e}"}),
                             use_container_width=True, hide_index=True)

with tab_teoria:
    render_theory()


# ─── 12 · RODAPÉ ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-bar">
  Fellipe Almässy · Integrais Definidas — Métodos Numéricos &nbsp;·&nbsp;
  SymPy &nbsp;+&nbsp; NumPy &nbsp;+&nbsp; SciPy &nbsp;+&nbsp; Plotly &nbsp;+&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
