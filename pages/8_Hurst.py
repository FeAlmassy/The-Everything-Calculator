# pages/Taylor_Series.py
# Séries de Taylor & Maclaurin — Página Completa
# Fellipe Almässy · Portfolio
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

# ─── 0 · PAGE CONFIG ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Séries de Taylor · Fellipe Almässy",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 1 · DESIGN SYSTEM ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300;1,9..40,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:         #07090f;
  --bg2:        #0c0f1a;
  --surface:    #111624;
  --surface2:   #171d2e;
  --surface3:   #1d2438;
  --border:     rgba(255,255,255,0.055);
  --border2:    rgba(255,255,255,0.10);
  --text:       #dde3f0;
  --muted:      rgba(200,212,235,0.52);
  --muted2:     rgba(200,212,235,0.28);
  --gold:       #e8b84b;
  --gold-dim:   rgba(232,184,75,0.12);
  --gold-border:rgba(232,184,75,0.30);
  --teal:       #3ecfb2;
  --teal-dim:   rgba(62,207,178,0.10);
  --red:        #e8524a;
  --red-dim:    rgba(232,82,74,0.10);
  --blue:       #5b9cf6;
  --blue-dim:   rgba(91,156,246,0.10);
  --purple:     #a78bfa;
  --purple-dim: rgba(167,139,250,0.10);
}

/* ── reset ── */
html,body,[class*="css"]{
  font-family:'DM Sans',sans-serif !important;
  background:var(--bg) !important;
  color:var(--text) !important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 !important; max-width:100% !important;}
section[data-testid="stSidebar"]{display:none}

/* ── scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.10);border-radius:3px}

/* ── typography ── */
.display{
  font-family:'Playfair Display',Georgia,serif;
  font-size:clamp(2.6rem,5vw,4.8rem);
  font-weight:900;
  line-height:1.08;
  letter-spacing:-0.025em;
  color:var(--text);
}
.display em{font-style:italic;color:var(--gold)}
.lead{
  font-size:1.05rem;
  color:var(--muted);
  line-height:1.78;
  max-width:680px;
}
.section-eyebrow{
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;
  text-transform:uppercase;
  letter-spacing:0.20em;
  color:var(--gold);
  margin-bottom:0.55rem;
  display:block;
}
.section-title{
  font-family:'Playfair Display',serif;
  font-size:clamp(1.5rem,2.5vw,2.1rem);
  font-weight:700;
  line-height:1.2;
  letter-spacing:-0.015em;
  color:var(--text);
  margin-bottom:0.5rem;
}
.section-sub{
  color:var(--muted);
  font-size:0.93rem;
  line-height:1.72;
  margin-bottom:1.4rem;
}
.prose{
  color:var(--muted);
  font-size:0.95rem;
  line-height:1.80;
}
.prose strong{color:var(--text);font-weight:500}
.prose em{color:var(--gold);font-style:italic}

/* ── layout wrappers ── */
.page-wrap{max-width:1280px;margin:0 auto;padding:0 2.5rem 6rem}

/* ── hero ── */
.hero{
  padding:5rem 0 4rem;
  border-bottom:1px solid var(--border2);
  margin-bottom:4rem;
  position:relative;
  overflow:hidden;
}
.hero-grid{display:grid;grid-template-columns:1fr 420px;gap:4rem;align-items:center}
.hero-badge{
  display:inline-flex;align-items:center;gap:0.5rem;
  padding:0.28rem 0.85rem;border-radius:999px;
  background:var(--gold-dim);border:1px solid var(--gold-border);
  font-family:'JetBrains Mono',monospace;
  font-size:0.70rem;color:var(--gold);
  text-transform:uppercase;letter-spacing:0.14em;
  margin-bottom:1.5rem;
}
.hero-stats{
  display:flex;gap:2rem;margin-top:2.5rem;
  padding-top:2rem;border-top:1px solid var(--border);
}
.hero-stat .num{
  font-family:'Playfair Display',serif;
  font-size:2rem;font-weight:700;color:var(--gold);
  line-height:1;
}
.hero-stat .lab{
  font-size:0.78rem;color:var(--muted2);
  margin-top:0.3rem;font-family:'JetBrains Mono',monospace;
  text-transform:uppercase;letter-spacing:0.10em;
}

/* ── formula hero card ── */
.formula-hero{
  background:var(--surface);
  border:1px solid var(--border2);
  border-radius:20px;
  padding:2rem;
  text-align:center;
  position:relative;
}
.formula-hero::before{
  content:'';position:absolute;inset:0;border-radius:20px;
  background:radial-gradient(ellipse at 50% 0%,rgba(232,184,75,0.06) 0%,transparent 70%);
  pointer-events:none;
}
.formula-label{
  font-family:'JetBrains Mono',monospace;
  font-size:0.68rem;text-transform:uppercase;letter-spacing:0.14em;
  color:var(--gold);margin-bottom:1rem;display:block;
}

/* ── concept cards ── */
.card-grid{display:grid;gap:1px;background:var(--border2);border-radius:16px;overflow:hidden}
.card-grid-2{grid-template-columns:1fr 1fr}
.card-grid-3{grid-template-columns:1fr 1fr 1fr}
.card{background:var(--surface);padding:1.6rem 1.8rem}
.card:first-child{border-radius:16px 16px 0 0}
.card-grid-2 .card:first-child{border-radius:16px 0 0 0}
.card-grid-2 .card:nth-child(2){border-radius:0 16px 0 0}
.card-grid-2 .card:nth-last-child(2){border-radius:0 0 0 16px}
.card-grid-2 .card:last-child{border-radius:0 0 16px 0}
.card-icon{font-size:1.6rem;margin-bottom:0.8rem;display:block}
.card-title{
  font-family:'Playfair Display',serif;
  font-size:1.05rem;font-weight:700;
  color:var(--text);margin-bottom:0.5rem;
}
.card p{font-size:0.88rem;color:var(--muted);line-height:1.65;margin:0}

/* ── highlight box ── */
.highlight-box{
  border-radius:12px;padding:1.3rem 1.6rem;margin:1.4rem 0;
}
.highlight-box.gold{background:var(--gold-dim);border:1px solid var(--gold-border)}
.highlight-box.teal{background:var(--teal-dim);border:1px solid rgba(62,207,178,0.25)}
.highlight-box.red {background:var(--red-dim); border:1px solid rgba(232,82,74,0.25)}
.highlight-box.blue{background:var(--blue-dim);border:1px solid rgba(91,156,246,0.25)}
.highlight-box.purple{background:var(--purple-dim);border:1px solid rgba(167,139,250,0.25)}
.highlight-box .hb-label{
  font-family:'JetBrains Mono',monospace;font-size:0.68rem;
  text-transform:uppercase;letter-spacing:0.13em;margin-bottom:0.45rem;display:block;
}
.highlight-box.gold   .hb-label{color:var(--gold)}
.highlight-box.teal   .hb-label{color:var(--teal)}
.highlight-box.red    .hb-label{color:var(--red)}
.highlight-box.blue   .hb-label{color:var(--blue)}
.highlight-box.purple .hb-label{color:var(--purple)}
.highlight-box p{font-size:0.90rem;color:var(--muted);line-height:1.68;margin:0}
.highlight-box p strong{color:var(--text)}

/* ── step flow ── */
.step-row{display:flex;gap:0;margin:1.5rem 0}
.step-box{
  flex:1;background:var(--surface);border:1px solid var(--border2);
  padding:1.2rem;text-align:center;position:relative;
}
.step-box:first-child{border-radius:12px 0 0 12px}
.step-box:last-child {border-radius:0 12px 12px 0}
.step-box:not(:last-child)::after{
  content:'→';position:absolute;right:-14px;top:50%;
  transform:translateY(-50%);color:var(--gold);font-size:1.1rem;z-index:2;
}
.step-num{
  font-family:'JetBrains Mono',monospace;font-size:0.62rem;
  color:var(--gold);text-transform:uppercase;letter-spacing:0.12em;
  display:block;margin-bottom:0.3rem;
}
.step-box strong{font-family:'Playfair Display',serif;font-size:0.95rem;
  font-weight:700;display:block;margin-bottom:0.25rem;color:var(--text)}
.step-box p{font-size:0.78rem;color:var(--muted2);margin:0;line-height:1.45}

/* ── proof block ── */
.proof-block{
  background:var(--surface2);border:1px solid var(--border2);
  border-left:3px solid var(--gold);border-radius:0 12px 12px 0;
  padding:1.4rem 1.6rem;margin:1.2rem 0;
}
.proof-label{
  font-family:'JetBrains Mono',monospace;font-size:0.68rem;
  text-transform:uppercase;letter-spacing:0.13em;color:var(--gold);
  margin-bottom:0.6rem;display:block;
}
.proof-block p{font-size:0.90rem;color:var(--muted);line-height:1.72;margin:0 0 0.5rem}
.proof-block p:last-child{margin:0}

/* ── section divider ── */
.sec-divider{
  border:none;border-top:1px solid var(--border);
  margin:4rem 0;
}

/* ── application card ── */
.app-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1.5rem 0}
.app-card{
  background:var(--surface);border:1px solid var(--border2);
  border-radius:14px;padding:1.4rem;
  transition:border-color 0.25s,transform 0.25s;
}
.app-card:hover{border-color:var(--gold-border);transform:translateY(-3px)}
.app-icon{font-size:1.8rem;margin-bottom:0.7rem;display:block}
.app-card h4{
  font-family:'Playfair Display',serif;font-size:1rem;
  font-weight:700;color:var(--text);margin-bottom:0.45rem;
}
.app-card p{font-size:0.84rem;color:var(--muted);line-height:1.60;margin:0}

/* ── table ── */
.styled-table{width:100%;border-collapse:collapse;font-size:0.87rem;margin:1rem 0}
.styled-table th{
  text-align:left;padding:0.6rem 1rem;
  background:var(--surface2);color:var(--gold);
  font-family:'JetBrains Mono',monospace;font-size:0.72rem;
  text-transform:uppercase;letter-spacing:0.10em;
  border-bottom:1px solid var(--border2);
}
.styled-table td{
  padding:0.6rem 1rem;border-bottom:1px solid var(--border);
  color:var(--muted);font-family:'JetBrains Mono',monospace;font-size:0.84rem;
}
.styled-table tr:hover td{background:var(--surface2)}

/* ── footer ── */
.footer{
  text-align:center;padding:2.5rem 0 1rem;
  border-top:1px solid var(--border);margin-top:4rem;
  font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--muted2);
  letter-spacing:0.06em;
}

/* ── streamlit overrides ── */
div[data-testid="stTabs"]{margin-top:0 !important}
div[data-baseweb="tab-list"]{
  background:var(--surface) !important;
  border:1px solid var(--border2) !important;
  border-radius:12px !important;padding:4px !important;gap:2px !important;
}
button[data-baseweb="tab"]{
  background:transparent !important;color:var(--muted) !important;
  border-radius:9px !important;font-family:'DM Sans',sans-serif !important;
  font-size:0.86rem !important;font-weight:500 !important;
  padding:0.42rem 1.2rem !important;transition:all 0.2s !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  background:var(--surface2) !important;color:var(--text) !important;
  border:1px solid var(--border2) !important;
}
div[data-testid="stSlider"] label{
  color:var(--muted) !important;font-size:0.85rem !important;
}
div[data-testid="stSelectbox"] label{
  color:var(--muted) !important;font-size:0.85rem !important;
}
div[data-baseweb="select"]>div:first-child{
  background:var(--surface2) !important;
  border-color:var(--border2) !important;
  color:var(--text) !important;
  border-radius:10px !important;
}
div[data-baseweb="popover"] li{background:var(--surface2) !important}
.stSlider [data-baseweb="slider"]{padding:0.5rem 0}
div[data-testid="stMetric"]{
  background:var(--surface) !important;
  border:1px solid var(--border2) !important;
  border-radius:14px !important;padding:1.1rem 1.3rem !important;
}
div[data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:0.75rem !important;text-transform:uppercase;letter-spacing:0.06em}
div[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace !important;font-size:1.3rem !important}
</style>
""", unsafe_allow_html=True)


# ─── 2 · HELPERS ─────────────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#9aacc8", size=12),
    margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
               zeroline=True, zerolinecolor="rgba(255,255,255,0.12)", zerolinewidth=1),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
               zeroline=True, zerolinecolor="rgba(255,255,255,0.12)", zerolinewidth=1),
)

GOLD   = "#e8b84b"
TEAL   = "#3ecfb2"
RED    = "#e8524a"
BLUE   = "#5b9cf6"
PURPLE = "#a78bfa"
COLORS = [GOLD, TEAL, BLUE, PURPLE, RED,
          "#f97316","#34d399","#fb7185","#60a5fa","#c084fc"]

@st.cache_data(show_spinner=False)
def taylor_coeffs(func_name: str, n_terms: int, a: float):
    x = sp.Symbol("x", real=True)
    funcs = {
        "eˣ":       sp.exp(x),
        "sin(x)":   sp.sin(x),
        "cos(x)":   sp.cos(x),
        "ln(1+x)":  sp.log(1 + x),
        "1/(1-x)":  1 / (1 - x),
        "tan(x)":   sp.tan(x),
        "sinh(x)":  sp.sinh(x),
        "cosh(x)":  sp.cosh(x),
        "arctan(x)":sp.atan(x),
        "√(1+x)":   sp.sqrt(1 + x),
    }
    expr = funcs[func_name]
    coeffs = []
    deriv = expr
    for k in range(n_terms):
        val = float(deriv.subs(x, a))
        coeffs.append(val / sp.factorial(k))
        deriv = sp.diff(deriv, x)
    return coeffs

@st.cache_data(show_spinner=False)
def eval_taylor(coeffs, a, xarr):
    result = np.zeros_like(xarr, dtype=float)
    for k, c in enumerate(coeffs):
        result += c * (xarr - a)**k
    return result

@st.cache_data(show_spinner=False)
def true_function(func_name: str, xarr):
    x = sp.Symbol("x")
    funcs = {
        "eˣ":       sp.exp(x),
        "sin(x)":   sp.sin(x),
        "cos(x)":   sp.cos(x),
        "ln(1+x)":  sp.log(1 + x),
        "1/(1-x)":  1 / (1 - x),
        "tan(x)":   sp.tan(x),
        "sinh(x)":  sp.sinh(x),
        "cosh(x)":  sp.cosh(x),
        "arctan(x)":sp.atan(x),
        "√(1+x)":   sp.sqrt(1 + x),
    }
    f = sp.lambdify(x, funcs[func_name], modules=["numpy"])
    try:
        y = np.array(f(xarr), dtype=float)
    except Exception:
        y = np.array([float(f(xi)) for xi in xarr], dtype=float)
    y[~np.isfinite(y)] = np.nan
    return y

@st.cache_data(show_spinner=False)
def get_sympy_series(func_name: str, n_terms: int, a: float):
    x = sp.Symbol("x")
    funcs = {
        "eˣ":       sp.exp(x),
        "sin(x)":   sp.sin(x),
        "cos(x)":   sp.cos(x),
        "ln(1+x)":  sp.log(1 + x),
        "1/(1-x)":  1 / (1 - x),
        "tan(x)":   sp.tan(x),
        "sinh(x)":  sp.sinh(x),
        "cosh(x)":  sp.cosh(x),
        "arctan(x)":sp.atan(x),
        "√(1+x)":   sp.sqrt(1 + x),
    }
    series_expr = sp.series(funcs[func_name], x, a, n_terms + 1).removeO()
    return sp.latex(series_expr), str(series_expr)

CONVERGENCE_RADII = {
    "eˣ":       ("∞", "Converge para todo x ∈ ℝ"),
    "sin(x)":   ("∞", "Converge para todo x ∈ ℝ"),
    "cos(x)":   ("∞", "Converge para todo x ∈ ℝ"),
    "ln(1+x)":  ("1", "Converge para x ∈ (−1, 1]"),
    "1/(1-x)":  ("1", "Converge para x ∈ (−1, 1)"),
    "tan(x)":   ("π/2", "Converge para |x| < π/2"),
    "sinh(x)":  ("∞", "Converge para todo x ∈ ℝ"),
    "cosh(x)":  ("∞", "Converge para todo x ∈ ℝ"),
    "arctan(x)":("1", "Converge para x ∈ [−1, 1]"),
    "√(1+x)":   ("1", "Converge para x ∈ (−1, 1]"),
}


# ─── 3 · INÍCIO DA PÁGINA ────────────────────────────────────────────────────
st.markdown('<div class="page-wrap">', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-grid">
    <div>
      <div class="hero-badge">∑ Análise Matemática Avançada</div>
      <div class="display">Séries de<br><em>Taylor</em></div>
      <div style="height:1.2rem"></div>
      <div class="lead">
        A ideia mais elegante do cálculo: qualquer função suficientemente suave pode ser
        <strong style="color:#dde3f0">representada exatamente</strong> por uma soma infinita de monômios.
        Aqui você vai entender <em>por quê</em> isso funciona, <em>quando</em> falha,
        e <em>onde</em> o mundo real depende disso todo dia.
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><div class="num">1715</div><div class="lab">Brook Taylor</div></div>
        <div class="hero-stat"><div class="num">∞</div><div class="lab">Termos na série completa</div></div>
        <div class="hero-stat"><div class="num">O(hⁿ)</div><div class="lab">Erro do resto de Lagrange</div></div>
      </div>
    </div>
    <div>""", unsafe_allow_html=True)

st.markdown("""
      <div class="formula-hero">
        <span class="formula-label">Fórmula Geral · Série de Taylor em torno de x = a</span>""",
unsafe_allow_html=True)
st.latex(r"""
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n
""")
st.markdown("""
        <div style="height:0.8rem"></div>""", unsafe_allow_html=True)
st.latex(r"""
= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots
""")
st.markdown("""
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── SEÇÃO 1 · MOTIVAÇÃO ──────────────────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 01 — Motivação</span>
<div class="section-title">Por que polinômios?</div>
<div class="section-sub">
  Antes de qualquer fórmula, é preciso entender qual problema a série de Taylor resolve.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="prose">
  Polinômios são os objetos mais fáceis de manipular em toda a análise matemática. Diferenciá-los,
  integrá-los, avaliá-los em qualquer ponto — tudo se reduz a operações aritméticas elementares.
  Funções como <strong>sin(x)</strong>, <strong>eˣ</strong> e <strong>ln(x)</strong>, por outro lado,
  exigem definições circulares ou séries infinitas para serem calculadas com precisão.<br><br>
  A pergunta que Taylor (e antes dele Newton, Gregory e Leibniz) se fez foi provocadora:
  <em>dado um ponto a, existe um polinômio que "imita" f(x) tão bem quanto quisermos perto de a?</em>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="step-row">
  <div class="step-box">
    <span class="step-num">Grau 0</span>
    <strong>Constante</strong>
    <p>P₀(x) = f(a)<br>Mesmo valor em a</p>
  </div>
  <div class="step-box">
    <span class="step-num">Grau 1</span>
    <strong>Linear</strong>
    <p>P₁ = P₀ + f'(a)(x−a)<br>Mesma inclinação</p>
  </div>
  <div class="step-box">
    <span class="step-num">Grau 2</span>
    <strong>Quadrático</strong>
    <p>P₂ = P₁ + f''(a)/2·(x−a)²<br>Mesma curvatura</p>
  </div>
  <div class="step-box">
    <span class="step-num">Grau n</span>
    <strong>Ordem n</strong>
    <p>Pₙ = Σ f⁽ᵏ⁾(a)/k! · (x−a)ᵏ<br>k = 0..n</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box gold">
  <span class="hb-label">💡 A ideia central</span>
  <p>Cada termo adicionado impõe uma nova condição: o polinômio passa a ter a mesma k-ésima derivada
  que f(x) no ponto a. Com n termos, garantimos que <strong>P⁽ᵏ⁾(a) = f⁽ᵏ⁾(a)</strong> para k = 0, 1, …, n.
  No limite n→∞, se a série convergir, temos igualdade perfeita.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 2 · DERIVAÇÃO FORMAL ──────────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 02 — Derivação Formal</span>
<div class="section-title">De onde vem a fórmula</div>
<div class="section-sub">
  A construção do polinômio de Taylor não é arbitrária — é a única possível dado o conjunto de restrições.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="prose">
  Suponha que f(x) seja infinitamente diferenciável em a e que exista um polinômio
  <strong>P(x) = c₀ + c₁(x−a) + c₂(x−a)² + ⋯</strong> que satisfaça
  P⁽ᵏ⁾(a) = f⁽ᵏ⁾(a) para todo k. O que são os coeficientes cₖ?
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
<div class="proof-block">
  <span class="proof-label">Derivação Passo a Passo</span>
  <p><strong>Avalie em x = a:</strong> P(a) = c₀ = f(a) → <em>c₀ = f(a)</em></p>
  <p><strong>Derive e avalie em a:</strong> P'(a) = c₁ = f'(a) → <em>c₁ = f'(a)</em></p>
  <p><strong>Derive duas vezes:</strong> P''(a) = 2c₂ = f''(a) → <em>c₂ = f''(a)/2!</em></p>
  <p><strong>k-ésima derivada:</strong> P⁽ᵏ⁾(a) = k! · cₖ = f⁽ᵏ⁾(a) → <em>cₖ = f⁽ᵏ⁾(a)/k!</em></p>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown('<div style="padding:0.4rem 0">', unsafe_allow_html=True)
    st.latex(r"c_k = \frac{f^{(k)}(a)}{k!}")
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    st.latex(r"P_n(x) = \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k")
    st.markdown("""
<div class="highlight-box teal" style="margin-top:1rem">
  <span class="hb-label">Unicidade</span>
  <p>O polinômio que satisfaz essas n+1 condições de derivadas é <strong>único</strong>.
  Não existe outro polinômio de grau ≤ n com as mesmas derivadas em a.</p>
</div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 3 · VISUALIZADOR INTERATIVO ───────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 03 — Visualizador Interativo</span>
<div class="section-title">Convergência em tempo real</div>
<div class="section-sub">
  Observe como cada novo termo expande a zona de concordância entre o polinômio e a função original.
</div>
""", unsafe_allow_html=True)

cA, cB, cC, cD = st.columns([1.8, 1.2, 1.0, 1.0])
with cA:
    func_choice = st.selectbox("Função f(x)", list(CONVERGENCE_RADII.keys()), index=0)
with cB:
    center_a = st.slider("Ponto de expansão a", -2.0, 2.0, 0.0, 0.25)
with cC:
    max_terms = st.slider("Máx. de termos", 2, 15, 8)
with cD:
    show_error = st.checkbox("Mostrar erro |f − Pₙ|", value=True)

# Compute
XMIN_VIZ = {
    "eˣ": (-3.5, 3.5), "sin(x)": (-5, 5), "cos(x)": (-5, 5),
    "ln(1+x)": (-0.95, 2.5), "1/(1-x)": (-0.95, 0.95),
    "tan(x)": (-1.45, 1.45), "sinh(x)": (-3, 3),
    "cosh(x)": (-3, 3), "arctan(x)": (-3, 3), "√(1+x)": (-0.95, 2.5),
}
xlo, xhi = XMIN_VIZ[func_choice]
xarr = np.linspace(xlo, xhi, 1200)
ytrue = true_function(func_choice, xarr)

all_coeffs = taylor_coeffs(func_choice, max_terms, center_a)
latex_series, _ = get_sympy_series(func_choice, min(max_terms, 6), center_a)

# Main plot
if show_error:
    fig = make_subplots(rows=2, cols=1, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06,
                        subplot_titles=["Aproximação de Taylor", "Erro absoluto |f(x) − Pₙ(x)|"])
else:
    fig = make_subplots(rows=1, cols=1)

# True function
fig.add_trace(go.Scatter(
    x=xarr, y=ytrue, mode="lines", name=f"f(x) = {func_choice}",
    line=dict(color=GOLD, width=2.8),
    hovertemplate="x=%{x:.3f}<br>f(x)=%{y:.6f}<extra></extra>"), row=1, col=1)

# Taylor approximations
step_terms = max(1, max_terms // 5)
term_range = list(range(1, max_terms + 1, step_terms))
if max_terms not in term_range:
    term_range.append(max_terms)

for i, n_t in enumerate(term_range):
    coeffs_n = all_coeffs[:n_t]
    yp = eval_taylor(coeffs_n, center_a, xarr)
    yp_clipped = np.where(np.abs(yp) > 20, np.nan, yp)
    alpha = 0.35 + 0.65 * (i / max(len(term_range) - 1, 1))
    col_idx = i % len(COLORS[1:])
    c = COLORS[1:][col_idx]
    is_last = (n_t == max_terms)
    fig.add_trace(go.Scatter(
        x=xarr, y=yp_clipped, mode="lines",
        name=f"P_{n_t}(x)  [{n_t} {'termo' if n_t==1 else 'termos'}]",
        line=dict(color=c, width=2.5 if is_last else 1.2,
                  dash="solid" if is_last else "dot"),
        opacity=1.0 if is_last else alpha,
        hovertemplate=f"n={n_t}: %{{y:.5f}}<extra></extra>"), row=1, col=1)

# Center point marker
ya_true = float(true_function(func_choice, np.array([center_a]))[0])
fig.add_trace(go.Scatter(
    x=[center_a], y=[ya_true], mode="markers",
    marker=dict(color=GOLD, size=10, symbol="circle",
                line=dict(color="white", width=1.5)),
    name=f"a = {center_a:.2f}", showlegend=True), row=1, col=1)

# Error plot
if show_error:
    yp_best = eval_taylor(all_coeffs, center_a, xarr)
    err = np.abs(ytrue - yp_best)
    err = np.where(err > 10, np.nan, err)
    fig.add_trace(go.Scatter(
        x=xarr, y=err, mode="lines", name="Erro",
        fill="tozeroy",
        fillcolor="rgba(232,82,74,0.08)",
        line=dict(color=RED, width=1.8),
        hovertemplate="erro=%{y:.2e}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1,
                     gridcolor="rgba(255,255,255,0.04)")

fig.update_layout(
    **PLOTLY_BASE,
    height=580 if show_error else 420,
    legend=dict(orientation="v", x=1.01, y=1,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10)),
    title=dict(text=f"Séries de Taylor — {func_choice}  ·  a = {center_a}",
               font=dict(family="Playfair Display, serif", size=14, color="#c8d2e6")),
)
fig.update_xaxes(row=1, col=1, range=[xlo, xhi])
if show_error:
    fig.update_xaxes(row=2, col=1, range=[xlo, xhi])
fig.update_annotations(font=dict(family="DM Sans, sans-serif",
                                  color="rgba(200,212,235,0.50)", size=11))

st.plotly_chart(fig, use_container_width=True)

# Série em LaTeX
st.markdown('<div style="text-align:center;margin:0.5rem 0 1.2rem">', unsafe_allow_html=True)
st.latex(rf"P_{{{min(max_terms,6)}}}(x) \approx {latex_series}")
st.markdown('</div>', unsafe_allow_html=True)

# Raio de convergência
r, desc = CONVERGENCE_RADII[func_choice]
st.markdown(f"""
<div class="highlight-box {'gold' if r=='∞' else 'red'}">
  <span class="hb-label">Raio de Convergência R = {r}</span>
  <p>{desc}. {"A série converge absolutamente para todo x real — o denominador fatorial domina qualquer potência de x." if r=="∞" else "Fora deste intervalo a série diverge. Isso <strong>não</strong> significa que f(x) não está definida lá — significa apenas que a representação em série de potências em torno de a = 0 falha."}</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 4 · RESTO DE LAGRANGE ─────────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 04 — Análise de Erro</span>
<div class="section-title">O Resto de Lagrange e o controle do erro</div>
<div class="section-sub">
  A série nos diz o que a aproximação converge para. O resto nos diz <em>quão rápido</em> e <em>com que garantia</em>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="prose">
  Truncar a série em n termos deixa um resíduo <strong>Rₙ(x) = f(x) − Pₙ(x)</strong>.
  O Teorema do Resto de Lagrange dá uma cota explícita para esse erro em termos da (n+1)-ésima derivada.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 0.9])
with col1:
    st.markdown("""
<div class="proof-block">
  <span class="proof-label">Teorema — Resto de Lagrange</span>
  <p>Se f é (n+1) vezes diferenciável em (a, x), existe ξ ∈ (a, x) tal que:</p>
</div>""", unsafe_allow_html=True)
    st.latex(r"R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}")
    st.markdown("""
<div class="prose" style="margin-top:1rem">
  Como ξ é desconhecido, usa-se a <strong>cota do resto</strong>: se M é o máximo de |f⁽ⁿ⁺¹⁾|
  no intervalo entre a e x, então:
</div>""", unsafe_allow_html=True)
    st.latex(r"|R_n(x)| \leq \frac{M}{(n+1)!}|x-a|^{n+1}")

with col2:
    st.markdown("""
<div class="highlight-box gold">
  <span class="hb-label">💡 Intuição do fatorial</span>
  <p>O fatorial (n+1)! no denominador cresce <em>muito mais rápido</em> do que qualquer
  potência fixa |x−a|ⁿ⁺¹. Por isso, para funções como eˣ, sin e cos onde as derivadas
  são limitadas, o resto vai a zero para todo x fixo quando n→∞.</p>
</div>
<div class="highlight-box red" style="margin-top:1rem">
  <span class="hb-label">⚠ Quando o resto não vai a zero</span>
  <p>Se as derivadas f⁽ⁿ⁾ crescem sem controle (como em 1/(1−x) perto de x=1),
  M também cresce com n e o fatorial pode não vencer.
  A série diverge — mesmo que f esteja bem definida no ponto.</p>
</div>""", unsafe_allow_html=True)

# Visualizador de erro por grau
st.markdown("""
<div class="section-sub" style="margin-top:2rem">
  <strong style="color:#dde3f0">Visualize:</strong>
  como o erro máximo decresce com o grau, para um x fixo.
</div>""", unsafe_allow_html=True)

cx, cmax2 = st.columns([1, 1])
with cx:
    x_test = st.slider("Ponto de teste x", float(xlo + 0.1), float(xhi - 0.1),
                        min(1.0, float(xhi - 0.1)), 0.1)
with cmax2:
    n_max_err = st.slider("Grau máximo", 3, 20, 12)

err_data = []
for n_t in range(1, n_max_err + 1):
    c_ = taylor_coeffs(func_choice, n_t, center_a)
    yp_x = float(eval_taylor(c_, center_a, np.array([x_test]))[0])
    yt_x = float(true_function(func_choice, np.array([x_test]))[0])
    if np.isfinite(yp_x) and np.isfinite(yt_x):
        err_data.append({"n": n_t, "erro": abs(yt_x - yp_x)})

if err_data:
    df_err = pd.DataFrame(err_data)
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=df_err["n"], y=df_err["erro"], mode="lines+markers",
        name="Erro absoluto",
        line=dict(color=TEAL, width=2),
        marker=dict(size=6, color=TEAL,
                    line=dict(color="white", width=1)),
        hovertemplate="n=%{x}<br>|Rₙ|=%{y:.3e}<extra></extra>"))
    fig_err.update_layout(
        **PLOTLY_BASE, height=280,
        title=dict(
            text=f"|f({x_test:.2f}) − Pₙ({x_test:.2f})|  em função do grau n  ·  {func_choice}",
            font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
        yaxis_type="log",
        xaxis_title="Grau n", yaxis_title="Erro |Rₙ|",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    if len(err_data) >= 3:
        last_err = err_data[-1]["erro"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Erro em n=1", f"{err_data[0]['erro']:.4e}")
        m2.metric(f"Erro em n={n_max_err//2}", f"{err_data[n_max_err//2 - 1]['erro']:.4e}")
        m3.metric(f"Erro em n={n_max_err}", f"{last_err:.4e}",
                  delta=f"÷{err_data[0]['erro']/max(last_err,1e-300):.1e} vs n=1")

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 5 · RAIO DE CONVERGÊNCIA ─────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 05 — Convergência</span>
<div class="section-title">Raio de Convergência e o Teste da Razão</div>
<div class="section-sub">
  Uma série de potências sempre converge num disco. O raio desse disco é determinado pelo comportamento dos coeficientes.
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("""
<div class="prose">
  Toda série de potências <strong>∑ cₙ(x−a)ⁿ</strong> tem um <em>raio de convergência</em> R ∈ [0, ∞]
  tal que a série converge absolutamente para |x−a| < R e diverge para |x−a| > R.
  No bordo |x−a| = R, o comportamento deve ser analisado caso a caso.
</div>
<br>
<div class="proof-block">
  <span class="proof-label">Fórmula de Hadamard</span>
  <p>O raio de convergência é dado por:</p>
</div>""", unsafe_allow_html=True)
    st.latex(r"\frac{1}{R} = \limsup_{n\to\infty} |c_n|^{1/n}")
    st.markdown("""
<div class="proof-block" style="margin-top:1rem">
  <span class="proof-label">Teste da Razão (para Taylor)</span>
  <p>Para séries de Taylor com cₙ = f⁽ⁿ⁾(a)/n!, o teste da razão dá:</p>
</div>""", unsafe_allow_html=True)
    st.latex(r"R = \lim_{n\to\infty}\left|\frac{c_n}{c_{n+1}}\right| = \lim_{n\to\infty}\left|\frac{f^{(n)}(a)}{f^{(n+1)}(a)}\cdot(n+1)\right|")

with col2:
    st.markdown("""
<div class="card-grid card-grid-2" style="margin-top:0.3rem">
  <div class="card">
    <span class="card-icon">🟢</span>
    <div class="card-title">|x − a| < R</div>
    <p>Convergência absoluta. A série pode ser diferenciada e integrada termo a termo. Tudo funciona.</p>
  </div>
  <div class="card">
    <span class="card-icon">🔴</span>
    <div class="card-title">|x − a| > R</div>
    <p>Divergência. Os termos crescem sem limite. A soma não existe (no sentido usual).</p>
  </div>
  <div class="card">
    <span class="card-icon">🟡</span>
    <div class="card-title">|x − a| = R</div>
    <p>Bordo: pode convergir (arctan em ±1), divergir, ou oscilar. Precisa de análise individual.</p>
  </div>
  <div class="card">
    <span class="card-icon">♾️</span>
    <div class="card-title">R = ∞</div>
    <p>Série inteira. Converge em todo ℝ (e em todo ℂ). Funções como eˣ, sin, cos pertencem a esta classe.</p>
  </div>
</div>""", unsafe_allow_html=True)

# Mapa interativo de convergência
st.markdown("""
<div class="section-sub" style="margin-top:2rem">
  <strong style="color:#dde3f0">Mapa de convergência:</strong>
  intensidade do erro ponto a ponto, revelando visualmente o raio de convergência.
</div>""", unsafe_allow_html=True)

n_map = st.slider("Grau do polinômio no mapa", 2, 12, 5)
func_map = func_choice

a_vals = np.linspace(-2.0, 2.0, 60)
x_vals = np.linspace(xlo, xhi, 60)
Z = np.zeros((len(a_vals), len(x_vals)))

for i, ai in enumerate(a_vals):
    try:
        ci = taylor_coeffs(func_map, n_map, float(ai))
        yp = eval_taylor(ci, float(ai), x_vals)
        yt = true_function(func_map, x_vals)
        err_row = np.abs(yt - yp)
        err_row = np.where(np.isfinite(err_row) & (err_row < 50), err_row, np.nan)
        Z[i, :] = err_row
    except Exception:
        Z[i, :] = np.nan

fig_map = go.Figure(go.Heatmap(
    x=x_vals, y=a_vals, z=np.log10(np.maximum(Z, 1e-10)),
    colorscale=[[0, "#0c1a0f"], [0.3, "#3ecfb2"],
                [0.6, "#e8b84b"], [1.0, "#e8524a"]],
    colorbar=dict(title="log₁₀(erro)", tickfont=dict(size=10, color="#9aacc8"),
                  titlefont=dict(size=10, color="#9aacc8")),
    hovertemplate="x=%{x:.2f}<br>a=%{y:.2f}<br>log₁₀(erro)=%{z:.2f}<extra></extra>",
    zmin=-10, zmax=1,
))
fig_map.update_layout(
    **PLOTLY_BASE, height=360,
    title=dict(
        text=f"Erro log₁₀|f(x)−P_{n_map}(x)|  —  {func_map}  (verde=preciso, vermelho=impreciso)",
        font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
    xaxis_title="x  (ponto de avaliação)",
    yaxis_title="a  (centro da expansão)",
)
st.plotly_chart(fig_map, use_container_width=True)
st.markdown("""
<div class="highlight-box teal">
  <span class="hb-label">Como ler o mapa</span>
  <p>Cada célula mostra o erro de Pₙ em x, com expansão centrada em a. A faixa verde diagonal
  é a "zona de conforto" — onde |x−a| é pequeno. Conforme nos afastamos da diagonal, entramos
  fora do raio de convergência e o erro explode (vermelho).</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 6 · SÉRIES CLÁSSICAS ──────────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 06 — Catálogo</span>
<div class="section-title">As Séries Clássicas e suas Derivações</div>
<div class="section-sub">
  As expansões mais importantes da análise, com derivação explícita e padrão dos coeficientes.
</div>
""", unsafe_allow_html=True)

tab_e, tab_sin, tab_cos, tab_ln, tab_geo = st.tabs([
    "  eˣ  ", "  sin(x)  ", "  cos(x)  ", "  ln(1+x)  ", "  1/(1−x)  "
])

with tab_e:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
<span class="section-eyebrow">Série de Maclaurin de eˣ</span>
<div class="prose">
  <strong>A derivação mais elegante do cálculo.</strong> A função eˣ é a única que é igual
  à própria derivada: f'(x) = f(x). Isso significa que <em>todas</em> as derivadas em x=0
  valem 1: f⁽ⁿ⁾(0) = e⁰ = 1 para todo n.
</div>
<br>""", unsafe_allow_html=True)
        st.latex(r"e^x = \sum_{n=0}^{\infty}\frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots")
        st.markdown("""
<div class="highlight-box gold" style="margin-top:1rem">
  <span class="hb-label">Teste: calcule e = e¹</span>
  <p>Substituindo x=1: e = 1 + 1 + 1/2 + 1/6 + 1/24 + 1/120 + ⋯<br>
  Com 10 termos: e ≈ 2.71828182... ✓</p>
</div>
<div class="highlight-box teal" style="margin-top:0.8rem">
  <span class="hb-label">R = ∞ · Prova via teste da razão</span>
  <p>|cₙ₊₁/cₙ| = |x/(n+1)| → 0 para qualquer x fixo. O fatorial vence sempre.</p>
</div>""", unsafe_allow_html=True)
    with col2:
        ns_e = np.arange(0, 13)
        coeffs_e = 1.0 / np.array([float(sp.factorial(n)) for n in ns_e])
        fig_e = go.Figure()
        fig_e.add_trace(go.Bar(
            x=ns_e, y=coeffs_e,
            marker=dict(color=COLORS[:len(ns_e)], opacity=0.85,
                        line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
            name="1/n!",
            hovertemplate="n=%{x}<br>1/n! = %{y:.3e}<extra></extra>"))
        fig_e.update_layout(
            **PLOTLY_BASE, height=320,
            title=dict(text="Coeficientes da série de eˣ: aₙ = 1/n!",
                       font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
            xaxis_title="n", yaxis_title="1/n!", yaxis_type="log",
            showlegend=False)
        st.plotly_chart(fig_e, use_container_width=True)

with tab_sin:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
<span class="section-eyebrow">Série de Maclaurin de sin(x)</span>
<div class="prose">
  As derivadas de sin(x) ciclam com período 4: sin→cos→−sin→−cos→sin→⋯
  Avaliadas em x=0, os valores são 0, 1, 0, −1, 0, 1, … — ou seja,
  apenas os termos <strong>ímpares sobrevivem</strong>, com sinais alternados.
</div>
<br>""", unsafe_allow_html=True)
        st.latex(r"\sin(x) = \sum_{n=0}^{\infty}\frac{(-1)^n}{(2n+1)!}x^{2n+1}")
        st.latex(r"= x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots")
        st.markdown("""
<div class="highlight-box gold" style="margin-top:1rem">
  <span class="hb-label">Consequência notável</span>
  <p>sin(x)/x = 1 − x²/3! + x⁴/5! − ⋯ → 1 quando x→0.<br>
  Isso prova o limite fundamental sem L'Hôpital.</p>
</div>""", unsafe_allow_html=True)
    with col2:
        ns_sin = np.arange(0, 8)
        vals_sin = np.array([(-1)**n / float(sp.factorial(2*n+1)) for n in ns_sin])
        fig_sin = go.Figure()
        fig_sin.add_trace(go.Bar(
            x=2*ns_sin+1, y=vals_sin,
            marker=dict(
                color=[TEAL if v > 0 else RED for v in vals_sin],
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
            hovertemplate="(2n+1)=%{x}<br>coef=%{y:.3e}<extra></extra>"))
        fig_sin.update_layout(
            **PLOTLY_BASE, height=320,
            title=dict(text="Coeficientes de sin(x): (−1)ⁿ/(2n+1)!",
                       font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
            xaxis_title="grau (ímpar)", yaxis_title="coeficiente",
            showlegend=False)
        st.plotly_chart(fig_sin, use_container_width=True)

with tab_cos:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
<span class="section-eyebrow">Série de Maclaurin de cos(x)</span>
<div class="prose">
  Mesma lógica do seno: o ciclo de derivadas é idêntico, mas <strong>os termos pares sobrevivem</strong>
  (porque cos(0) = 1 e −cos(0) = −1, enquanto sin(0) = 0 zera os termos ímpares).
  Além disso, a série do cosseno é a <em>derivada termo a termo</em> da série do seno.
</div>
<br>""", unsafe_allow_html=True)
        st.latex(r"\cos(x) = \sum_{n=0}^{\infty}\frac{(-1)^n}{(2n)!}x^{2n}")
        st.latex(r"= 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots")
        st.markdown("""
<div class="highlight-box purple" style="margin-top:1rem">
  <span class="hb-label">Fórmula de Euler (derivada desta série)</span>
  <p>Combinando as séries de eⁱˣ, cos e sin: <strong>eⁱˣ = cos(x) + i·sin(x)</strong>.<br>
  Em x=π: e^(iπ) + 1 = 0 — a identidade de Euler.</p>
</div>""", unsafe_allow_html=True)
    with col2:
        ns_cos = np.arange(0, 8)
        vals_cos = np.array([(-1)**n / float(sp.factorial(2*n)) for n in ns_cos])
        fig_cos = go.Figure()
        fig_cos.add_trace(go.Bar(
            x=2*ns_cos, y=vals_cos,
            marker=dict(
                color=[BLUE if v > 0 else PURPLE for v in vals_cos],
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
            hovertemplate="(2n)=%{x}<br>coef=%{y:.3e}<extra></extra>"))
        fig_cos.update_layout(
            **PLOTLY_BASE, height=320,
            title=dict(text="Coeficientes de cos(x): (−1)ⁿ/(2n)!",
                       font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
            xaxis_title="grau (par)", yaxis_title="coeficiente",
            showlegend=False)
        st.plotly_chart(fig_cos, use_container_width=True)

with tab_ln:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
<span class="section-eyebrow">Série de Maclaurin de ln(1+x)</span>
<div class="prose">
  Esta série é derivada integrando a série geométrica 1/(1+x) = 1 − x + x² − ⋯
  termo a termo (operação válida dentro do raio de convergência).
  A série <strong>converge apenas para |x| ≤ 1, x ≠ −1</strong>.
</div>
<br>""", unsafe_allow_html=True)
        st.latex(r"\ln(1+x) = \sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{n}x^n")
        st.latex(r"= x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots")
        st.markdown("""
<div class="highlight-box red" style="margin-top:1rem">
  <span class="hb-label">⚠ Convergência lenta</span>
  <p>Para x próximo de ±1, a série converge muito lentamente (harmônica).
  Na prática, usa-se ln(x) = 2·arctanh((x−1)/(x+1)) para convergência mais rápida.</p>
</div>
<div class="highlight-box teal" style="margin-top:0.8rem">
  <span class="hb-label">Resultado bonito: ln(2)</span>
  <p>Em x=1: ln(2) = 1 − 1/2 + 1/3 − 1/4 + ⋯ (série de Leibniz-Mercator, R=1).</p>
</div>""", unsafe_allow_html=True)
    with col2:
        ns_ln = np.arange(1, 14)
        vals_ln = np.array([(-1)**(n+1) / float(n) for n in ns_ln])
        fig_ln = go.Figure()
        fig_ln.add_trace(go.Bar(
            x=ns_ln, y=vals_ln,
            marker=dict(
                color=[GOLD if v > 0 else RED for v in vals_ln],
                opacity=0.85,
                line=dict(color="rgba(255,255,255,0.2)", width=0.5)),
            hovertemplate="n=%{x}<br>coef=%{y:.4f}<extra></extra>"))
        fig_ln.update_layout(
            **PLOTLY_BASE, height=320,
            title=dict(text="Coeficientes de ln(1+x): (−1)ⁿ⁺¹/n",
                       font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
            xaxis_title="n", yaxis_title="coeficiente",
            showlegend=False)
        st.plotly_chart(fig_ln, use_container_width=True)

with tab_geo:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
<span class="section-eyebrow">Série Geométrica — 1/(1−x)</span>
<div class="prose">
  A série geométrica é o protótipo de série de potências — e a mãe de quase todas as outras.
  A derivação é direta: se S = 1 + x + x² + ⋯ então xS = x + x² + ⋯,
  logo S − xS = 1, e <strong>S = 1/(1−x)</strong> para |x| < 1.
</div>
<br>""", unsafe_allow_html=True)
        st.latex(r"\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots \quad |x|<1")
        st.markdown("""
<div class="highlight-box gold" style="margin-top:1rem">
  <span class="hb-label">Geradora de outras séries</span>
  <p>Integrar → ln(1/(1−x)). Substituir x→−x² → arctan(x). Derivar → 1/(1−x)².
  É a série mais produtiva do arsenal.</p>
</div>
<div class="highlight-box blue" style="margin-top:0.8rem">
  <span class="hb-label">Soma finita (telescopagem)</span>
  <p>1 + x + ⋯ + xⁿ = (1 − xⁿ⁺¹)/(1 − x). No limite n→∞ com |x|<1: xⁿ⁺¹ → 0.</p>
</div>""", unsafe_allow_html=True)
    with col2:
        x_geo = np.linspace(-0.95, 0.95, 400)
        y_true_geo = 1.0 / (1.0 - x_geo)
        fig_geo = go.Figure()
        fig_geo.add_trace(go.Scatter(x=x_geo, y=y_true_geo, mode="lines",
                                      name="1/(1−x)", line=dict(color=GOLD, width=2.5)))
        for n_t in [2, 4, 6, 10]:
            yp_geo = sum(x_geo**k for k in range(n_t + 1))
            yp_geo = np.where(np.abs(yp_geo) > 15, np.nan, yp_geo)
            fig_geo.add_trace(go.Scatter(
                x=x_geo, y=yp_geo, mode="lines",
                name=f"P_{n_t}(x)",
                line=dict(width=1.4, dash="dot"),
                opacity=0.75,
                hovertemplate=f"n={n_t}: %{{y:.4f}}<extra></extra>"))
        fig_geo.update_layout(
            **PLOTLY_BASE, height=320,
            title=dict(text="1/(1−x) e suas aproximações polinomiais",
                       font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
            yaxis=dict(range=[-2, 12], showgrid=True,
                       gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(font=dict(size=10)))
        fig_geo.add_vline(x=1.0, line_dash="dot",
                           line_color="rgba(232,82,74,0.5)",
                           annotation_text="R=1", annotation_font_size=10)
        fig_geo.add_vline(x=-1.0, line_dash="dot",
                           line_color="rgba(232,82,74,0.5)")
        st.plotly_chart(fig_geo, use_container_width=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 7 · APLICAÇÕES ────────────────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 07 — Mundo Real</span>
<div class="section-title">Onde as Séries de Taylor vivem na prática</div>
<div class="section-sub">
  Não é abstração acadêmica: séries de Taylor estão na base de toda computação numérica moderna.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-grid">
  <div class="app-card">
    <span class="app-icon">💻</span>
    <h4>Hardware de Ponto Flutuante</h4>
    <p>Processadores calculam sin, cos, eˣ e ln via aproximações de Taylor/Chebyshev
    implementadas em microcódigo. A precisão do IEEE 754 double é 15–16 dígitos —
    obtida com poucos termos da série para x no intervalo de redução [0, π/4].</p>
  </div>
  <div class="app-card">
    <span class="app-icon">🤖</span>
    <h4>Machine Learning & Otimização</h4>
    <p>Métodos de segunda ordem (Newton, L-BFGS) aproximam a função de custo
    localmente por um polinômio de Taylor de grau 2 (Hessiana). A expansão
    f(θ + δ) ≈ f(θ) + ∇f·δ + δᵀHδ/2 define o passo ótimo de atualização.</p>
  </div>
  <div class="app-card">
    <span class="app-icon">⚛️</span>
    <h4>Física Quântica & Perturbação</h4>
    <p>Teoria de perturbação expande o Hamiltoniano em série de Taylor no parâmetro λ:
    H = H₀ + λH₁ + λ²H₂ + ⋯ Cada ordem corrige a energia e a função de onda
    do sistema não-perturbado.</p>
  </div>
  <div class="app-card">
    <span class="app-icon">📡</span>
    <h4>Processamento de Sinais (DSP)</h4>
    <p>Filtros IIR e FIR são construídos aproximando funções de transferência por
    polinômios. A análise de estabilidade via expansão z⁻¹ é essencialmente
    uma série de Laurent (Taylor generalizada).</p>
  </div>
  <div class="app-card">
    <span class="app-icon">🛸</span>
    <h4>Relatividade & Física Clássica</h4>
    <p>O fator de Lorentz γ = 1/√(1−v²/c²) ≈ 1 + v²/2c² + 3v⁴/8c⁴ + ⋯
    Para v ≪ c, os termos de ordem superior são desprezíveis — recuperamos
    a física Newtoniana como aproximação de Taylor de ordem 1.</p>
  </div>
  <div class="app-card">
    <span class="app-icon">📊</span>
    <h4>Finanças — Modelo Black-Scholes</h4>
    <p>Expansão de Taylor do preço de opções (gregas: Δ, Γ, Θ) permite
    aproximar o P&L de um portfólio sem recalcular o modelo completo.
    A expansão de segunda ordem dá a correção de convexidade (gamma).</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Demonstração: Lorentz
st.markdown("""
<div class="section-sub" style="margin-top:2rem">
  <strong style="color:#dde3f0">Demonstração interativa:</strong>
  fator de Lorentz γ(β) e sua expansão de Taylor, onde β = v/c.
</div>""", unsafe_allow_html=True)

beta = np.linspace(0, 0.95, 600)
gamma_exact = 1.0 / np.sqrt(1 - beta**2)

fig_lorentz = go.Figure()
fig_lorentz.add_trace(go.Scatter(
    x=beta, y=gamma_exact, mode="lines", name="γ exato = 1/√(1−β²)",
    line=dict(color=GOLD, width=2.8),
    hovertemplate="β=%{x:.3f}<br>γ=%{y:.4f}<extra></extra>"))

# Taylor approx
taylor_terms = {
    "Ordem 1 (Newton)": lambda b: 1 + b**2 / 2,
    "Ordem 2":          lambda b: 1 + b**2/2 + 3*b**4/8,
    "Ordem 3":          lambda b: 1 + b**2/2 + 3*b**4/8 + 5*b**6/16,
}
cols_l = [TEAL, BLUE, PURPLE]
for (label, fn), col in zip(taylor_terms.items(), cols_l):
    y_approx = fn(beta)
    y_approx = np.where(np.abs(y_approx) > 20, np.nan, y_approx)
    fig_lorentz.add_trace(go.Scatter(
        x=beta, y=y_approx, mode="lines", name=label,
        line=dict(color=col, width=1.6, dash="dot"),
        opacity=0.85))

fig_lorentz.update_layout(
    **PLOTLY_BASE, height=340,
    title=dict(text="Fator de Lorentz γ(β): exato vs expansão de Taylor em β = 0",
               font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
    xaxis_title="β = v/c", yaxis_title="γ",
    legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
    yaxis=dict(range=[0.9, 8], showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
)
st.plotly_chart(fig_lorentz, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
<div class="highlight-box teal">
  <span class="hb-label">Expansão de Taylor do fator de Lorentz</span>
</div>""", unsafe_allow_html=True)
    st.latex(r"\gamma = \frac{1}{\sqrt{1-\beta^2}} = 1 + \frac{\beta^2}{2} + \frac{3\beta^4}{8} + \frac{5\beta^6}{16} + \cdots")
with col2:
    st.markdown("""
<div class="highlight-box gold">
  <span class="hb-label">Energia cinética relativística</span>
</div>""", unsafe_allow_html=True)
    st.latex(r"K = (\gamma - 1)mc^2 \approx \frac{1}{2}mv^2 + \frac{3mv^4}{8c^2} + \cdots")

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 8 · TABELA DE REFERÊNCIA ──────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 08 — Referência</span>
<div class="section-title">Tabela Completa de Séries de Maclaurin</div>
""", unsafe_allow_html=True)

st.markdown("""
<table class="styled-table">
<thead>
  <tr><th>Função</th><th>Série de Maclaurin</th><th>Raio R</th><th>Padrão dos coef.</th></tr>
</thead>
<tbody>
  <tr><td>eˣ</td><td>∑ xⁿ/n!</td><td>∞</td><td>1/n!</td></tr>
  <tr><td>sin(x)</td><td>∑ (−1)ⁿ x²ⁿ⁺¹/(2n+1)!</td><td>∞</td><td>Ímpares alternados</td></tr>
  <tr><td>cos(x)</td><td>∑ (−1)ⁿ x²ⁿ/(2n)!</td><td>∞</td><td>Pares alternados</td></tr>
  <tr><td>sinh(x)</td><td>∑ x²ⁿ⁺¹/(2n+1)!</td><td>∞</td><td>Ímpares positivos</td></tr>
  <tr><td>cosh(x)</td><td>∑ x²ⁿ/(2n)!</td><td>∞</td><td>Pares positivos</td></tr>
  <tr><td>ln(1+x)</td><td>∑ (−1)ⁿ⁺¹ xⁿ/n</td><td>1</td><td>1/n alternado</td></tr>
  <tr><td>1/(1−x)</td><td>∑ xⁿ</td><td>1</td><td>Todos 1</td></tr>
  <tr><td>arctan(x)</td><td>∑ (−1)ⁿ x²ⁿ⁺¹/(2n+1)</td><td>1</td><td>1/(2n+1) alternado</td></tr>
  <tr><td>arcsin(x)</td><td>∑ (2n)! x²ⁿ⁺¹ / (4ⁿ(n!)² (2n+1))</td><td>1</td><td>Coef. binomiais</td></tr>
  <tr><td>(1+x)ᵅ</td><td>∑ C(α,n) xⁿ</td><td>1</td><td>Coef. binomiais generalizados</td></tr>
  <tr><td>eˣ·cos(x)</td><td>1 + x − x³/3 − x⁴/6 − ⋯</td><td>∞</td><td>Produto de séries</td></tr>
</tbody>
</table>
""", unsafe_allow_html=True)

st.markdown("<hr class='sec-divider'>", unsafe_allow_html=True)


# ─── SEÇÃO 9 · IDENTIDADE DE EULER ───────────────────────────────────────────
st.markdown("""
<span class="section-eyebrow">§ 09 — Coda</span>
<div class="section-title">A Identidade de Euler: séries de Taylor no plano complexo</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 0.9])
with col1:
    st.markdown("""
<div class="prose">
  As séries de Taylor não se restringem à reta real — elas se estendem naturalmente ao
  plano complexo ℂ, onde revelam conexões profundas entre funções aparentemente não relacionadas.
  <br><br>
  Substituindo <strong>x = iθ</strong> na série de eˣ e separando as partes real e imaginária:
</div>
<br>""", unsafe_allow_html=True)
    st.latex(r"e^{i\theta} = \sum_{n=0}^{\infty}\frac{(i\theta)^n}{n!}")
    st.latex(r"= \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)}_{\cos\theta} + i\underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)}_{\sin\theta}")
    st.latex(r"\boxed{e^{i\theta} = \cos\theta + i\sin\theta}")
    st.markdown("""
<div class="highlight-box purple" style="margin-top:1rem">
  <span class="hb-label">Em θ = π</span>
  <p><strong>e^(iπ) = cos(π) + i·sin(π) = −1 + 0·i = −1</strong><br>
  Portanto: e^(iπ) + 1 = 0.<br>
  Cinco das constantes mais importantes da matemática — e, i, π, 1, 0 —
  reunidas numa única equação, derivada diretamente das séries de Taylor.</p>
</div>""", unsafe_allow_html=True)

with col2:
    theta = np.linspace(0, 2 * np.pi, 600)
    x_circ = np.cos(theta)
    y_circ = np.sin(theta)
    theta_mark = np.pi / 3
    fig_euler = go.Figure()
    fig_euler.add_trace(go.Scatter(
        x=x_circ, y=y_circ, mode="lines", name="Círculo unitário",
        line=dict(color="rgba(200,212,235,0.20)", width=1.5)))
    fig_euler.add_shape(type="line", x0=0, y0=0,
                         x1=np.cos(theta_mark), y1=np.sin(theta_mark),
                         line=dict(color=GOLD, width=2))
    fig_euler.add_trace(go.Scatter(
        x=[np.cos(theta_mark)], y=[np.sin(theta_mark)],
        mode="markers+text",
        marker=dict(color=GOLD, size=12, line=dict(color="white", width=1.5)),
        text=["e^(iθ)"], textposition="top right",
        textfont=dict(color=GOLD, size=13), name="e^(iθ)"))
    theta_arc = np.linspace(0, theta_mark, 80)
    fig_euler.add_trace(go.Scatter(
        x=0.22 * np.cos(theta_arc), y=0.22 * np.sin(theta_arc),
        mode="lines", line=dict(color=TEAL, width=1.5), showlegend=False))
    fig_euler.add_annotation(x=0.28, y=0.10, text="θ",
                               showarrow=False, font=dict(color=TEAL, size=14))
    fig_euler.add_shape(type="line", x0=0, y0=0,
                         x1=np.cos(theta_mark), y1=0,
                         line=dict(color=BLUE, width=1.2, dash="dot"))
    fig_euler.add_shape(type="line",
                         x0=np.cos(theta_mark), y0=0,
                         x1=np.cos(theta_mark), y1=np.sin(theta_mark),
                         line=dict(color=RED, width=1.2, dash="dot"))
    fig_euler.add_annotation(x=np.cos(theta_mark)/2, y=-0.10,
                               text="cos θ", showarrow=False,
                               font=dict(color=BLUE, size=12))
    fig_euler.add_annotation(x=np.cos(theta_mark) + 0.14,
                               y=np.sin(theta_mark)/2,
                               text="sin θ", showarrow=False,
                               font=dict(color=RED, size=12))
    fig_euler.update_layout(
        **PLOTLY_BASE, height=360,
        title=dict(text="Fórmula de Euler no plano complexo",
                   font=dict(family="Playfair Display, serif", size=13, color="#c8d2e6")),
        xaxis=dict(range=[-1.3, 1.6], showgrid=True,
                   gridcolor="rgba(255,255,255,0.04)", scaleanchor="y"),
        yaxis=dict(range=[-1.3, 1.3], showgrid=True,
                   gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False,
    )
    fig_euler.add_trace(go.Scatter(
        x=[-1], y=[0], mode="markers+text",
        marker=dict(color=PURPLE, size=10),
        text=["e^(iπ) = −1"], textposition="bottom right",
        textfont=dict(color=PURPLE, size=11), showlegend=False))
    st.plotly_chart(fig_euler, use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
  Fellipe Almässy &nbsp;·&nbsp; Séries de Taylor — Análise Matemática &nbsp;·&nbsp;
  SymPy &nbsp;+&nbsp; NumPy &nbsp;+&nbsp; Plotly &nbsp;+&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # /page-wrap
