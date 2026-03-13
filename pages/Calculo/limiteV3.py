from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# ─────────────────────────────────────────────────────────────
# 0) PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="TEC · Limites", layout="wide", page_icon="∂")

# ─────────────────────────────────────────────────────────────
# 1) CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;1,300&family=IBM+Plex+Sans:wght@300;400;500&family=Cormorant+Garamond:ital,wght@1,300;1,400&display=swap');

:root {
  --bg:        #060810;
  --surface:   #0c0f1a;
  --panel:     #0f1220;
  --rim:       rgba(255,255,255,0.06);
  --rim2:      rgba(255,255,255,0.10);
  --orange:     #ff6b2b;
  --orange-dim: rgba(255,107,43,0.12);
  --orange-glow:rgba(255,107,43,0.04);
  --green:     #00d4aa;
  --green-dim: rgba(0,212,170,0.10);
  --red:       #ff4d4d;
  --red-dim:   rgba(255,77,77,0.10);
  --blue:      #4da6ff;
  --text:      rgba(220,225,235,0.92);
  --muted:     rgba(220,225,235,0.45);
  --muted2:    rgba(220,225,235,0.20);
  --mono:      'IBM Plex Mono', monospace;
  --sans:      'IBM Plex Sans', sans-serif;
  --serif:     'Cormorant Garamond', Georgia, serif;
}

html, body, .stApp { background-color: var(--bg) !important; font-family: var(--sans); }

.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px);
  pointer-events: none;
  z-index: 0;
}

section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--orange-dim);
}
section[data-testid="stSidebar"] * { font-family: var(--mono) !important; font-size: 0.82rem !important; }

input, .stTextInput input, .stNumberInput input {
  background: rgba(255,107,43,0.03) !important;
  border: 1px solid rgba(255,107,43,0.20) !important;
  border-radius: 2px !important;
  color: var(--orange) !important;
  font-family: var(--mono) !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.04em !important;
}
input:focus {
  border-color: var(--orange) !important;
  box-shadow: 0 0 12px var(--orange-dim), 0 0 0 1px var(--orange) !important;
  outline: none !important;
}
div[data-baseweb="select"] > div {
  background: rgba(255,107,43,0.03) !important;
  border: 1px solid rgba(255,107,43,0.20) !important;
  border-radius: 2px !important;
  color: var(--orange) !important;
  font-family: var(--mono) !important;
}
div[role="radiogroup"] label {
  font-family: var(--mono) !important;
  font-size: 0.80rem !important;
  letter-spacing: 0.06em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}

/* ── Cabeçalho ─────────────────────────────────── */
.bbg-header {
  position: relative;
  padding: 2rem 0 1.5rem;
  border-bottom: 1px solid var(--orange-dim);
  margin-bottom: 1.5rem;
  overflow: hidden;
}
.bbg-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0; width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--orange), var(--green), transparent);
  animation: scan-h 3s ease-in-out infinite;
}
@keyframes scan-h {
  0%   { transform: scaleX(0); transform-origin: left; opacity: 0; }
  20%  { transform: scaleX(1); transform-origin: left; opacity: 1; }
  80%  { transform: scaleX(1); transform-origin: right; opacity: 1; }
  100% { transform: scaleX(0); transform-origin: right; opacity: 0; }
}
.bbg-ticker { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.18em; color: var(--orange); text-transform: uppercase; opacity: 0.7; margin-bottom: 0.5rem; }
.bbg-title  { font-family: var(--mono); font-size: 2rem; font-weight: 500; color: var(--text); letter-spacing: -0.01em; line-height: 1; }
.bbg-title span { color: var(--orange); }
.bbg-sub    { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); letter-spacing: 0.12em; margin-top: 0.5rem; }

/* ── Ticker tape ───────────────────────────────── */
.ticker-wrap {
  overflow: hidden;
  background: rgba(255,107,43,0.04);
  border-top: 1px solid var(--orange-dim);
  border-bottom: 1px solid var(--orange-dim);
  padding: 5px 0; margin-bottom: 1.5rem;
}
.ticker-content {
  display: inline-flex; white-space: nowrap;
  animation: ticker 20s linear infinite;
  font-family: var(--mono); font-size: 0.72rem; color: var(--muted); letter-spacing: 0.08em;
}
.ticker-content .pos { color: var(--green); }
.ticker-content .neg { color: var(--red); }
.ticker-content .sep { color: var(--orange); margin: 0 1.5rem; }
@keyframes ticker { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

/* ── Métricas ─────────────────────────────────── */
@keyframes metric-in   { 0% { opacity:0; transform:translateY(10px); } 100% { opacity:1; transform:translateY(0); } }
@keyframes value-flash { 0% { color:var(--bg); } 30% { color:var(--orange); text-shadow:0 0 20px var(--orange); } 100% { color:var(--orange); text-shadow:none; } }
@keyframes border-pulse { 0%,100% { border-color:rgba(255,107,43,0.25); } 50% { border-color:var(--orange); box-shadow:0 0 15px var(--orange-dim); } }
@keyframes shimmer { 0% { left:-60%; } 100% { left:120%; } }

div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid rgba(255,107,43,0.25) !important;
  border-top: 2px solid var(--orange) !important;
  border-radius: 2px !important;
  padding: 14px 16px 12px !important;
  animation: metric-in 0.5s ease forwards, border-pulse 2.5s ease 0.5s 2;
  position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::after {
  content: ''; position: absolute; top:0; left:-100%; width:60%; height:100%;
  background: linear-gradient(90deg, transparent, rgba(255,107,43,0.06), transparent);
  animation: shimmer 2.5s ease 0.3s 1;
}
div[data-testid="stMetricLabel"] p { font-family:var(--mono) !important; font-size:0.66rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; color:var(--muted) !important; }
div[data-testid="stMetricValue"]   { font-family:var(--mono) !important; font-size:1.55rem !important; font-weight:400 !important; color:var(--orange) !important; letter-spacing:-0.02em !important; animation:value-flash 0.8s ease forwards !important; }
div[data-testid="stMetricDelta"]   { font-family:var(--mono) !important; font-size:0.72rem !important; }

/* ── Função ────────────────────────────────────── */
@keyframes fn-in { 0% { opacity:0; transform:translateX(-8px); } 100% { opacity:1; transform:translateX(0); } }
.fn-display {
  background: rgba(255,107,43,0.03);
  border: 1px solid rgba(255,107,43,0.15);
  border-left: 3px solid var(--orange);
  padding: 1.2rem 1.8rem; margin: 1.2rem 0;
  animation: fn-in 0.4s ease forwards;
}
.fn-label { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.16em; text-transform:uppercase; color:var(--orange); opacity:0.7; margin-bottom:0.4rem; }

/* ── Resultado ─────────────────────────────────── */
@keyframes result-in { 0% { opacity:0; transform:translateY(-6px); } 100% { opacity:1; transform:translateY(0); } }
@keyframes glow-green { 0%,100% { box-shadow:0 0 0 rgba(0,212,170,0); } 50% { box-shadow:0 0 25px rgba(0,212,170,0.15); } }
@keyframes glow-red   { 0%,100% { box-shadow:0 0 0 rgba(255,77,77,0); }  50% { box-shadow:0 0 25px rgba(255,77,77,0.15); } }

.result-exists {
  background: linear-gradient(135deg, rgba(0,212,170,0.06), rgba(0,212,170,0.02));
  border: 1px solid rgba(0,212,170,0.30); border-left: 3px solid var(--green);
  border-radius: 2px; padding: 1.2rem 1.8rem; margin: 1rem 0;
  animation: result-in 0.4s ease forwards, glow-green 2s ease 0.4s 3;
}
.result-not-exists {
  background: linear-gradient(135deg, rgba(255,77,77,0.06), rgba(255,77,77,0.02));
  border: 1px solid rgba(255,77,77,0.30); border-left: 3px solid var(--red);
  border-radius: 2px; padding: 1.2rem 1.8rem; margin: 1rem 0;
  animation: result-in 0.4s ease forwards, glow-red 2s ease 0.4s 3;
}
.result-label { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.16em; text-transform:uppercase; margin-bottom:0.5rem; }
.result-exists .result-label     { color: var(--green); }
.result-not-exists .result-label { color: var(--red); }

/* ── Teoria ────────────────────────────────────── */
.teoria-section {
  background: var(--surface);
  border: 1px solid var(--rim2);
  border-top: 2px solid var(--orange);
  border-radius: 2px;
  padding: 2rem 2.5rem;
  margin-top: 2rem;
}
.teoria-titulo {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--orange);
  margin-bottom: 1.5rem;
  opacity: 0.8;
}
.teoria-bloco {
  border-left: 2px solid rgba(255,107,43,0.25);
  padding: 0.8rem 0 0.8rem 1.5rem;
  margin-bottom: 1.8rem;
}
.teoria-bloco-titulo {
  font-family: var(--mono);
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--orange);
  margin-bottom: 0.6rem;
}
.teoria-texto {
  font-family: var(--sans);
  font-size: 0.90rem;
  color: var(--muted);
  line-height: 1.8;
}
.teoria-texto strong { color: var(--text); }
.teoria-texto code {
  font-family: var(--mono);
  font-size: 0.82rem;
  color: var(--orange);
  background: rgba(255,107,43,0.08);
  padding: 1px 5px;
  border-radius: 2px;
}

/* ── Misc ──────────────────────────────────────── */
.hr-orange { border:none; height:1px; background:linear-gradient(90deg,transparent,var(--orange),transparent); opacity:0.2; margin:1.8rem 0; }
.hr-thin   { border:none; border-top:1px solid var(--rim); margin:1rem 0; }
.piece-label { font-family:var(--mono); font-size:0.65rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--muted2); margin-bottom:6px; }
.stButton button { background:transparent !important; border:1px solid rgba(255,107,43,0.35) !important; color:var(--orange) !important; font-family:var(--mono) !important; font-size:0.72rem !important; letter-spacing:0.10em !important; text-transform:uppercase !important; border-radius:2px !important; transition:all 0.2s !important; }
.stButton button:hover { background:var(--orange-dim) !important; border-color:var(--orange) !important; box-shadow:0 0 15px var(--orange-dim) !important; }
details { border:1px solid var(--rim) !important; border-radius:2px !important; background:var(--surface) !important; }
details summary { font-family:var(--mono) !important; font-size:0.75rem !important; color:var(--muted) !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
.footer { font-family:var(--mono); font-size:0.68rem; letter-spacing:0.12em; color:var(--muted2); text-align:center; padding:2.5rem 0 1rem; border-top:1px solid var(--rim); margin-top:3rem; text-transform:uppercase; }
.footer span { color: var(--orange); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 2) ENGINE
# ─────────────────────────────────────────────────────────────
x_sym = sp.Symbol("x", real=True)
LOCALS = {
    "x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi, "E": sp.E,
}

@st.cache_resource(show_spinner=False)
def parse_simples(expr_str: str):
    expr = sp.sympify(expr_str, locals=LOCALS)
    return sp.lambdify(x_sym, expr, modules=["numpy"]), expr

@st.cache_resource(show_spinner=False)
def parse_piecewise(key: str, exprs: tuple, conds: tuple):
    partes = []
    for e, c in zip(exprs, conds):
        es = sp.sympify(e, locals=LOCALS)
        cs = True if c.lower() in ("otherwise","else","senão","") else sp.sympify(c, locals=LOCALS)
        partes.append((es, cs))
    expr_pw = sp.Piecewise(*partes)
    return sp.lambdify(x_sym, expr_pw, modules=["numpy"]), expr_pw

def calcular_limite(f, ponto, delta, tolerancia):
    le = float(f(ponto - delta))
    ld = float(f(ponto + delta))
    existe = abs(le - ld) <= tolerancia
    return {"existe": existe, "lim_esq": le, "lim_dir": ld,
            "lim_final": (le + ld) / 2 if existe else None, "delta": delta}

def lim_simbolico(expr, ponto):
    try:    return sp.limit(expr, x_sym, ponto)
    except: return None

# ─────────────────────────────────────────────────────────────
# 3) GRÁFICO — curva se desenhando com animação
# ─────────────────────────────────────────────────────────────
def gerar_grafico(f, ponto, resultado):
    margem = max(2.5, abs(ponto) * 1.0) if ponto != 0 else 3.5
    N = 200
    xs_full = np.linspace(ponto - margem, ponto + margem, N)

    try:
        ys_full = np.array(f(xs_full), dtype=float)
    except Exception:
        ys_full = np.array([float(f(xi)) for xi in xs_full], dtype=float)

    if np.any(np.isfinite(ys_full)):
        cap = np.nanpercentile(np.abs(ys_full[np.isfinite(ys_full)]), 98) * 4
        ys_full = np.where(np.isfinite(ys_full) & (np.abs(ys_full) < cap), ys_full, np.nan)

    ORANGE = "#ff6b2b"
    GREEN  = "#00d4aa"
    RED    = "#ff4d4d"
    BLUE   = "#4da6ff"
    BG     = "#060810"
    GRID   = "rgba(255,255,255,0.04)"

    # ── Frames: curva cresce da esquerda pra direita ──────────
    frames = []
    n_frames = 50
    step = max(1, N // n_frames)
    for k in range(step, N + 1, step):
        frames.append(go.Frame(
            data=[
                go.Scatter(x=xs_full[:k], y=ys_full[:k], mode="lines",
                           line=dict(color="rgba(255,107,43,0.15)", width=12),
                           hoverinfo="skip", showlegend=False),
                go.Scatter(x=xs_full[:k], y=ys_full[:k], mode="lines",
                           name="f(x)", line=dict(color=ORANGE, width=2.5),
                           hovertemplate="x=%{x:.5f}<br>f(x)=%{y:.5f}<extra></extra>"),
            ],
            name=str(k)
        ))

    # Estado inicial: curva completa visível (sem animação ativa)
    # O usuário clica ▶ TRAÇAR para ver o desenho
    fig = go.Figure(
        data=[
            # glow
            go.Scatter(x=xs_full, y=ys_full, mode="lines",
                       line=dict(color="rgba(255,107,43,0.15)", width=12),
                       hoverinfo="skip", showlegend=False),
            # curva principal — já visível no carregamento
            go.Scatter(x=xs_full, y=ys_full, mode="lines",
                       name="f(x)", line=dict(color=ORANGE, width=2.5),
                       hovertemplate="x=%{x:.5f}<br>f(x)=%{y:.5f}<extra></extra>"),
            # linha vertical x₀
            go.Scatter(
                x=[ponto, ponto],
                y=[float(np.nanmin(ys_full)) if np.any(np.isfinite(ys_full)) else -5,
                   float(np.nanmax(ys_full)) if np.any(np.isfinite(ys_full)) else 5],
                mode="lines", name="x₀",
                line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1),
                hoverinfo="skip"),
        ],
        frames=frames,
    )

    d = resultado["delta"]

    # Limite esquerdo
    fig.add_trace(go.Scatter(
        x=[ponto - d], y=[resultado["lim_esq"]], mode="markers",
        name=f"esq  {resultado['lim_esq']:.5f}",
        marker=dict(color=BLUE, size=11, symbol="circle",
                    line=dict(color="white", width=1.2))))

    # Limite direito
    fig.add_trace(go.Scatter(
        x=[ponto + d], y=[resultado["lim_dir"]], mode="markers",
        name=f"dir  {resultado['lim_dir']:.5f}",
        marker=dict(color=RED if not resultado["existe"] else GREEN,
                    size=11, symbol="circle",
                    line=dict(color="white", width=1.2))))

    # Ponto do limite
    if resultado["existe"]:
        fig.add_trace(go.Scatter(
            x=[ponto], y=[resultado["lim_final"]], mode="markers",
            name=f"L ≈ {resultado['lim_final']:.5f}",
            marker=dict(color=GREEN, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="'IBM Plex Mono', monospace", color="rgba(220,225,235,0.7)", size=11),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(6,8,16,0.85)", bordercolor="rgba(255,107,43,0.20)",
            borderwidth=1, font=dict(size=10)),
        margin=dict(l=10, r=10, t=50, b=10),
        height=430,
        xaxis=dict(showgrid=True, gridcolor=GRID,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(family="'IBM Plex Mono', monospace", size=10),
                   title=dict(text="x", font=dict(color="rgba(255,107,43,0.6)", size=11))),
        yaxis=dict(showgrid=True, gridcolor=GRID,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.08)",
                   tickfont=dict(family="'IBM Plex Mono', monospace", size=10),
                   title=dict(text="f(x)", font=dict(color="rgba(255,107,43,0.6)", size=11))),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.02, y=1.18, xanchor="left", yanchor="top",
            buttons=[dict(
                label="▶  TRAÇAR",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=30, redraw=True),
                    fromcurrent=False,
                    transition=dict(duration=0),
                    mode="immediate",
                )]
            )],
            bgcolor="rgba(255,107,43,0.08)",
            bordercolor="rgba(255,107,43,0.35)",
            font=dict(family="'IBM Plex Mono', monospace", color="#ff6b2b", size=10),
        )],
    )

    return fig

# ─────────────────────────────────────────────────────────────
# 4) CABEÇALHO
# ─────────────────────────────────────────────────────────────
ticker_text = (
    "LIMITE &nbsp;<span class='sep'>·</span>&nbsp; "
    "lim x→0 sin(x)/x <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→∞ (1+1/x)^x <span class='pos'>= e</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0 (1-cos x)/x² <span class='pos'>= 1/2</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0⁺ x·ln(x) <span class='pos'>= 0</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→∞ n^(1/n) <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0 (eˣ-1)/x <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "LIMITE &nbsp;<span class='sep'>·</span>&nbsp; "
    "lim x→0 sin(x)/x <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→∞ (1+1/x)^x <span class='pos'>= e</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0 (1-cos x)/x² <span class='pos'>= 1/2</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0⁺ x·ln(x) <span class='pos'>= 0</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→∞ n^(1/n) <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
    "lim x→0 (eˣ-1)/x <span class='pos'>= 1</span> &nbsp;<span class='sep'>|</span>&nbsp; "
)

st.markdown(f"""
<div class="bbg-header">
  <div class="bbg-ticker">TEC · CÁLCULO · ANÁLISE · MÓDULO DE LIMITES</div>
  <div class="bbg-title">ANALISADOR DE <span>LIMITES</span></div>
  <div class="bbg-sub">ESTIMATIVA NUMÉRICA &nbsp;·&nbsp; CÁLCULO SIMBÓLICO &nbsp;·&nbsp; FUNÇÕES POR PARTES</div>
</div>
<div class="ticker-wrap">
  <div class="ticker-content">{ticker_text}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5) SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### // PARÂMETROS")
ponto      = st.sidebar.number_input("PONTO  x₀",     value=0.0,   step=0.1,  format="%.4f")
delta      = st.sidebar.number_input("DELTA  δ",      value=0.001, min_value=1e-10, max_value=1.0,  format="%.6f")
tolerancia = st.sidebar.number_input("TOLERÂNCIA  ε", value=0.01,  min_value=1e-10, max_value=10.0, format="%.6f")
st.sidebar.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p style='font-size:0.70rem;color:rgba(220,225,235,0.3);line-height:1.8;font-family:var(--mono);'>
CONDIÇÕES:<br>
x &lt; 0 &nbsp;|&nbsp; x &gt;= 2 &nbsp;|&nbsp; otherwise
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 6) MODO DE ENTRADA
# ─────────────────────────────────────────────────────────────
modo = st.radio("", ["// FUNÇÃO SIMPLES", "// FUNÇÃO POR PARTES"],
                horizontal=True, label_visibility="collapsed")
st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)

f = None; expr_sym = None; parse_ok = False

if modo == "// FUNÇÃO SIMPLES":
    exemplos = {
        "sin(x)/x":           ("sin(x)/x",        0.0),
        "(x²−4)/(x−2)":       ("(x**2-4)/(x-2)",  2.0),
        "exp(−x²)":           ("exp(-x**2)",       0.0),
        "Abs(x)/x  [sinal]":  ("Abs(x)/x",         0.0),
        "1/x":                ("1/x",              0.0),
    }
    c1, c2 = st.columns([1, 2])
    ex = c1.selectbox("EXEMPLO", list(exemplos.keys()), label_visibility="visible")
    default_fn, _ = exemplos[ex]
    expressao = c2.text_input("f(x) =", value=default_fn, placeholder="ex: sin(x)/x")
    try:
        f, expr_sym = parse_simples(expressao)
        parse_ok = True
    except Exception as e:
        st.error(f"ERRO DE PARSE: {e}")

else:
    exemplos_pw = {
        "SINAL  x=0  [limite não existe]": [
            {"expr": "-1",    "cond": "x < 0"},
            {"expr": "1",     "cond": "otherwise"},
        ],
        "SUAVE  x=0  [limite existe]": [
            {"expr": "x**2",  "cond": "x < 0"},
            {"expr": "x",     "cond": "otherwise"},
        ],
        "SALTO  x=2": [
            {"expr": "x + 1", "cond": "x < 2"},
            {"expr": "x + 3", "cond": "otherwise"},
        ],
    }
    c_ex, _ = st.columns([1, 2])
    ex_pw = c_ex.selectbox("EXEMPLO", list(exemplos_pw.keys()), label_visibility="visible")
    defaults = exemplos_pw[ex_pw]

    if "n_pecas" not in st.session_state:
        st.session_state.n_pecas = len(defaults)

    n = st.session_state.n_pecas
    pecas_input = []

    hc1, hc2, hc3 = st.columns([0.22, 1.5, 1.5])
    hc2.markdown("<p class='piece-label'>f(x)</p>", unsafe_allow_html=True)
    hc3.markdown("<p class='piece-label'>condição</p>", unsafe_allow_html=True)

    for i in range(n):
        d_e = defaults[i]["expr"] if i < len(defaults) else ""
        d_c = defaults[i]["cond"] if i < len(defaults) else ""
        c0, c1, c2 = st.columns([0.22, 1.5, 1.5])
        c0.markdown(f"<div style='font-family:var(--mono);font-size:0.85rem;color:var(--orange);opacity:0.6;padding-top:8px;text-align:center;'>[{i+1}]</div>", unsafe_allow_html=True)
        ei = c1.text_input(f"e{i}", value=d_e, key=f"expr_{i}", label_visibility="collapsed", placeholder="x**2")
        ci = c2.text_input(f"c{i}", value=d_c, key=f"cond_{i}", label_visibility="collapsed", placeholder="x < 0")
        pecas_input.append({"expr": ei, "cond": ci})

    ca, cr, _ = st.columns([1, 1, 5])
    if ca.button("+ ADICIONAR"):
        st.session_state.n_pecas += 1
        st.rerun()
    if cr.button("− REMOVER") and st.session_state.n_pecas > 1:
        st.session_state.n_pecas -= 1
        st.rerun()

    try:
        key   = str(pecas_input)
        exprs = tuple(p["expr"] for p in pecas_input)
        conds = tuple(p["cond"] for p in pecas_input)
        f, expr_sym = parse_piecewise(key, exprs, conds)
        parse_ok = True
    except Exception as e:
        st.error(f"ERRO DE PARSE: {e}")

# ─────────────────────────────────────────────────────────────
# 7) DISPLAY DA FUNÇÃO
# ─────────────────────────────────────────────────────────────
if parse_ok and expr_sym is not None:
    st.markdown("<div class='hr-orange'></div>", unsafe_allow_html=True)
    st.markdown("<div class='fn-display'><div class='fn-label'>// função reconhecida</div>", unsafe_allow_html=True)
    st.latex(r"f(x) \ = \ " + sp.latex(expr_sym))
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 8) CÁLCULO AUTOMÁTICO
# ─────────────────────────────────────────────────────────────
if parse_ok and f is not None:
    try:
        res = calcular_limite(f, ponto, delta, tolerancia)
    except Exception as e:
        st.error(f"ERRO DE AVALIAÇÃO: {e}")
        st.stop()

    lim_sym = lim_simbolico(expr_sym, ponto)

    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RESULTADO NUMÉRICO",
              f"{res['lim_final']:.6f}" if res["existe"] else "INDEFINIDO")
    m2.metric("LIMITE ESQUERDO", f"{res['lim_esq']:.6f}")
    m3.metric("LIMITE DIREITO",  f"{res['lim_dir']:.6f}",
              delta=f"Δ = {abs(res['lim_esq']-res['lim_dir']):.2e}",
              delta_color="normal" if res["existe"] else "inverse")
    m4.metric("SIMBÓLICO (SymPy)",
              sp.latex(lim_sym) if lim_sym is not None else "N/D")

    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)

    if res["existe"]:
        st.markdown("<div class='result-exists'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>▸ LIMITE EXISTE</div>", unsafe_allow_html=True)
        st.latex(rf"\lim_{{x \to {ponto}}} f(x) \ \approx \ {res['lim_final']:.6f}")
        if lim_sym is not None:
            st.latex(rf"\lim_{{x \to {ponto}}} f(x) \ = \ {sp.latex(lim_sym)} \qquad \scriptstyle{{\text{{[exato — SymPy]}}}}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-not-exists'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>▸ LIMITE NÃO EXISTE</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:var(--muted);font-family:var(--mono);font-size:0.82rem;margin:0;'>"
            f"DIVERGÊNCIA LATERAL = <span style='color:var(--red);'>{abs(res['lim_esq']-res['lim_dir']):.2e}</span>"
            f"  ·  TOLERÂNCIA = <span style='color:var(--red);'>{tolerancia}</span></p>",
            unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr-orange'></div>", unsafe_allow_html=True)
    try:
        fig = gerar_grafico(f, ponto, res)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"ERRO NO GRÁFICO: {e}")

    # ─────────────────────────────────────────────────────────
    # 9) TEORIA EXPANDIDA
    # ─────────────────────────────────────────────────────────
    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
    st.markdown("<div class='teoria-section'>", unsafe_allow_html=True)
    st.markdown("<div class='teoria-titulo'>// teoria matemática · definição, propriedades e método numérico</div>", unsafe_allow_html=True)

    # Bloco 1 — Definição informal
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>01 · Intuição</div><div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("""
O limite de uma função descreve o **comportamento de aproximação** — não o valor no ponto, mas o valor para o qual a função *tende* conforme x se aproxima de x₀. Uma função pode nem sequer estar definida em x₀ e ainda assim ter limite.

O exemplo mais clássico é `sin(x)/x` em x=0: a função não está definida em zero (divisão por zero), mas à medida que x → 0, o quociente tende inequivocamente a 1.
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Bloco 2 — Definição formal ε-δ
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>02 · Definição Formal (ε-δ de Cauchy-Weierstrass)</div>", unsafe_allow_html=True)
    st.latex(r"\lim_{x \to x_0} f(x) = L \iff \forall\,\varepsilon > 0,\;\exists\,\delta > 0 : 0 < |x - x_0| < \delta \;\Rightarrow\; |f(x) - L| < \varepsilon")
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("""
Esta definição formaliza a ideia de aproximação: para qualquer margem de erro ε que você escolha (por menor que seja), existe uma vizinhança δ ao redor de x₀ tal que f(x) fica dentro da margem ε de L. A condição `0 < |x - x₀|` exclui explicitamente o próprio ponto x₀.
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Bloco 3 — Limites laterais
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>03 · Limites Laterais e Condição de Existência</div>", unsafe_allow_html=True)
    st.latex(r"\lim_{x \to x_0^-} f(x) = \lim_{x \to x_0^+} f(x) = L \iff \lim_{x \to x_0} f(x) = L")
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("""
O **limite bilateral** só existe quando os dois limites laterais coincidem. Se a função se aproxima de valores diferentes pela esquerda e pela direita, o limite não existe — mesmo que cada lateral isoladamente seja bem definida.

**Exemplo:** a função sinal `sgn(x) = |x|/x` tem limite lateral esquerdo −1 e direito +1 em x=0. Como −1 ≠ 1, o limite bilateral não existe.
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Bloco 4 — Propriedades algébricas
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>04 · Propriedades Algébricas dos Limites</div>", unsafe_allow_html=True)
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("Sejam L e M limites de f e g em x₀, respectivamente. Então:")
    st.markdown("</div>", unsafe_allow_html=True)
    st.latex(r"""
\lim_{x\to x_0}[f(x)+g(x)] = L+M \qquad
\lim_{x\to x_0}[f(x)\cdot g(x)] = L\cdot M
""")
    st.latex(r"""
\lim_{x\to x_0}\frac{f(x)}{g(x)} = \frac{L}{M},\; M\neq 0 \qquad
\lim_{x\to x_0}[f(x)]^n = L^n
""")
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("Essas propriedades permitem decompor limites complexos em partes menores e mais simples.")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Bloco 5 — Limites notáveis
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>05 · Limites Fundamentais</div>", unsafe_allow_html=True)
    st.latex(r"\lim_{x \to 0} \frac{\sin x}{x} = 1 \qquad \lim_{x \to 0} \frac{e^x - 1}{x} = 1 \qquad \lim_{x \to 0} \frac{\ln(1+x)}{x} = 1")
    st.latex(r"\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e \qquad \lim_{x \to 0} \frac{1 - \cos x}{x^2} = \frac{1}{2} \qquad \lim_{x \to 0^+} x \ln x = 0")
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("Esses limites aparecem recorrentemente em cálculo, análise e finanças quantitativas. O segundo e o quarto são a base da definição do número de Euler **e** — a taxa de crescimento contínuo.")
    st.markdown("</div></div>", unsafe_allow_html=True)

    # Bloco 6 — Método numérico e limitações
    st.markdown("<div class='teoria-bloco'><div class='teoria-bloco-titulo'>06 · Método Numérico Implementado</div>", unsafe_allow_html=True)
    st.latex(r"L \approx \frac{f(x_0 - \delta) + f(x_0 + \delta)}{2} \quad \text{se} \quad |f(x_0-\delta) - f(x_0+\delta)| \leq \varepsilon")
    st.markdown("<div class='teoria-texto'>", unsafe_allow_html=True)
    st.markdown("""
Este módulo estima o limite avaliando f em dois pontos simétricos ao redor de x₀, separados por δ.

**Parâmetros e seu efeito:**
- `δ` muito grande → a estimativa não captura o comportamento local; você está medindo longe do ponto
- `δ` muito pequeno → erros de ponto flutuante dominam; a aritmética de 64 bits perde precisão
- `ε (tolerância)` → define o quanto os laterais podem diferir para ainda declarar que o limite existe

**Comparação com o SymPy:** o resultado simbólico (mostrado na 4ª métrica) é calculado analiticamente via regras de L'Hôpital, expansão em série de Taylor e simplificação algébrica. É **exato** quando possível — use-o para validar a estimativa numérica.

**Cuidado com funções por partes:** o método numérico avalia cada lado independentemente via `lambdify`, o que funciona corretamente. O SymPy, no entanto, pode ter dificuldade com `Piecewise` em alguns pontos de descontinuidade — nesse caso, prefira o resultado numérico.
""")
    st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # fecha teoria-section

# ─────────────────────────────────────────────────────────────
# 10) RODAPÉ
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  THE EVERYTHING CALCULATOR &nbsp;·&nbsp; <span>FELLIPE ALMÄSSY</span> &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
