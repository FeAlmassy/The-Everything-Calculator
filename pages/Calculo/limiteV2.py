
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
# 1) CSS — instrumento científico de luxo
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=JetBrains+Mono:wght@300;400;500&family=Archivo:wght@300;400;500&display=swap');

/* ── Reset e base ─────────────────────────────── */
:root {
  --ink:      #0a0b0f;
  --surface:  #0f1117;
  --panel:    #13161f;
  --rim:      rgba(255,255,255,0.06);
  --rim2:     rgba(255,255,255,0.10);
  --gold:     #c8a96e;
  --gold-dim: rgba(200,169,110,0.18);
  --gold-glow:rgba(200,169,110,0.06);
  --ice:      #7eb8d4;
  --ice-dim:  rgba(126,184,212,0.15);
  --ember:    #e07050;
  --text:     rgba(235,235,228,0.90);
  --muted:    rgba(235,235,228,0.45);
  --muted2:   rgba(235,235,228,0.22);
  --mono:     'JetBrains Mono', monospace;
  --serif:    'Cormorant Garamond', Georgia, serif;
  --sans:     'Archivo', sans-serif;
}

html, body, .stApp {
  background-color: var(--ink) !important;
  color: var(--text);
  font-family: var(--sans);
}

/* ── Sidebar ─────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--panel) !important;
  border-right: 1px solid var(--rim2);
}
section[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
section[data-testid="stSidebar"] .stMarkdown p {
  color: var(--muted); font-size: 0.82rem; letter-spacing: 0.03em;
}

/* ── Inputs & sliders ────────────────────────── */
input[type="number"], input[type="text"], .stTextInput input, .stNumberInput input {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--rim2) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.88rem !important;
}
input:focus { border-color: var(--gold) !important; outline: none !important; box-shadow: 0 0 0 2px var(--gold-dim) !important; }

/* ── Selectbox ───────────────────────────────── */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--rim2) !important;
  border-radius: 6px !important;
}

/* ── Radio ───────────────────────────────────── */
div[role="radiogroup"] label {
  font-family: var(--sans) !important;
  font-size: 0.88rem !important;
  color: var(--muted) !important;
}
div[role="radiogroup"] label[data-selected="true"],
div[role="radiogroup"] input:checked + div { color: var(--gold) !important; }

/* ── Métricas ────────────────────────────────── */
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, rgba(200,169,110,0.04), rgba(255,255,255,0.02));
  border: 1px solid var(--rim2);
  border-top: 2px solid var(--gold);
  border-radius: 2px 2px 8px 8px;
  padding: 18px 20px 14px;
  position: relative;
}
div[data-testid="stMetric"]::before {
  content: '';
  position: absolute;
  top: -1px; left: 20%; right: 20%;
  height: 1px;
  background: var(--gold);
  opacity: 0.6;
}
div[data-testid="stMetricLabel"] p {
  font-family: var(--sans) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
div[data-testid="stMetricValue"] {
  font-family: var(--serif) !important;
  font-size: 1.9rem !important;
  font-weight: 300 !important;
  color: var(--gold) !important;
  letter-spacing: -0.01em;
}
div[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}

/* ── Expander ────────────────────────────────── */
details { border: 1px solid var(--rim) !important; border-radius: 8px !important; background: var(--panel) !important; }
details summary { font-family: var(--sans) !important; font-size: 0.85rem !important; color: var(--muted) !important; letter-spacing: 0.06em; }

/* ── Divisores personalizados ────────────────── */
.hr-gold {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold), transparent);
  opacity: 0.3;
  margin: 2rem 0;
}
.hr-thin {
  border: none;
  border-top: 1px solid var(--rim);
  margin: 1.2rem 0;
}

/* ── Cabeçalho da página ─────────────────────── */
.page-header {
  display: flex;
  align-items: flex-end;
  gap: 1.5rem;
  padding: 2.5rem 0 1.5rem;
  border-bottom: 1px solid var(--rim2);
  margin-bottom: 2rem;
}
.page-symbol {
  font-family: var(--serif);
  font-size: 5rem;
  font-weight: 300;
  font-style: italic;
  color: var(--gold);
  line-height: 1;
  opacity: 0.85;
  text-shadow: 0 0 60px rgba(200,169,110,0.3);
}
.page-title {
  font-family: var(--serif);
  font-size: 2.4rem;
  font-weight: 300;
  letter-spacing: -0.02em;
  color: var(--text);
  line-height: 1.1;
  margin: 0;
}
.page-subtitle {
  font-family: var(--sans);
  font-size: 0.82rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 0.4rem;
}

/* ── Display da função ───────────────────────── */
.fn-display {
  background: linear-gradient(135deg, var(--gold-glow), transparent);
  border: 1px solid rgba(200,169,110,0.20);
  border-left: 3px solid var(--gold);
  border-radius: 0 10px 10px 0;
  padding: 1.4rem 2rem;
  margin: 1.5rem 0;
  font-family: var(--serif);
}
.fn-label {
  font-family: var(--sans);
  font-size: 0.70rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--gold);
  margin-bottom: 0.5rem;
  opacity: 0.8;
}

/* ── Badge de resultado ──────────────────────── */
.result-exists {
  background: linear-gradient(135deg, rgba(126,184,212,0.08), rgba(126,184,212,0.03));
  border: 1px solid rgba(126,184,212,0.25);
  border-radius: 8px;
  padding: 1.2rem 1.6rem;
  margin: 1rem 0;
}
.result-not-exists {
  background: linear-gradient(135deg, rgba(224,112,80,0.08), rgba(224,112,80,0.03));
  border: 1px solid rgba(224,112,80,0.25);
  border-radius: 8px;
  padding: 1.2rem 1.6rem;
  margin: 1rem 0;
}
.result-label {
  font-family: var(--sans);
  font-size: 0.70rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.result-exists .result-label   { color: var(--ice); }
.result-not-exists .result-label { color: var(--ember); }

/* ── Piece box (por partes) ──────────────────── */
.piece-row {
  display: grid;
  grid-template-columns: 28px 1fr 1fr;
  align-items: center;
  gap: 12px;
  background: rgba(255,255,255,0.018);
  border: 1px solid var(--rim);
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 8px;
}
.piece-num {
  font-family: var(--serif);
  font-style: italic;
  font-size: 1.1rem;
  color: var(--gold);
  opacity: 0.7;
  text-align: center;
}
.piece-sep {
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--muted2);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-align: center;
  margin: 4px 0;
}

/* ── Badge tags ──────────────────────────────── */
.tag {
  display: inline-block;
  padding: 0.15rem 0.6rem;
  border-radius: 3px;
  border: 1px solid var(--rim2);
  background: rgba(255,255,255,0.03);
  font-family: var(--sans);
  font-size: 0.72rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-right: 6px;
}

/* ── Rodapé ──────────────────────────────────── */
.footer {
  text-align: center;
  color: var(--muted2);
  font-family: var(--sans);
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  padding: 3rem 0 1.5rem;
  border-top: 1px solid var(--rim);
  margin-top: 4rem;
}
.footer span { color: var(--gold); opacity: 0.6; }

/* ── Streamlit overrides gerais ──────────────── */
h1, h2, h3 { font-family: var(--serif) !important; font-weight: 300 !important; }
.stButton button {
  background: transparent !important;
  border: 1px solid var(--gold) !important;
  color: var(--gold) !important;
  font-family: var(--sans) !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
  padding: 0.4rem 1rem !important;
  transition: all 0.2s !important;
}
.stButton button:hover {
  background: var(--gold-dim) !important;
  box-shadow: 0 0 20px var(--gold-dim) !important;
}
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
# 3) GRÁFICO
# ─────────────────────────────────────────────────────────────
def gerar_grafico(f, ponto, resultado):
    margem = max(2.5, abs(ponto) * 1.0) if ponto != 0 else 3.5
    xs = np.linspace(ponto - margem, ponto + margem, 900)
    try:
        ys = np.array(f(xs), dtype=float)
    except Exception:
        ys = np.array([float(f(xi)) for xi in xs], dtype=float)

    if np.any(np.isfinite(ys)):
        cap = np.nanpercentile(np.abs(ys[np.isfinite(ys)]), 98) * 4
        ys = np.where(np.isfinite(ys) & (np.abs(ys) < cap), ys, np.nan)

    GOLD   = "#c8a96e"
    ICE    = "#7eb8d4"
    EMBER  = "#e07050"
    GREEN  = "#7ecba1"
    BG     = "#0a0b0f"
    GRID   = "rgba(255,255,255,0.05)"

    fig = go.Figure()

    # glow
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
        line=dict(color="rgba(200,169,110,0.12)", width=14),
        hoverinfo="skip", showlegend=False))

    # curva
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
        line=dict(color=GOLD, width=2.2),
        hovertemplate="<b>x</b> = %{x:.5f}<br><b>f(x)</b> = %{y:.5f}<extra></extra>"))

    # linha vertical x₀
    fig.add_vline(x=ponto,
        line=dict(color="rgba(255,255,255,0.18)", dash="dot", width=1))

    d = resultado["delta"]
    cor_esq = ICE
    cor_dir = EMBER

    # limite esquerdo
    fig.add_trace(go.Scatter(
        x=[ponto - d], y=[resultado["lim_esq"]], mode="markers",
        name=f"esq  {resultado['lim_esq']:.5f}",
        marker=dict(color=cor_esq, size=10, symbol="circle",
                    line=dict(color="white", width=1))))

    # limite direito
    fig.add_trace(go.Scatter(
        x=[ponto + d], y=[resultado["lim_dir"]], mode="markers",
        name=f"dir  {resultado['lim_dir']:.5f}",
        marker=dict(color=cor_dir, size=10, symbol="circle",
                    line=dict(color="white", width=1))))

    # ponto do limite
    if resultado["existe"]:
        fig.add_trace(go.Scatter(
            x=[ponto], y=[resultado["lim_final"]], mode="markers",
            name=f"L ≈ {resultado['lim_final']:.5f}",
            marker=dict(color=GREEN, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="'Archivo', sans-serif", color="rgba(235,235,228,0.75)", size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(10,11,15,0.8)", bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1, font=dict(family="'JetBrains Mono', monospace", size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=400,
        xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.10)",
                   tickfont=dict(family="'JetBrains Mono', monospace", size=10)),
        yaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=1,
                   zeroline=True, zerolinecolor="rgba(255,255,255,0.10)",
                   tickfont=dict(family="'JetBrains Mono', monospace", size=10)),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# 4) CABEÇALHO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-symbol">∂</div>
  <div>
    <div class="page-title">Teoria dos Limites</div>
    <div class="page-subtitle">Análise numérica &nbsp;·&nbsp; Cálculo simbólico &nbsp;·&nbsp; Funções por partes</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 5) SIDEBAR
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("### Parâmetros")

ponto     = st.sidebar.number_input("Ponto  x₀", value=0.0, step=0.1, format="%.4f")
delta     = st.sidebar.number_input("δ  (delta)", value=0.001, min_value=1e-10, max_value=1.0, format="%.6f",
                                     help="Distância lateral usada para estimar os limites.")
tolerancia= st.sidebar.number_input("Tolerância", value=0.01,  min_value=1e-10, max_value=10.0, format="%.6f",
                                     help="Máxima diferença entre lim esq e lim dir para aceitar existência.")

st.sidebar.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p style='font-size:0.75rem;color:var(--muted2);line-height:1.6;'>
Condições aceitas:<br>
<code style='color:var(--gold);font-size:0.72rem;'>x &lt; 0</code> &nbsp;
<code style='color:var(--gold);font-size:0.72rem;'>x &gt;= 2</code> &nbsp;
<code style='color:var(--gold);font-size:0.72rem;'>otherwise</code>
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 6) MODO DE ENTRADA
# ─────────────────────────────────────────────────────────────
modo = st.radio("", ["Função simples", "Função por partes"], horizontal=True, label_visibility="collapsed")
st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)

f        = None
expr_sym = None
parse_ok = False

# ── Simples ──────────────────────────────────────────────────
if modo == "Função simples":
    exemplos = {
        "sin(x) / x":        ("sin(x)/x",        0.0),
        "(x²−4) / (x−2)":    ("(x**2-4)/(x-2)",  2.0),
        "exp(−x²)":          ("exp(-x**2)",       0.0),
        "|x| / x  (sinal)":  ("Abs(x)/x",         0.0),
        "1 / x":             ("1/x",              0.0),
    }
    col_ex, col_fn = st.columns([1, 2])
    ex = col_ex.selectbox("Exemplo", list(exemplos.keys()), label_visibility="collapsed")
    default_fn, default_pt = exemplos[ex]
    expressao = col_fn.text_input("f(x)", value=default_fn, label_visibility="collapsed",
                                   placeholder="ex: sin(x)/x")
    try:
        f, expr_sym = parse_simples(expressao)
        parse_ok = True
    except Exception as e:
        st.error(f"Expressão inválida: {e}")

# ── Por partes ───────────────────────────────────────────────
else:
    exemplos_pw = {
        "Sinal em x=0 (não existe)": [
            {"expr": "-1",    "cond": "x < 0"},
            {"expr": "1",     "cond": "otherwise"},
        ],
        "Contínua em x=0 (existe)": [
            {"expr": "x**2",  "cond": "x < 0"},
            {"expr": "x",     "cond": "otherwise"},
        ],
        "Salto em x=2": [
            {"expr": "x + 1", "cond": "x < 2"},
            {"expr": "x + 3", "cond": "otherwise"},
        ],
    }

    col_ex2, _ = st.columns([1, 2])
    ex_pw = col_ex2.selectbox("Exemplo", list(exemplos_pw.keys()), label_visibility="collapsed")
    defaults = exemplos_pw[ex_pw]

    if "n_pecas" not in st.session_state:
        st.session_state.n_pecas = len(defaults)

    n = st.session_state.n_pecas
    pecas_input = []

    # cabeçalho das colunas
    hc1, hc2, hc3 = st.columns([0.25, 1.5, 1.5])
    hc2.markdown("<p style='font-size:0.70rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted2);margin-bottom:4px;'>f(x)</p>", unsafe_allow_html=True)
    hc3.markdown("<p style='font-size:0.70rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted2);margin-bottom:4px;'>quando</p>", unsafe_allow_html=True)

    for i in range(n):
        d_e = defaults[i]["expr"] if i < len(defaults) else ""
        d_c = defaults[i]["cond"] if i < len(defaults) else ""
        c0, c1, c2 = st.columns([0.25, 1.5, 1.5])
        c0.markdown(f"<div style='font-family:var(--serif);font-style:italic;font-size:1.2rem;color:var(--gold);opacity:0.7;padding-top:6px;text-align:center;'>{i+1}</div>", unsafe_allow_html=True)
        ei = c1.text_input(f"e{i}", value=d_e, key=f"expr_{i}", label_visibility="collapsed", placeholder="x**2")
        ci = c2.text_input(f"c{i}", value=d_c, key=f"cond_{i}", label_visibility="collapsed", placeholder="x < 0")
        pecas_input.append({"expr": ei, "cond": ci})

    ca, cr, _ = st.columns([1, 1, 5])
    if ca.button("＋ pedaço"):
        st.session_state.n_pecas += 1
        st.rerun()
    if cr.button("－ remover") and st.session_state.n_pecas > 1:
        st.session_state.n_pecas -= 1
        st.rerun()

    try:
        key = str(pecas_input)
        exprs = tuple(p["expr"] for p in pecas_input)
        conds = tuple(p["cond"] for p in pecas_input)
        f, expr_sym = parse_piecewise(key, exprs, conds)
        parse_ok = True
    except Exception as e:
        st.error(f"Erro na função por partes: {e}")

# ─────────────────────────────────────────────────────────────
# 7) DISPLAY DA FUNÇÃO (automático)
# ─────────────────────────────────────────────────────────────
if parse_ok and expr_sym is not None:
    st.markdown("<div class='hr-gold'></div>", unsafe_allow_html=True)
    st.markdown("<div class='fn-display'><div class='fn-label'>Função reconhecida</div>", unsafe_allow_html=True)
    st.latex(r"f(x) \ = \ " + sp.latex(expr_sym))
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 8) CÁLCULO AUTOMÁTICO
# ─────────────────────────────────────────────────────────────
if parse_ok and f is not None:
    try:
        res = calcular_limite(f, ponto, delta, tolerancia)
    except Exception as e:
        st.error(f"Erro ao avaliar: {e}")
        st.stop()

    lim_sym = lim_simbolico(expr_sym, ponto)

    # ── Métricas ─────────────────────────────────────────────
    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Resultado numérico",
              f"{res['lim_final']:.6f}" if res["existe"] else "—")
    m2.metric("Lim. pela esquerda", f"{res['lim_esq']:.6f}")
    m3.metric("Lim. pela direita",  f"{res['lim_dir']:.6f}",
              delta=f"Δ = {abs(res['lim_esq']-res['lim_dir']):.2e}",
              delta_color="normal" if res["existe"] else "inverse")
    m4.metric("Simbólico (SymPy)", sp.latex(lim_sym) if lim_sym is not None else "n/a")

    # ── Banner de resultado ───────────────────────────────────
    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)

    if res["existe"]:
        st.markdown("<div class='result-exists'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>✦ Limite existe</div>", unsafe_allow_html=True)
        st.latex(rf"\lim_{{x \to {ponto}}} f(x) \ \approx \ {res['lim_final']:.6f}")
        if lim_sym is not None:
            st.latex(rf"\lim_{{x \to {ponto}}} f(x) \ = \ {sp.latex(lim_sym)} \qquad \scriptstyle{{\text{{(exato — SymPy)}}}}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-not-exists'>", unsafe_allow_html=True)
        st.markdown("<div class='result-label'>✦ Limite não existe</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:var(--muted);font-size:0.88rem;margin:0;'>Os limites laterais divergem em <code style='color:var(--ember);'>{abs(res['lim_esq']-res['lim_dir']):.2e}</code>, acima da tolerância <code style='color:var(--ember);'>{tolerancia}</code>.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Gráfico ───────────────────────────────────────────────
    st.markdown("<div class='hr-gold'></div>", unsafe_allow_html=True)
    try:
        fig = gerar_grafico(f, ponto, res)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Gráfico indisponível: {e}")

    # ── Teoria (expander) ─────────────────────────────────────
    st.markdown("<div class='hr-thin'></div>", unsafe_allow_html=True)
    with st.expander("teoria  ·  definição formal e método numérico"):
        st.markdown("<span class='tag'>ε-δ</span><span class='tag'>limites laterais</span><span class='tag'>piecewise</span><span class='tag'>numérico vs simbólico</span>", unsafe_allow_html=True)
        st.markdown("")

        st.markdown("**Definição formal**")
        st.latex(r"\forall\,\varepsilon > 0,\;\exists\,\delta > 0 : 0 < |x - x_0| < \delta \;\Rightarrow\; |f(x) - L| < \varepsilon")

        st.markdown("**Condição de existência (limites laterais)**")
        st.latex(r"\lim_{x \to x_0} f(x) = L \iff \lim_{x \to x_0^-} f(x) = \lim_{x \to x_0^+} f(x) = L")

        st.markdown("**Estimativa numérica usada aqui**")
        st.latex(r"L \approx \frac{f(x_0 - \delta) + f(x_0 + \delta)}{2} \quad \text{se} \quad |f(x_0-\delta) - f(x_0+\delta)| \leq \varepsilon")

        st.markdown("""
**Cuidados:**  δ muito pequeno → erros de ponto flutuante.
δ muito grande → não captura comportamento local.
O resultado simbólico do SymPy é exato quando possível — use-o para validar.
        """)

# ─────────────────────────────────────────────────────────────
# 9) RODAPÉ
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  The Everything Calculator &nbsp;·&nbsp; <span>Fellipe Almässy</span> &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)
