# streamlit_app.py
# Mecanismo de Integração Quantitativa — Versão Profissional (Streamlit + Plotly)
# ------------------------------------------------------------
# - Parsing robusto com SymPy + lambdify do NumPy
# - Múltiplos métodos de quadratura
# - Referência SciPy quad quando disponível
# - Diagnósticos de convergência + estimativa de ordem observada (log-log safe)
# - Computação em cache para performance
# - Arquitetura de UI limpa (abas) + estilo institucional

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st
from scipy.integrate import quad


# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA)
# ----------------------------
st.set_page_config(page_title="Integrais Indefinidas", layout="wide")


# ----------------------------
# 1) ESTILO (CSS)
# ----------------------------
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

.main { background-color: var(--bg); }
section[data-testid="stSidebar"] { background-color: #0b1020; border-right: 1px solid var(--border); }
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px;
}

.hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 0.75rem 0 1.0rem 0;
}

.small-muted { color: var(--muted); font-size: 0.92rem; }
.badge {
  display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
  color: rgba(229,231,235,0.80); font-size: 0.82rem;
}
.footer { text-align:center; color: var(--muted2); margin-top: 14px; font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 2) MÉTODOS NUMÉRICOS (ENGINE)
# ----------------------------
def riemann_esquerda(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return float(np.sum(f(x)) * h)


def riemann_direita(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return float(np.sum(f(x)) * h)


def riemann_ponto_medio(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return float(np.sum(f(x)) * h)


def trapezoidal(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (np.sum(y) - 0.5 * (y[0] + y[-1])))


def simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float((h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])))


METODOS: Dict[str, Callable[[Callable[[np.ndarray], np.ndarray], float, float, int], float]] = {
    "Riemann Esquerda": riemann_esquerda,
    "Riemann Direita": riemann_direita,
    "Ponto Médio": riemann_ponto_medio,
    "Trapezoidal": trapezoidal,
    "Simpson": simpson,
}

ORDEM_TEORICA = {
    "Riemann Esquerda": 1,
    "Riemann Direita": 1,
    "Ponto Médio": 2,
    "Trapezoidal": 2,
    "Simpson": 4,
}


# ----------------------------
# 3) UTILITÁRIOS DE CACHE
# ----------------------------
@st.cache_data(show_spinner=False)
def build_xcurve(a: float, b: float, pontos: int = 1400) -> np.ndarray:
    pad = 0.15 * (b - a)
    return np.linspace(a - pad, b + pad, pontos)


@st.cache_resource(show_spinner=False)
def parse_function(expr_str: str):
    x_sym = sp.Symbol("x", real=True)
    locals_map = {
        "x": x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    expr = sp.sympify(expr_str, locals=locals_map)
    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
    return expr, f_num


@st.cache_data(show_spinner=False)
def safe_eval_curve(expr_str: str, a: float, b: float, pontos: int = 1400) -> Tuple[np.ndarray, np.ndarray]:
    expr, f_num = parse_function(expr_str)
    x_curve = build_xcurve(a, b, pontos)

    try:
        y = f_num(x_curve)
        y = np.array(y, dtype=float)
    except Exception:
        y = np.array([float(f_num(xx)) for xx in x_curve], dtype=float)

    y[~np.isfinite(y)] = np.nan
    return x_curve, y


@st.cache_data(show_spinner=False)
def compute_reference_quad(expr_str: str, a: float, b: float, max_subdiv: int = 200) -> Tuple[Optional[float], Optional[float]]:
    try:
        _, f_num = parse_function(expr_str)
        val, err = quad(lambda t: float(f_num(t)), a, b, limit=max_subdiv)
        return float(val), float(err)
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def series_convergencia(expr_str: str, a: float, b: float, nome_metodo: str, n_max: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    _, f_num = parse_function(expr_str)
    ref, _ = compute_reference_quad(expr_str, a, b)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    ns = np.arange(10, n_max + 1, step, dtype=int)
    fn = METODOS[nome_metodo]

    errs = []
    for nn in ns:
        v = fn(f_num, a, b, int(nn))
        errs.append(abs(ref - v))
    return ns, np.array(errs, dtype=float)


# ----------------------------
# 4) PAINEL TEÓRICO
# ----------------------------
def theory_panel():
    st.markdown("### Teoria Explicada")
    st.markdown(
        "<span class='badge'>Tipos de apoximação</span> "
        "<span class='badge'>Ordens de Erro</span> "
        "<span class='badge'>Log-Log</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Abrir teoria completa (métodos, exemplos, erro e convergência)", expanded=False):
        st.markdown("Este motor aproxima a integral definida")
        st.latex(r"\displaystyle I = \int_a^b f(x)\,dx")

        st.markdown("discretizando o intervalo em **n** subintervalos de largura")
        st.latex(r"\displaystyle h = \Delta x = \frac{b-a}{n}")

        st.markdown("""
e substituindo a função localmente por uma aproximação de baixo grau (constante, linear, quadrática).

---

### 1) Somas de Riemann (Esquerda / Direita)
**Modelo:** aproximação constante por partes.
""")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f(x_i)\,h")
        st.markdown("""
- Regra da Esquerda usa \(x_i=a+ih\); Regra da Direita usa \(x_i=a+(i+1)h\).
- Se \(f\) for suave, o erro global escala como:
""")
        st.latex(r"\displaystyle \text{Erro} = O(h)")
        st.markdown("""
**Viés prático:** - \(f\) crescente ⇒ Esquerda subestima, Direita superestima.  
- \(f\) decrescente ⇒ Esquerda superestima, Direita subestima.  

**Exemplo:** \(\int_0^1 x^2 dx = 1/3\). Com n pequeno, os retângulos da esquerda ficam abaixo porque \(x^2\) é crescente.
""")

        st.markdown("---")

        st.markdown("""
### 2) Regra do Ponto Médio
**Modelo:** constante por partes no ponto central (cancela o termo de primeira ordem).
""")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f\!\left(a+\left(i+\tfrac{1}{2}\right)h\right) h")
        st.markdown("Para \(f\) suave:")
        st.latex(r"\displaystyle \text{Erro} = O(h^2)")
        st.markdown("Reduzir \(h\) pela metade diminui o erro por ~4 (convergência quadrática).")

        st.markdown("---")

        st.markdown("""
### 3) Regra Trapezoidal
**Modelo:** interpolação linear por partes.
""")
        st.latex(r"\displaystyle I \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]")
        st.markdown("O erro no caso suave comporta-se como:")
        st.latex(r"\displaystyle \text{Erro} = -\frac{(b-a)}{12} h^2 f''(\xi) \quad \Rightarrow \quad O(h^2)")
        st.markdown("Estável e amplamente utilizada em contextos aplicados.")

        st.markdown("---")

        st.markdown("""
### 4) Regra de Simpson
**Modelo:** interpolação quadrática por partes (parabólica). Requer **n par**.
""")
        st.latex(r"\displaystyle I \approx \frac{h}{3}\left[f(x_0) + 4\sum_{\text{ímpar}} f(x_i) + 2\sum_{\text{par}} f(x_i) + f(x_n)\right]")
        st.markdown("O erro no caso suave comporta-se como:")
        st.latex(r"\displaystyle \text{Erro} = -\frac{(b-a)}{180} h^4 f^{(4)}(\xi) \quad \Rightarrow \quad O(h^4)")
        st.markdown("Reduzir \(h\) pela metade pode reduzir o erro por ~16 para funções suaves.")

        st.markdown("---")

        st.markdown("""
### 5) Diagnóstico de Convergência Log-Log
Se
""")
        st.latex(r"\displaystyle \text{Erro} \approx C h^p")
        st.markdown("então")
        st.latex(r"\displaystyle \log(\text{Erro}) \approx p\log(h) + \log(C)")
        st.markdown("A inclinação \(p\) é a **ordem de convergência observada**, estimada numericamente por este app.")


# ----------------------------
# 5) GRÁFICOS
# ----------------------------
def make_main_plot(expr_str: str, expr: sp.Expr, f_num, a: float, b: float, n: int, show_rectangles: bool) -> go.Figure:
    h = (b - a) / n
    x_curve, y_curve = safe_eval_curve(expr_str, a, b, pontos=1400)

    fig = go.Figure()

    mask = (x_curve >= a) & (x_curve <= b)
    fig.add_trace(go.Scatter(
        x=x_curve[mask], y=y_curve[mask],
        fill="tozeroy",
        name="Área (visual)",
        fillcolor="rgba(255, 75, 75, 0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines",
        line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines",
        name="f(x)",
        line=dict(color="#FF4B4B", width=3),
        hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>",
    ))

    if show_rectangles:
        x_left = np.linspace(a, b - h, n)
        y_left = f_num(x_left)
        y_left = np.array(y_left, dtype=float)
        y_left[~np.isfinite(y_left)] = np.nan

        fig.add_trace(go.Bar(
            x=x_left, y=y_left, width=h,
            name="Barras da Partição",
            marker=dict(color="#1E90FF", opacity=0.55, line=dict(color="rgba(255,255,255,0.35)", width=0.5)),
            hovertemplate="x=%{x:.6f}<br>altura=%{y:.6f}<extra></extra>",
        ))

    fig.add_vline(x=a, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")
    fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=450),
        title=f"f(x) = {sp.sstr(expr)}   |   [{a:.6g}, {b:.6g}]   |   n={n}",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def estimate_observed_order(ns: np.ndarray, errs: np.ndarray, a: float, b: float) -> Optional[float]:
    if len(ns) < 5:
        return None

    eps = 1e-300
    h = (b - a) / ns.astype(float)
    y = np.log(np.maximum(errs, eps))
    x = np.log(h)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return None

    p = np.polyfit(x, y, 1)[0]
    return float(p)


def make_convergence_plot(expr_str: str, a: float, b: float, nome_metodo: str, n_max: int, step: int, loglog: bool):
    ns, errs = series_convergencia(expr_str, a, b, nome_metodo, n_max, step)
    if ns.size == 0:
        return None, None, ns, errs

    eps = 1e-300
    errs_plot = np.array(errs, dtype=float)
    if loglog:
        errs_plot = np.maximum(errs_plot, eps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=errs_plot,
        mode="lines",
        name=f"Erro ({nome_metodo})",
        hovertemplate="n=%{x}<br>erro=%{y:.3e}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title="Erro Absoluto vs n",
        xaxis_title="n (partições)",
        yaxis_title="Erro Absoluto",
        transition=dict(duration=450),
    )

    if loglog:
        fig.update_layout(xaxis_type="log", yaxis_type="log")

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    p_obs = estimate_observed_order(ns, np.maximum(errs, eps), a, b)
    return fig, p_obs, ns, errs


# ----------------------------
# 6) CABEÇALHO
# ----------------------------
st.title("Integrais Indefinidas")
st.caption("Métodos Numéricos e Teoria Explicada")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

theory_panel()

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 7) SIDEBAR (BARRA LATERAL)
# ----------------------------
st.sidebar.header("Controles")

exemplos = {
    "Suave": "x**2 * sin(x)",
    "Oscilatória": "sin(50*x) / (1 + x**2)",
    "Mod": "Abs(x)",
    "Exponencial": "exp(-x**2)",
}
exemplo_escolhido = st.sidebar.selectbox("Exemplos rápidos", list(exemplos.keys()), index=3)
default_expr = exemplos[exemplo_escolhido]

expr_str = st.sidebar.text_input("f(x) (Sintaxe SymPy)", value=default_expr)

colA, colB = st.sidebar.columns(2)
a = colA.number_input("a", value=-2.0, format="%.6f")
b = colB.number_input("b", value=2.0, format="%.6f")

if a == b:
    st.error("a e b não podem ser iguais.")
    st.stop()
if a > b:
    st.sidebar.warning("Invertendo limites pois a > b.")
    a, b = b, a

nome_metodo = st.sidebar.selectbox("Método principal", list(METODOS.keys()), index=2)

n = st.sidebar.slider("Quantidade de partições", 10, 4000, 400, step=10)
show_rectangles = st.sidebar.checkbox("Mostrar barras de partição", value=True)

st.sidebar.markdown("---")
show_conv = st.sidebar.checkbox("Mostrar diagnósticos de convergência", value=True)
n_max = st.sidebar.slider("N máx de convergência", 200, 12000, 4000, step=100)
step = st.sidebar.slider("Passo de convergência", 10, 400, 40, step=10)
loglog = st.sidebar.checkbox("Visão Log-log", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Nota: descontinuidades severas podem quebrar o quad e distorcer a convergência.")


# ----------------------------
# 8) PARSE DA FUNÇÃO
# ----------------------------
try:
    expr, f_num = parse_function(expr_str)
    test = f_num(np.array([a, (a + b) / 2, b], dtype=float))
    _ = np.array(test, dtype=float)
except Exception as e:
    st.error(f"Função inválida. Falha no parsing/avaliação: {e}")
    st.stop()


# ----------------------------
# 9) REFERÊNCIA (quad)
# ----------------------------
ref_val, ref_err = compute_reference_quad(expr_str, a, b)


# ----------------------------
# 10) CALCULAR RESULTADOS PARA n ATUAL
# ----------------------------
h = (b - a) / n

rows = []
for nome, fn in METODOS.items():
    t0 = time.time()
    val = fn(f_num, a, b, n)
    t1 = time.time()
    err = abs(ref_val - val) if ref_val is not None else np.nan
    rows.append([nome, val, err, t1 - t0, ORDEM_TEORICA.get(nome, np.nan)])

df = pd.DataFrame(rows, columns=["Método", "Aproximação", "Erro Abs (vs quad)", "Tempo (s)", "Ordem Teórica p"])
df_sorted = df.sort_values(by=["Erro Abs (vs quad)"], ascending=True, na_position="last")

primary_val = float(df[df["Método"] == nome_metodo]["Aproximação"].iloc[0])


# ----------------------------
# 11) FAIXA DE MÉTRICAS
# ----------------------------
m1, m2, m3, m4, m5 = st.columns([1.2, 1.1, 1.1, 1.0, 1.0])

m1.metric("Aprox. Principal", f"{primary_val:.8f}", f"h = {h:.6g}")
if ref_val is not None:
    m2.metric("SciPy quad", f"{ref_val:.8f}", f"± {ref_err:.2e}")
    m3.metric("Erro Absoluto", f"{abs(ref_val - primary_val):.6e}", delta_color="inverse")
else:
    m2.metric("SciPy quad", "n/a", "ref. indisponível")
    m3.metric("Erro Absoluto", "n/a", "—")

m4.metric("n partições", f"{n}", "runtime escala ~O(n)")
m5.metric("Ordem p do Método", f"{ORDEM_TEORICA.get(nome_metodo,'—')}", "teórica")


st.markdown("<div class='small-muted'>Tabela de diagnóstico (calculada no n atual):</div>", unsafe_allow_html=True)
st.dataframe(df_sorted, use_container_width=True, hide_index=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 12) ABAS: ENGINE / DIAGNÓSTICOS
# ----------------------------
tab_engine, tab_diag = st.tabs(["Visão do Motor", "Diagnósticos"])

with tab_engine:
    fig = make_main_plot(expr_str, expr, f_num, a, b, n, show_rectangles)
    st.plotly_chart(fig, use_container_width=True)

with tab_diag:
    if show_conv and ref_val is not None:
        t0c = time.time()
        fig_c, p_obs, ns, errs = make_convergence_plot(expr_str, a, b, nome_metodo, n_max, step, loglog)
        t1c = time.time()

        if fig_c is not None:
            st.plotly_chart(fig_c, use_container_width=True)

            theo = ORDEM_TEORICA.get(nome_metodo, None)
            c1, c2, c3 = st.columns(3)

            if p_obs is not None:
                c1.metric("Ordem observada (slope)", f"{p_obs:.3f}", "log(err) vs log(h)")
            else:
                c1.metric("Ordem observada (slope)", "n/a", "dados insuficientes")

            c2.metric("Ordem teórica", f"{theo}" if theo is not None else "—", "esperada")
            c3.metric("Runtime do diagnóstico", f"{(t1c - t0c):.3f}s", f"{len(ns)} pontos")

            small = pd.DataFrame({"n": ns, "erro_abs": errs})
            st.markdown("<div class='small-muted'>Amostras brutas de convergência:</div>", unsafe_allow_html=True)
            st.dataframe(small, use_container_width=True, hide_index=True)
        else:
            st.info("Não foi possível gerar o gráfico de convergência.")
    else:
        st.info("Diagnósticos de convergência requerem uma referência quad válida. Tente uma função mais suave.")


# ----------------------------
# 13) RODAPÉ
# ----------------------------
st.markdown("<div class='footer'>The Everything Calculator - Fellipe Almässy • </div>", unsafe_allow_html=True)
