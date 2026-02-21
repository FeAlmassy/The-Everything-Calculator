# streamlit_app.py
# Quant Integration Engine — Professional Build (Streamlit + Plotly)
# ------------------------------------------------------------
# - Robust SymPy parsing + NumPy lambdify
# - Multiple quadrature methods
# - SciPy quad reference when available
# - Convergence diagnostics + observed order estimation
# - Cached computations for performance
# - Clean UI architecture (tabs) + institutional styling

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st
from scipy.integrate import quad


# ----------------------------
# 0) PAGE CONFIG (MUST BE FIRST)
# ----------------------------
st.set_page_config(page_title="Quant Integration Engine", layout="wide")


# ----------------------------
# 1) STYLE
# ----------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --panel: rgba(255,255,255,0.04);
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
# 2) NUMERICAL METHODS (ENGINE)
# ----------------------------
def riemann_left(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return float(np.sum(f(x)) * h)

def riemann_right(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return float(np.sum(f(x)) * h)

def riemann_midpoint(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)
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
    return float((h / 3) * (y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2])))

METHODS: Dict[str, Callable[[Callable[[np.ndarray], np.ndarray], float, float, int], float]] = {
    "Riemann Left": riemann_left,
    "Riemann Right": riemann_right,
    "Midpoint": riemann_midpoint,
    "Trapezoidal": trapezoidal,
    "Simpson": simpson,
}

THEORETICAL_ORDER = {
    "Riemann Left": 1,
    "Riemann Right": 1,
    "Midpoint": 2,
    "Trapezoidal": 2,
    "Simpson": 4,
}


# ----------------------------
# 3) DATA MODELS
# ----------------------------
@dataclass(frozen=True)
class ParsedFunction:
    expr: sp.Expr
    f_num: Callable[[np.ndarray], np.ndarray]


# ----------------------------
# 4) CACHING UTILITIES
# ----------------------------
@st.cache_data(show_spinner=False)
def build_xcurve(a: float, b: float, points: int = 1400) -> np.ndarray:
    pad = 0.15 * (b - a)
    return np.linspace(a - pad, b + pad, points)

@st.cache_data(show_spinner=False)
def safe_eval_curve(f_num, x_curve: np.ndarray) -> np.ndarray:
    # vector eval with fallback
    try:
        y = f_num(x_curve)
        y = np.array(y, dtype=float)
    except Exception:
        y = np.array([float(f_num(xx)) for xx in x_curve], dtype=float)

    # sanitize
    y[~np.isfinite(y)] = np.nan
    return y

@st.cache_data(show_spinner=False)
def compute_reference_quad(expr_str: str, a: float, b: float, max_subdiv: int = 200) -> Tuple[Optional[float], Optional[float]]:
    # The key is expr_str so cache invalidates properly when user changes expression
    try:
        x_sym = sp.Symbol("x", real=True)
        locals_map = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                      "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi}
        expr = sp.sympify(expr_str, locals=locals_map)
        f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
        val, err = quad(lambda t: float(f_num(t)), a, b, limit=max_subdiv)
        return float(val), float(err)
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def convergence_series(expr_str: str, a: float, b: float, method_name: str, n_max: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    x_sym = sp.Symbol("x", real=True)
    locals_map = {"x": x_sym, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                  "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi}
    expr = sp.sympify(expr_str, locals=locals_map)
    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])

    ref, _ = compute_reference_quad(expr_str, a, b)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    ns = np.arange(10, n_max + 1, step, dtype=int)
    fn = METHODS[method_name]

    errs = []
    for nn in ns:
        v = fn(f_num, a, b, int(nn))
        errs.append(abs(ref - v))
    return ns, np.array(errs, dtype=float)


# ----------------------------
# 5) PARSER
# ----------------------------
@st.cache_data(show_spinner=False)
def parse_function(expr_str: str) -> ParsedFunction:
    x_sym = sp.Symbol("x", real=True)
    locals_map = {
        "x": x_sym,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    expr = sp.sympify(expr_str, locals=locals_map)
    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])
    return ParsedFunction(expr=expr, f_num=f_num)


# ----------------------------
# 6) THEORY PANEL (PRO LOOK)
# ----------------------------
def theory_panel():
    st.markdown("## 📘 Mathematical Foundations")
    st.markdown(
        "<span class='badge'>Quadrature</span> "
        "<span class='badge'>Error Orders</span> "
        "<span class='badge'>Log-Log Diagnostics</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Open full theory (methods, examples, error & convergence)", expanded=False):
        st.markdown("""
This engine approximates the definite integral

""")
        st.latex(r"\displaystyle I = \int_a^b f(x)\,dx")

        st.markdown("""
by discretizing the interval into **n** subintervals of width

""")
        st.latex(r"\displaystyle h = \Delta x = \frac{b-a}{n}")

        st.markdown("""
and replacing the function locally by a low-degree approximation (constant, linear, quadratic).

---

### 1) Riemann Sums (Left / Right)
**Model:** piecewise-constant approximation.

""")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f(x_i)\,h")

        st.markdown("""
- Left rule uses \(x_i=a+ih\); Right rule uses \(x_i=a+(i+1)h\).
- If \(f\) is smooth, global error scales like:

""")
        st.latex(r"\displaystyle \text{Error} = O(h)")

        st.markdown("""
**Practical bias:**  
- Increasing \(f\) ⇒ Left underestimates, Right overestimates.  
- Decreasing \(f\) ⇒ Left overestimates, Right underestimates.

**Example:**  
\(\int_0^1 x^2 dx = 1/3\). With small n, left rectangles are systematically low because \(x^2\) increases.

---

### 2) Midpoint Rule
**Model:** piecewise-constant at midpoints (cancels first-order term).

""")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f\!\left(a+\left(i+\tfrac{1}{2}\right)h\right) h")

        st.markdown("""
For smooth \(f\), the symmetry around each midpoint cancels the linear term in the Taylor expansion, leading to:

""")
        st.latex(r"\displaystyle \text{Error} = O(h^2)")

        st.markdown("""
So halving \(h\) typically reduces error by a factor of ~4.

---

### 3) Trapezoidal Rule
**Model:** piecewise-linear interpolation.

""")
        st.latex(r"\displaystyle I \approx \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]")

        st.markdown("""
Smooth case error behaves like:

""")
        st.latex(r"\displaystyle \text{Error} = -\frac{(b-a)}{12} h^2 f''(\xi) \quad \Rightarrow \quad O(h^2)")

        st.markdown("""
This is a classic workhorse in applied settings due to stability and simplicity.

---

### 4) Simpson’s Rule
**Model:** piecewise-quadratic (parabolic) interpolation. Requires **even n**.

""")
        st.latex(r"\displaystyle I \approx \frac{h}{3}\left[f(x_0) + 4\sum_{\text{odd}} f(x_i) + 2\sum_{\text{even}} f(x_i) + f(x_n)\right]")

        st.markdown("""
Smooth case error behaves like:

""")
        st.latex(r"\displaystyle \text{Error} = -\frac{(b-a)}{180} h^4 f^{(4)}(\xi) \quad \Rightarrow \quad O(h^4)")

        st.markdown("""
So halving \(h\) can reduce error by ~16 for sufficiently smooth functions.

---

### 5) Log-Log Convergence Diagnostics
If

""")
        st.latex(r"\displaystyle \text{Error} \approx C h^p")

        st.markdown("""
then taking logs yields a line:

""")
        st.latex(r"\displaystyle \log(\text{Error}) \approx p\log(h) + \log(C)")

        st.markdown("""
The slope \(p\) is the **observed order of convergence**, which this engine estimates numerically.
""")


# ----------------------------
# 7) PLOTS
# ----------------------------
def make_main_plot(expr: sp.Expr, f_num, a: float, b: float, n: int, show_rectangles: bool) -> go.Figure:
    h = (b - a) / n
    x_curve = build_xcurve(a, b)
    y_curve = safe_eval_curve(f_num, x_curve)

    fig = go.Figure()

    # fill only on [a,b]
    mask = (x_curve >= a) & (x_curve <= b)
    fig.add_trace(go.Scatter(
        x=x_curve[mask], y=y_curve[mask],
        fill="tozeroy",
        name="Area (visual)",
        fillcolor="rgba(255, 75, 75, 0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
    ))

    # glow
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve,
        mode="lines",
        line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    # main curve
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
            name="Partition bars",
            marker=dict(color="#1E90FF", opacity=0.55, line=dict(color="rgba(255,255,255,0.35)", width=0.5)),
            hovertemplate="x=%{x:.6f}<br>height=%{y:.6f}<extra></extra>",
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
    # Fit log(err) = alpha + p log(h), where h = (b-a)/n
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

    # linear regression slope
    p = np.polyfit(x, y, 1)[0]
    return float(p)


def make_convergence_plot(expr_str: str, a: float, b: float, method_name: str, n_max: int, step: int, loglog: bool) -> Tuple[Optional[go.Figure], Optional[float], np.ndarray, np.ndarray]:
    ns, errs = convergence_series(expr_str, a, b, method_name, n_max, step)
    if ns.size == 0:
        return None, None, ns, errs

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=errs,
        mode="lines",
        name=f"Error ({method_name})",
        hovertemplate="n=%{x}<br>err=%{y:.3e}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title="Absolute Error vs n",
        xaxis_title="n (partitions)",
        yaxis_title="Absolute Error",
        transition=dict(duration=450),
    )
    if loglog:
        fig.update_layout(xaxis_type="log", yaxis_type="log")

    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    p_obs = estimate_observed_order(ns, errs, a, b)
    return fig, p_obs, ns, errs


# ----------------------------
# 8) HEADER
# ----------------------------
st.title("📈 Quant Integration Engine")
st.caption("Numerical Methods • Convergence Diagnostics • Precision & Profiling")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# Optional: top theory panel
theory_panel()

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ----------------------------
# 9) SIDEBAR
# ----------------------------
st.sidebar.header("Controls")

examples = {
    "Smooth (default)": "x**2 * sin(x)",
    "Oscillatory": "sin(50*x) / (1 + x**2)",
    "Non-smooth": "Abs(x)",
    "Discontinuous": "1/(x) ",
    "Exponential": "exp(-x**2)",
}
example_pick = st.sidebar.selectbox("Quick examples", list(examples.keys()), index=0)
default_expr = examples[example_pick]

expr_str = st.sidebar.text_input("f(x) (SymPy syntax)", value=default_expr)

colA, colB = st.sidebar.columns(2)
a = colA.number_input("a", value=-2.0, format="%.6f")
b = colB.number_input("b", value=2.0, format="%.6f")

if a == b:
    st.error("a and b cannot be equal.")
    st.stop()
if a > b:
    st.sidebar.warning("Swapping bounds because a > b.")
    a, b = b, a

method_name = st.sidebar.selectbox("Primary method", list(METHODS.keys()), index=2)

n = st.sidebar.slider("Refinement (n partitions)", 10, 4000, 400, step=10)
show_rectangles = st.sidebar.checkbox("Show partition bars", value=True)

st.sidebar.markdown("---")
show_conv = st.sidebar.checkbox("Show convergence diagnostics", value=True)
n_max = st.sidebar.slider("Convergence max n", 200, 12000, 4000, step=100)
step = st.sidebar.slider("Convergence step", 10, 400, 40, step=10)
loglog = st.sidebar.checkbox("Log-log view", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Note: discontinuities may break quad reference and distort convergence.")

# ----------------------------
# 10) PARSE FUNCTION
# ----------------------------
try:
    parsed = parse_function(expr_str)
    expr = parsed.expr
    f_num = parsed.f_num

    # quick vector sanity
    test = f_num(np.array([a, (a + b) / 2, b], dtype=float))
    test = np.array(test, dtype=float)
except Exception as e:
    st.error(f"Invalid function. Parsing/evaluation failed: {e}")
    st.stop()

# ----------------------------
# 11) REFERENCE (quad)
# ----------------------------
ref_val, ref_err = compute_reference_quad(expr_str, a, b)

# ----------------------------
# 12) COMPUTE CURRENT n RESULTS
# ----------------------------
h = (b - a) / n

rows = []
for name, fn in METHODS.items():
    t0 = time.time()
    val = fn(f_num, a, b, n)
    t1 = time.time()
    err = abs(ref_val - val) if ref_val is not None else np.nan
    rows.append([name, val, err, t1 - t0, THEORETICAL_ORDER.get(name, np.nan)])

df = pd.DataFrame(rows, columns=["Method", "Approximation", "Abs Error (vs quad)", "Time (s)", "Theoretical Order p"])
df_sorted = df.sort_values(by=["Abs Error (vs quad)"], ascending=True, na_position="last")

primary_val = float(df[df["Method"] == method_name]["Approximation"].iloc[0])

# ----------------------------
# 13) STATUS STRIP (METRICS)
# ----------------------------
m1, m2, m3, m4, m5 = st.columns([1.2, 1.1, 1.1, 1.0, 1.0])

m1.metric("Primary Approx", f"{primary_val:.8f}", f"h = {h:.6g}")
if ref_val is not None:
    m2.metric("SciPy quad", f"{ref_val:.8f}", f"± {ref_err:.2e}")
    m3.metric("Abs Error", f"{abs(ref_val - primary_val):.6e}", delta_color="inverse")
else:
    m2.metric("SciPy quad", "n/a", "reference unavailable")
    m3.metric("Abs Error", "n/a", "—")

m4.metric("n partitions", f"{n}", "runtime scales ~O(n)")
m5.metric("Method order p", f"{THEORETICAL_ORDER.get(method_name,'—')}", "theoretical")

st.markdown("<div class='small-muted'>Diagnostics table (computed at the current n):</div>", unsafe_allow_html=True)
st.dataframe(df_sorted, use_container_width=True, hide_index=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ----------------------------
# 14) MAIN PLOT
# ----------------------------
tab_engine, tab_diag = st.tabs(["Engine View", "Diagnostics"])

with tab_engine:
    fig = make_main_plot(expr, f_num, a, b, n, show_rectangles)
    st.plotly_chart(fig, use_container_width=True)

with tab_diag:
    if show_conv and ref_val is not None:
        t0c = time.time()
        fig_c, p_obs, ns, errs = make_convergence_plot(expr_str, a, b, method_name, n_max, step, loglog)
        t1c = time.time()

        if fig_c is not None:
            st.plotly_chart(fig_c, use_container_width=True)

            # Observed order report
            theo = THEORETICAL_ORDER.get(method_name, None)
            c1, c2, c3 = st.columns(3)
            if p_obs is not None:
                c1.metric("Observed order (slope)", f"{p_obs:.3f}", "from log(err) vs log(h)")
            else:
                c1.metric("Observed order (slope)", "n/a", "insufficient data")

            c2.metric("Theoretical order", f"{theo}" if theo is not None else "—", "expected")
            c3.metric("Diagnostics runtime", f"{(t1c - t0c):.3f}s", f"{len(ns)} points")

            # Optional small dataframe
            small = pd.DataFrame({"n": ns, "abs_error": errs})
            st.markdown("<div class='small-muted'>Raw convergence samples:</div>", unsafe_allow_html=True)
            st.dataframe(small, use_container_width=True, hide_index=True)
        else:
            st.info("Unable to compute convergence plot.")
    else:
        st.info("Convergence diagnostics requires a valid quad reference. Try a smoother function or different interval.")

# ----------------------------
# 15) FOOTER
# ----------------------------
st.markdown("<div class='footer'>Unconventional Analysis Group • Quantitative Research Division</div>", unsafe_allow_html=True)
