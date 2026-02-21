# streamlit_app.py
# Quant-style Numerical Integration Engine (Streamlit + Plotly)
# ------------------------------------------------------------
# Features:
# - Safe-ish parsing with sympy
# - Multiple deterministic methods: Left/Right/Midpoint/Trapezoid/Simpson
# - SciPy quad as reference (when possible)
# - Convergence diagnostics (linear + log-log)
# - Smooth plot transitions + glow effect
# - Performance timing

import time
import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import quad

st.markdown("## 📘 Mathematical Foundations of Numerical Integration")

with st.expander("Full theoretical background — methods, derivations, examples and error analysis", expanded=False):

    st.markdown("""
Numerical integration (quadrature) aims to approximate a definite integral

""")

    st.latex(r"\int_a^b f(x)\,dx")

    st.markdown("""
when a closed-form antiderivative is unavailable or computationally impractical.

All classical deterministic methods rely on the same structural idea:

> Partition the interval [a, b] into n subintervals of width  
""")

    st.latex(r"\Delta x = \frac{b-a}{n}")

    st.markdown("""
and approximate the function locally by a simpler polynomial.

The quality of the approximation depends on:
- Smoothness of the function
- Polynomial degree used locally
- Step size h = Δx
""")

    st.markdown("---")

    st.markdown("## 1️⃣ Riemann Sums")

    st.markdown("### Left / Right Riemann")

    st.latex(r"""
    \int_a^b f(x)\,dx \approx \sum_{i=0}^{n-1} f(x_i)\,\Delta x
    """)

    st.markdown("""
Interpretation:
- Each subinterval contributes a rectangle.
- Height determined by endpoint evaluation.
- First-order method.

### Error analysis

If f is continuously differentiable:

""")

    st.latex(r"""
    \text{Error} = O(h)
    """)

    st.markdown("""
This means:

If h is divided by 2 → error is approximately divided by 2.

### Example

Let:

""")

    st.latex(r"\int_0^1 x^2 dx = \frac{1}{3}")

    st.markdown("""
Using n=10 (h=0.1), Riemann Left underestimates because x² is increasing.

This bias is systematic:
- Increasing function → Left underestimates
- Decreasing function → Left overestimates
""")

    st.markdown("---")

    st.markdown("## 2️⃣ Midpoint Rule")

    st.latex(r"""
    \int_a^b f(x)\,dx \approx 
    \sum_{i=0}^{n-1} f\left(x_i + \frac{h}{2}\right)h
    """)

    st.markdown("""
Why is this better?

Because the first-order Taylor error cancels.

Expanding f around midpoint:

""")

    st.latex(r"""
    f(x) = f(m_i) + f'(m_i)(x-m_i) + \frac{f''(\xi)}{2}(x-m_i)^2
    """)

    st.markdown("""
The linear term integrates to zero over symmetric interval.

Thus:

""")

    st.latex(r"""
    \text{Error} = O(h^2)
    """)

    st.markdown("""
This is quadratic convergence.

If h → h/2  
Error → roughly 1/4.
""")

    st.markdown("---")

    st.markdown("## 3️⃣ Trapezoidal Rule")

    st.latex(r"""
    \int_a^b f(x)\,dx \approx 
    \frac{h}{2}\left[f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n)\right]
    """)

    st.markdown("""
Interpretation:
- Linear interpolation between adjacent points.
- Approximates curve with straight lines.

Error term:

""")

    st.latex(r"""
    \text{Error} = -\frac{(b-a)}{12} h^2 f''(\xi)
    """)

    st.markdown("""
Also O(h²).

In finance:
- Used in yield curve integration
- Used in volatility surface approximations
""")

    st.markdown("---")

    st.markdown("## 4️⃣ Simpson’s Rule")

    st.markdown("Polynomial degree: 2 (quadratic interpolation)")

    st.latex(r"""
    \int_a^b f(x)\,dx \approx 
    \frac{h}{3}
    \left[
    f(x_0) + 4\sum_{odd} f(x_i)
    + 2\sum_{even} f(x_i)
    + f(x_n)
    \right]
    """)

    st.markdown("""
Derived from integrating Lagrange quadratic polynomial.

Error term:

""")

    st.latex(r"""
    \text{Error} = -\frac{(b-a)}{180} h^4 f^{(4)}(\xi)
    """)

    st.markdown("""
Fourth-order convergence.

If h → h/2  
Error → approximately divided by 16.

For smooth functions, extremely accurate.
""")

    st.markdown("---")

    st.markdown("## 5️⃣ Monte Carlo Integration")

    st.latex(r"""
    \int_a^b f(x)\,dx \approx 
    (b-a)\frac{1}{N}\sum_{i=1}^N f(X_i)
    """)

    st.markdown("""
Where X_i ~ Uniform(a,b).

Key property:

""")

    st.latex(r"""
    \text{Std Error} \sim \frac{\sigma}{\sqrt{N}}
    """)

    st.markdown("""
Convergence rate:

""")

    st.latex(r"O(N^{-1/2})")

    st.markdown("""
Slower in 1D than Simpson,
but dimension-independent.

This makes Monte Carlo dominant in high-dimensional problems:
- Option pricing
- Risk simulation
- Stochastic differential equations
""")

    st.markdown("---")

    st.markdown("## 📊 Convergence and Log-Log Analysis")

    st.markdown("""
If error behaves as:
""")

    st.latex(r"""
    \text{Error} = C h^p
    """)

    st.markdown("""
Taking logarithms:
""")

    st.latex(r"""
    \log(\text{Error}) = p\log(h) + \log(C)
    """)

    st.markdown("""
Thus in a log-log plot:

Slope = p (order of convergence)

Expected slopes:
- Riemann: 1
- Midpoint: 2
- Trapezoidal: 2
- Simpson: 4
- Monte Carlo: 1/2 (in N scale)

This is why convergence diagnostics in this engine are mathematically meaningful.
""")

    st.markdown("---")

    st.markdown("## 🧠 Practical Interpretation")

    st.markdown("""
Choosing a method depends on:

- Smoothness of function
- Dimensionality
- Required precision
- Computational budget

In quantitative finance:

- Low dimension & smooth payoff → Simpson / adaptive quadrature
- High dimension → Monte Carlo
- Real-time systems → Trapezoidal / vectorized midpoint

This engine allows empirical verification of theoretical convergence rates.
""")

# ----------------------------
# 1) PAGE / THEME
# ----------------------------
st.set_page_config(page_title="Quant Integration Engine", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --panel: #111827;
  --panel2: #0b1220;
  --muted: #9aa4b2;
  --text: #e5e7eb;
  --accent: #FF4B4B;
  --accent2: #1E90FF;
  --good: #22c55e;
  --warn: #f59e0b;
}

.main { background-color: var(--bg); }
section[data-testid="stSidebar"] { background-color: #0b1020; border-right: 1px solid rgba(255,255,255,0.06); }
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px;
}
.small-muted { color: var(--muted); font-size: 0.9rem; }
.footer { text-align:center; color: rgba(229,231,235,0.35); margin-top: 12px; font-size: 0.85rem; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.6rem 0 1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Quant Integration Engine")
st.caption("Numerical Methods • Convergence Diagnostics • Precision & Profiling")
st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# 2) NUMERICAL METHODS
# ----------------------------
def riemann_left(f, a, b, n):
    dx = (b - a) / n
    x = np.linspace(a, b - dx, n)
    return np.sum(f(x)) * dx

def riemann_right(f, a, b, n):
    dx = (b - a) / n
    x = np.linspace(a + dx, b, n)
    return np.sum(f(x)) * dx

def riemann_midpoint(f, a, b, n):
    dx = (b - a) / n
    x = np.linspace(a + dx / 2, b - dx / 2, n)
    return np.sum(f(x)) * dx

def trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    return dx * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def simpson(f, a, b, n):
    # Simpson requires even n
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    dx = (b - a) / n
    return (dx / 3) * (
        y[0]
        + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-2:2])
    )

METHODS = {
    "Riemann Left": riemann_left,
    "Riemann Right": riemann_right,
    "Midpoint": riemann_midpoint,
    "Trapezoidal": trapezoidal,
    "Simpson": simpson,
}

# ----------------------------
# 3) SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Model Controls")

funcao_input = st.sidebar.text_input("f(x) (SymPy syntax)", value="x**2 * sin(x)")
colA, colB = st.sidebar.columns(2)
a = colA.number_input("a", value=-2.0, format="%.6f")
b = colB.number_input("b", value=2.0, format="%.6f")

if a == b:
    st.error("a and b cannot be equal.")
    st.stop()

if a > b:
    st.sidebar.warning("Swapping bounds because a > b.")
    a, b = b, a

method_choice = st.sidebar.selectbox("Primary method", list(METHODS.keys()), index=2)

n = st.sidebar.slider("Refinement (n partitions)", min_value=10, max_value=2000, value=200, step=10)
dx = (b - a) / n

show_rectangles = st.sidebar.checkbox("Show Riemann bars (visual partition)", value=True)
show_convergence = st.sidebar.checkbox("Show convergence diagnostics", value=True)

conv_max_n = st.sidebar.slider("Convergence max n", 200, 8000, 3000, step=100)
conv_step = st.sidebar.slider("Convergence step", 10, 200, 30, step=10)

loglog = st.sidebar.checkbox("Log-log convergence view", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: try non-smooth functions too (abs(x), sign(x), etc.).")

# ----------------------------
# 4) SAFE PARSING + NUMERIC FUNCTION
# ----------------------------
try:
    x_sym = sp.Symbol("x", real=True)
    # "locals" to reduce surprises / allow common functions
    locals_map = {
        "x": x_sym,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "pi": sp.pi,
    }

    expr = sp.sympify(funcao_input, locals=locals_map)
    f_num = sp.lambdify(x_sym, expr, modules=["numpy"])

    # quick sanity test
    _ = f_num(np.array([a, (a + b) / 2, b], dtype=float))

except Exception as e:
    st.error(f"Invalid mathematical expression: {e}")
    st.stop()

# ----------------------------
# 5) REFERENCE INTEGRAL (SCIPY)
# ----------------------------
res_real = None
quad_err = None
try:
    # quad is robust but can fail for nasty expressions or discontinuities
    res_real, quad_err = quad(lambda t: float(f_num(t)), a, b, limit=200)
except Exception:
    # We'll still run; just won't display SciPy reference
    res_real, quad_err = None, None

# ----------------------------
# 6) COMPUTE PRIMARY METHOD + TABLE
# ----------------------------
t0 = time.time()
approx_primary = METHODS[method_choice](f_num, a, b, n)
t1 = time.time()
elapsed_primary = t1 - t0

rows = []
for name, fn in METHODS.items():
    t0m = time.time()
    val = fn(f_num, a, b, n)
    t1m = time.time()
    if res_real is not None:
        err = abs(res_real - val)
    else:
        err = np.nan
    rows.append([name, val, err, t1m - t0m])

df = pd.DataFrame(rows, columns=["Method", "Approximation", "Abs Error (vs quad)", "Time (s)"])
df_sorted = df.sort_values(by=["Abs Error (vs quad)"], ascending=True, na_position="last")

# ----------------------------
# 7) TOP METRICS
# ----------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Primary Approx", f"{approx_primary:.6f}", f"dx = {dx:.6g}")
if res_real is not None:
    m2.metric("SciPy quad", f"{res_real:.6f}", f"± {quad_err:.2e}")
    m3.metric("Abs Error", f"{abs(res_real - approx_primary):.6e}", delta_color="inverse")
else:
    m2.metric("SciPy quad", "n/a", "reference unavailable")
    m3.metric("Abs Error", "n/a", "—")

m4.metric("Execution Time", f"{elapsed_primary:.6f}s", f"n={n}")

st.markdown("<div class='small-muted'>Diagnostics table (methods at the current n):</div>", unsafe_allow_html=True)
st.dataframe(df_sorted, use_container_width=True, hide_index=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# 8) PLOT: FUNCTION + BARS + GLOW
# ----------------------------
# choose a nice view range beyond [a,b]
pad = 0.15 * (b - a)
x_min = a - pad
x_max = b + pad
x_curve = np.linspace(x_min, x_max, 1200)

# Evaluate function carefully (avoid crashing on vector errors)
try:
    y_curve = f_num(x_curve).astype(float)
except Exception:
    # fallback pointwise
    y_curve = np.array([float(f_num(xx)) for xx in x_curve], dtype=float)

fig = go.Figure()

# soft filled area for [a,b] only
mask_ab = (x_curve >= a) & (x_curve <= b)
fig.add_trace(
    go.Scatter(
        x=x_curve[mask_ab],
        y=y_curve[mask_ab],
        fill="tozeroy",
        name="Area (visual)",
        fillcolor="rgba(255, 75, 75, 0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
    )
)

# glow layer
fig.add_trace(
    go.Scatter(
        x=x_curve,
        y=y_curve,
        mode="lines",
        name="Glow",
        line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip",
        showlegend=False,
    )
)

# main curve
fig.add_trace(
    go.Scatter(
        x=x_curve,
        y=y_curve,
        mode="lines",
        name="f(x)",
        line=dict(color="#FF4B4B", width=3),
        hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>",
    )
)

# bars (left partition for visuals, regardless of method)
if show_rectangles:
    x_left = np.linspace(a, b - dx, n)
    y_left = f_num(x_left)
    fig.add_trace(
        go.Bar(
            x=x_left,
            y=y_left,
            width=dx,
            offset=0,
            name="Partition bars",
            marker=dict(color="#1E90FF", opacity=0.55, line=dict(color="rgba(255,255,255,0.35)", width=0.5)),
            hovertemplate="x=%{x:.6f}<br>height=%{y:.6f}<extra></extra>",
        )
    )

# reference lines
fig.add_vline(x=a, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")
fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")

fig.update_layout(
    template="plotly_dark",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    transition=dict(duration=500),
    title=f"f(x) = {sp.sstr(expr)}   |   [{a:.4g}, {b:.4g}]   |   n={n}",
)

fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 9) CONVERGENCE DIAGNOSTICS
# ----------------------------
if show_convergence and (res_real is not None):
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.subheader("Convergence Diagnostics")

    ns = np.arange(10, conv_max_n + 1, conv_step, dtype=int)
    errs = []
    vals = []

    # Use midpoint as a stable "default" for convergence, or use selected method
    conv_method = METHODS[method_choice]

    t0c = time.time()
    for nn in ns:
        v = conv_method(f_num, a, b, int(nn))
        vals.append(v)
        errs.append(abs(res_real - v))
    t1c = time.time()

    fig_c = go.Figure()
    fig_c.add_trace(
        go.Scatter(
            x=ns,
            y=errs,
            mode="lines",
            name=f"Error ({method_choice})",
            hovertemplate="n=%{x}<br>err=%{y:.3e}<extra></extra>",
        )
    )

    fig_c.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Error vs n  (computed in {t1c - t0c:.3f}s)",
        xaxis_title="n (partitions)",
        yaxis_title="Absolute Error",
        transition=dict(duration=500),
    )

    if loglog:
        fig_c.update_layout(xaxis_type="log", yaxis_type="log")

    fig_c.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig_c.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

    st.plotly_chart(fig_c, use_container_width=True)

elif show_convergence and (res_real is None):
    st.info("Convergence diagnostics needs the SciPy quad reference. Your function may be discontinuous or hard for quad here.")

# ----------------------------
# 10) FOOTER
# ----------------------------
st.markdown("<div class='footer'>Unconventional Analysis Group • Quantitative Research Division</div>", unsafe_allow_html=True)


