# streamlit_app.py
# Quant Integration Engine — 1D + 2D (Double Integral) + 3D Surface Viewer
# -----------------------------------------------------------------------
# Fixes included:
# - Quick examples now sync correctly (selectbox -> text_input) via session_state + st.rerun()
# - NumPy new versions: np.trapz -> np.trapezoid
# - 2D Engine: Heatmap + Interactive 3D Surface (zoom/rotate/pan)
# - Log-log convergence safe (no vanish on zeros) via EPS_LOG
# - Cache correctness: sympy lambdify cached with st.cache_resource (picklable issue fixed)

from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st
from scipy.integrate import quad, dblquad


# ----------------------------
# 0) PAGE CONFIG (MUST BE FIRST)
# ----------------------------
st.set_page_config(page_title="Quant Integration Engine", layout="wide")

EPS_LOG = 1e-300


# ----------------------------
# 1) STYLE
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
# 2) SMALL UTILS
# ----------------------------
def _sanitize_array(y) -> np.ndarray:
    y = np.array(y, dtype=float)
    y[~np.isfinite(y)] = np.nan
    return y


# ----------------------------
# 3) 1D METHODS
# ----------------------------
def riemann_left_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a, b - h, n)
    return float(np.sum(f(x)) * h)


def riemann_right_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h, b, n)
    return float(np.sum(f(x)) * h)


def midpoint_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    return float(np.sum(f(x)) * h)


def trapezoidal_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float(h * (np.sum(y) - 0.5 * (y[0] + y[-1])))


def simpson_1d(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return float((h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2])))


METHODS_1D: Dict[str, Callable[[Callable[[np.ndarray], np.ndarray], float, float, int], float]] = {
    "Riemann Left": riemann_left_1d,
    "Riemann Right": riemann_right_1d,
    "Midpoint": midpoint_1d,
    "Trapezoidal": trapezoidal_1d,
    "Simpson": simpson_1d,
}

THEORETICAL_ORDER_1D = {
    "Riemann Left": 1,
    "Riemann Right": 1,
    "Midpoint": 2,
    "Trapezoidal": 2,
    "Simpson": 4,
}


# ----------------------------
# 4) 2D METHODS (DOUBLE INTEGRAL over rectangle)
# ----------------------------
def midpoint_2d(fxy: Callable[[np.ndarray, np.ndarray], np.ndarray],
                ax: float, bx: float, ay: float, by: float,
                nx: int, ny: int) -> float:
    hx = (bx - ax) / nx
    hy = (by - ay) / ny

    x = np.linspace(ax + hx / 2, bx - hx / 2, nx)
    y = np.linspace(ay + hy / 2, by - hy / 2, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = _sanitize_array(fxy(X, Y))
    return float(np.nansum(Z) * hx * hy)


def trapezoidal_2d(fxy: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   ax: float, bx: float, ay: float, by: float,
                   nx: int, ny: int) -> float:
    # product trapezoid: integrate along y then x
    x = np.linspace(ax, bx, nx + 1)
    y = np.linspace(ay, by, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = _sanitize_array(fxy(X, Y))

    # ✅ NumPy new versions: trapz removed -> trapezoid
    Iy = np.trapezoid(Z, y, axis=0)
    Ixy = np.trapezoid(Iy, x, axis=0)
    return float(Ixy)


def simpson_2d(fxy: Callable[[np.ndarray, np.ndarray], np.ndarray],
               ax: float, bx: float, ay: float, by: float,
               nx: int, ny: int) -> float:
    # Simpson requires even nx, ny
    if nx % 2 == 1:
        nx += 1
    if ny % 2 == 1:
        ny += 1

    hx = (bx - ax) / nx
    hy = (by - ay) / ny

    x = np.linspace(ax, bx, nx + 1)
    y = np.linspace(ay, by, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = _sanitize_array(fxy(X, Y))

    wx = np.ones(nx + 1)
    wx[1:-1:2] = 4
    wx[2:-2:2] = 2

    wy = np.ones(ny + 1)
    wy[1:-1:2] = 4
    wy[2:-2:2] = 2

    W = np.outer(wy, wx)
    val = np.nansum(W * Z) * (hx * hy / 9.0)
    return float(val)


def monte_carlo_2d(fxy: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   ax: float, bx: float, ay: float, by: float,
                   n_samples: int, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(ax, bx, size=n_samples)
    ys = rng.uniform(ay, by, size=n_samples)
    vals = _sanitize_array(fxy(xs, ys))

    mean = np.nanmean(vals)
    std = np.nanstd(vals, ddof=1) if n_samples > 1 else 0.0

    area = (bx - ax) * (by - ay)
    estimate = area * mean
    stderr = area * (std / np.sqrt(max(n_samples, 1)))
    return float(estimate), float(stderr)


METHODS_2D: Dict[str, str] = {
    "Midpoint (Product)": "midpoint",
    "Trapezoidal (Product)": "trapezoid",
    "Simpson (Product)": "simpson",
    "Monte Carlo (2D)": "mc",
}


# ----------------------------
# 5) PARSERS (CACHE RESOURCE)
# ----------------------------
@st.cache_resource(show_spinner=False)
def parse_function_1d(expr_str: str):
    x = sp.Symbol("x", real=True)
    locals_map = {
        "x": x,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    expr = sp.sympify(expr_str, locals=locals_map)
    f_num = sp.lambdify(x, expr, modules=["numpy"])
    return expr, f_num


@st.cache_resource(show_spinner=False)
def parse_function_2d(expr_str: str):
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    locals_map = {
        "x": x, "y": y,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi,
    }
    expr = sp.sympify(expr_str, locals=locals_map)
    f_xy = sp.lambdify((x, y), expr, modules=["numpy"])
    return expr, f_xy


# ----------------------------
# 6) 1D CACHE DATA
# ----------------------------
@st.cache_data(show_spinner=False)
def build_xcurve_1d(a: float, b: float, points: int = 1400) -> np.ndarray:
    pad = 0.15 * (b - a)
    return np.linspace(a - pad, b + pad, points)


@st.cache_data(show_spinner=False)
def eval_curve_1d(expr_str: str, a: float, b: float, points: int = 1400) -> Tuple[np.ndarray, np.ndarray]:
    _, f_num = parse_function_1d(expr_str)
    x_curve = build_xcurve_1d(a, b, points)
    try:
        y = f_num(x_curve)
    except Exception:
        y = np.array([float(f_num(xx)) for xx in x_curve], dtype=float)
    y = _sanitize_array(y)
    return x_curve, y


@st.cache_data(show_spinner=False)
def reference_quad_1d(expr_str: str, a: float, b: float) -> Tuple[Optional[float], Optional[float]]:
    try:
        _, f_num = parse_function_1d(expr_str)
        val, err = quad(lambda t: float(f_num(t)), a, b, limit=200)
        return float(val), float(err)
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def convergence_series_1d(expr_str: str, a: float, b: float, method_name: str, n_max: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    _, f_num = parse_function_1d(expr_str)
    ref, _ = reference_quad_1d(expr_str, a, b)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    ns = np.arange(10, n_max + 1, step, dtype=int)
    fn = METHODS_1D[method_name]

    errs = []
    for nn in ns:
        v = fn(f_num, a, b, int(nn))
        errs.append(abs(ref - v))
    return ns, np.array(errs, dtype=float)


# ----------------------------
# 7) 2D CACHE DATA
# ----------------------------
@st.cache_data(show_spinner=False)
def reference_dblquad_2d(expr_str: str, ax: float, bx: float, ay: float, by: float) -> Tuple[Optional[float], Optional[float]]:
    try:
        _, f_xy = parse_function_2d(expr_str)

        def integrand(y, x):
            return float(f_xy(x, y))

        val, err = dblquad(integrand, ax, bx, lambda _: ay, lambda _: by)
        return float(val), float(err)
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def surface_grid_2d(expr_str: str, ax: float, bx: float, ay: float, by: float, gx: int, gy: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, f_xy = parse_function_2d(expr_str)
    xs = np.linspace(ax, bx, gx)
    ys = np.linspace(ay, by, gy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = _sanitize_array(f_xy(X, Y))
    return xs, ys, Z


@st.cache_data(show_spinner=False)
def convergence_series_2d(expr_str: str, ax: float, bx: float, ay: float, by: float,
                          method_key: str, n_max: int, step: int, mc_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    ref, _ = reference_dblquad_2d(expr_str, ax, bx, ay, by)
    if ref is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    _, f_xy = parse_function_2d(expr_str)

    ns = np.arange(max(10, step), n_max + 1, step, dtype=int)
    errs = []

    for n in ns:
        n = int(n)
        if method_key == "midpoint":
            val = midpoint_2d(f_xy, ax, bx, ay, by, n, n)
        elif method_key == "trapezoid":
            val = trapezoidal_2d(f_xy, ax, bx, ay, by, n, n)
        elif method_key == "simpson":
            val = simpson_2d(f_xy, ax, bx, ay, by, n, n)
        else:
            samples = int(max(500, n * n))
            val, _se = monte_carlo_2d(f_xy, ax, bx, ay, by, samples, seed=int(mc_seed))

        errs.append(abs(ref - val))

    return ns, np.array(errs, dtype=float)


# ----------------------------
# 8) ORDER ESTIMATION (LOG-LOG)
# ----------------------------
def estimate_order_from_h(h: np.ndarray, errs: np.ndarray) -> Optional[float]:
    if len(h) < 6:
        return None
    x = np.log(np.maximum(h, EPS_LOG))
    y = np.log(np.maximum(errs, EPS_LOG))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 6:
        return None
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def estimate_order_1d(ns: np.ndarray, errs: np.ndarray, a: float, b: float) -> Optional[float]:
    h = (b - a) / ns.astype(float)
    return estimate_order_from_h(h, errs)


def estimate_order_2d(ns: np.ndarray, errs: np.ndarray, ax: float, bx: float, ay: float, by: float) -> Optional[float]:
    hx = (bx - ax) / ns.astype(float)
    hy = (by - ay) / ns.astype(float)
    h = np.maximum(hx, hy)
    return estimate_order_from_h(h, errs)


# ----------------------------
# 9) THEORY PANELS
# ----------------------------
def theory_panel_1d():
    st.markdown("## 📘 Mathematical Foundations (1D)")
    st.markdown(
        "<span class='badge'>Quadrature</span> "
        "<span class='badge'>Error Orders</span> "
        "<span class='badge'>Log-Log Diagnostics</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Open full theory (1D methods, formulas, examples & diagnostics)", expanded=False):
        st.markdown("We approximate the definite integral")
        st.latex(r"\displaystyle I=\int_a^b f(x)\,dx")
        st.markdown("with step")
        st.latex(r"\displaystyle h=\frac{b-a}{n}")

        st.markdown("---")
        st.markdown("### Riemann (Left/Right)")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f(x_i)\,h \quad,\quad \text{Error}=O(h)")
        st.markdown("Fast, but first-order. Bias depends on monotonicity.")

        st.markdown("---")
        st.markdown("### Midpoint")
        st.latex(r"\displaystyle I \approx \sum_{i=0}^{n-1} f\!\left(a+(i+\tfrac12)h\right)h \quad,\quad \text{Error}=O(h^2)")

        st.markdown("---")
        st.markdown("### Trapezoidal")
        st.latex(r"\displaystyle I \approx \frac{h}{2}\left[f(x_0)+2\sum_{i=1}^{n-1}f(x_i)+f(x_n)\right]\quad,\quad \text{Error}=O(h^2)")

        st.markdown("---")
        st.markdown("### Simpson")
        st.latex(r"\displaystyle I \approx \frac{h}{3}\left[f(x_0)+4\sum_{\text{odd}}f(x_i)+2\sum_{\text{even}}f(x_i)+f(x_n)\right]\quad,\quad \text{Error}=O(h^4)")

        st.markdown("---")
        st.markdown("### Log-Log Diagnostics")
        st.latex(r"\displaystyle \text{Error}\approx C h^p \Rightarrow \log(\text{Error})\approx p\log(h)+\log(C)")
        st.markdown("Slope \(p\) is the observed convergence order.")


def theory_panel_2d():
    st.markdown("## 📘 Mathematical Foundations (2D / Double Integral)")
    st.markdown(
        "<span class='badge'>Double Integral</span> "
        "<span class='badge'>Product Rules</span> "
        "<span class='badge'>Monte Carlo</span> "
        "<span class='badge'>Convergence</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Open full theory (2D methods, formulas, examples & diagnostics)", expanded=False):
        st.markdown("We approximate the double integral over a rectangular domain:")
        st.latex(r"\displaystyle I=\int_{x=a}^{b}\int_{y=c}^{d} f(x,y)\,dy\,dx")

        st.markdown("Discretize the domain into an \(n_x\times n_y\) mesh:")
        st.latex(r"\displaystyle h_x=\frac{b-a}{n_x}\quad,\quad h_y=\frac{d-c}{n_y}")

        st.markdown("---")
        st.markdown("### Midpoint (Product Rule)")
        st.latex(r"""
\displaystyle
I \approx \sum_{i=0}^{n_x-1}\sum_{j=0}^{n_y-1}
f\!\left(a+\left(i+\tfrac12\right)h_x,\; c+\left(j+\tfrac12\right)h_y\right)\; h_x h_y
""")

        st.markdown("---")
        st.markdown("### Trapezoidal (Product Rule)")
        st.latex(r"\displaystyle I \approx \text{trap}_x(\text{trap}_y(f(x,y)))")
        st.markdown("(Implementation uses NumPy `trapezoid` for compatibility with new versions.)")

        st.markdown("---")
        st.markdown("### Simpson (Product Rule)")
        st.latex(r"""
\displaystyle
I \approx \frac{h_x}{3}\frac{h_y}{3}\sum_{i=0}^{n_x}\sum_{j=0}^{n_y} w_i^{(x)} w_j^{(y)} f(x_i,y_j)
""")

        st.markdown("---")
        st.markdown("### Monte Carlo (2D)")
        st.latex(r"""
\displaystyle
I \approx (b-a)(d-c)\,\frac{1}{N}\sum_{k=1}^N f(X_k, Y_k)
""")
        st.latex(r"\displaystyle \text{Std Error} \sim O(N^{-1/2})")


# ----------------------------
# 10) PLOTS (1D + 2D + 3D)
# ----------------------------
def plot_main_1d(expr_str: str, expr: sp.Expr, f_num, a: float, b: float, n: int, show_rectangles: bool) -> go.Figure:
    h = (b - a) / n
    x_curve, y_curve = eval_curve_1d(expr_str, a, b, points=1400)

    fig = go.Figure()
    mask = (x_curve >= a) & (x_curve <= b)

    fig.add_trace(go.Scatter(
        x=x_curve[mask], y=y_curve[mask],
        fill="tozeroy",
        name="Area (visual)",
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
        y_left = _sanitize_array(f_num(x_left))
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


def plot_conv(ns: np.ndarray, errs: np.ndarray, title: str, loglog: bool) -> go.Figure:
    eplot = np.array(errs, dtype=float)
    if loglog:
        eplot = np.maximum(eplot, EPS_LOG)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=eplot, mode="lines",
        hovertemplate="n=%{x}<br>err=%{y:.3e}<extra></extra>",
        name="Abs Error",
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title=title,
        xaxis_title="n",
        yaxis_title="Absolute Error",
        transition=dict(duration=450),
    )
    if loglog:
        fig.update_layout(xaxis_type="log", yaxis_type="log")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_heatmap_2d(expr_str: str, ax: float, bx: float, ay: float, by: float, gx: int, gy: int, show_grid: bool) -> go.Figure:
    xs, ys, Z = surface_grid_2d(expr_str, ax, bx, ay, by, gx, gy)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z,
        colorbar=dict(title="f(x,y)"),
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>f=%{z:.6f}<extra></extra>",
        showscale=True,
        name="Heatmap",
    ))

    if show_grid:
        px = xs[:: max(1, gx // 30)]
        py = ys[:: max(1, gy // 30)]
        Xp, Yp = np.meshgrid(px, py, indexing="xy")
        fig.add_trace(go.Scatter(
            x=Xp.ravel(), y=Yp.ravel(),
            mode="markers",
            marker=dict(size=3, color="rgba(255,255,255,0.35)"),
            name="Grid samples",
            hoverinfo="skip",
        ))

    fig.add_shape(
        type="rect",
        x0=ax, x1=bx, y0=ay, y1=by,
        line=dict(color="rgba(229,231,235,0.5)", width=1, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title=f"f(x,y) heatmap  |  x∈[{ax:.4g},{bx:.4g}]  y∈[{ay:.4g},{by:.4g}]",
        xaxis_title="x",
        yaxis_title="y",
        transition=dict(duration=450),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_surface_3d(expr_str: str, ax: float, bx: float, ay: float, by: float, gx: int, gy: int) -> go.Figure:
    xs, ys, Z = surface_grid_2d(expr_str, ax, bx, ay, by, gx, gy)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        showscale=True,
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.6f}<extra></extra>",
        contours=dict(z=dict(show=True, usecolormap=True, highlightwidth=2, project=dict(z=True))),
        name="Surface",
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title=f"3D Surface: z = f(x,y)  |  x∈[{ax:.4g},{bx:.4g}]  y∈[{ay:.4g},{by:.4g}]",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z=f(x,y)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.15)),
        ),
    )
    return fig


# ----------------------------
# 11) HEADER
# ----------------------------
st.title("📈 Quant Integration Engine")
st.caption("1D + 2D Numerical Integration • Convergence Diagnostics • Precision & Profiling")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

tab_1d, tab_2d = st.tabs(["1D Integrals", "2D (Double Integral)"])


# ======================================================================
# 1D TAB
# ======================================================================
with tab_1d:
    theory_panel_1d()
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Sidebar controls (1D)
    st.sidebar.header("Controls — 1D")

    examples_1d = {
        "Smooth (default)": "x**2 * sin(x)",
        "Oscillatory": "sin(50*x) / (1 + x**2)",
        "Non-smooth": "Abs(x)",
        "Exponential": "exp(-x**2)",
    }

    # ✅ Sync selectbox -> text_input (1D)
    if "ex1d" not in st.session_state:
        st.session_state.ex1d = "Smooth (default)"
    if "expr_1d" not in st.session_state:
        st.session_state.expr_1d = examples_1d[st.session_state.ex1d]

    def _apply_example_1d():
        st.session_state.expr_1d = examples_1d[st.session_state.ex1d]
        st.rerun()

    st.sidebar.selectbox(
        "Quick examples (1D)",
        list(examples_1d.keys()),
        key="ex1d",
        on_change=_apply_example_1d,
    )

    expr_1d = st.sidebar.text_input(
        "f(x) (SymPy syntax)",
        key="expr_1d",
    )

    colA, colB = st.sidebar.columns(2)
    a = colA.number_input("a", value=-2.0, format="%.6f", key="a_1d")
    b = colB.number_input("b", value=2.0, format="%.6f", key="b_1d")

    if a == b:
        st.error("a and b cannot be equal.")
        st.stop()
    if a > b:
        st.sidebar.warning("Swapping bounds because a > b.")
        a, b = b, a

    method_1d = st.sidebar.selectbox("Primary method (1D)", list(METHODS_1D.keys()), index=2, key="method_1d")
    n = st.sidebar.slider("Refinement (n partitions)", 10, 4000, 400, step=10, key="n_1d")
    show_rect = st.sidebar.checkbox("Show partition bars", value=True, key="bars_1d")

    st.sidebar.markdown("---")
    show_conv_1d = st.sidebar.checkbox("Show convergence diagnostics", value=True, key="conv_1d")
    nmax_1d = st.sidebar.slider("Convergence max n", 200, 12000, 4000, step=100, key="nmax_1d")
    step_1d = st.sidebar.slider("Convergence step", 10, 400, 40, step=10, key="step_1d")
    loglog_1d = st.sidebar.checkbox("Log-log view", value=True, key="loglog_1d")

    st.sidebar.markdown("---")
    st.sidebar.caption("Note: hard discontinuities may break quad and distort convergence.")

    # Parse 1D
    try:
        expr_sym_1d, f1 = parse_function_1d(expr_1d)
        _ = np.array(f1(np.array([a, (a + b) / 2, b], dtype=float)), dtype=float)
    except Exception as e:
        st.error(f"Invalid function. Parsing/evaluation failed: {e}")
        st.stop()

    # Reference
    ref_1d, referr_1d = reference_quad_1d(expr_1d, a, b)

    # Compute methods table at current n
    h = (b - a) / n
    rows = []
    for name, fn in METHODS_1D.items():
        t0 = time.time()
        val = fn(f1, a, b, n)
        t1 = time.time()
        err = abs(ref_1d - val) if ref_1d is not None else np.nan
        rows.append([name, val, err, t1 - t0, THEORETICAL_ORDER_1D.get(name, np.nan)])

    df = pd.DataFrame(rows, columns=["Method", "Approximation", "Abs Error (vs quad)", "Time (s)", "Theoretical Order p"])
    df_sorted = df.sort_values(by=["Abs Error (vs quad)"], ascending=True, na_position="last")

    primary_val = float(df[df["Method"] == method_1d]["Approximation"].iloc[0])

    # Metrics
    m1, m2, m3, m4, m5 = st.columns([1.2, 1.1, 1.1, 1.0, 1.0])
    m1.metric("Primary Approx", f"{primary_val:.8f}", f"h = {h:.6g}")
    if ref_1d is not None:
        m2.metric("SciPy quad", f"{ref_1d:.8f}", f"± {referr_1d:.2e}")
        m3.metric("Abs Error", f"{abs(ref_1d - primary_val):.6e}", delta_color="inverse")
    else:
        m2.metric("SciPy quad", "n/a", "reference unavailable")
        m3.metric("Abs Error", "n/a", "—")
    m4.metric("n partitions", f"{n}", "runtime scales ~O(n)")
    m5.metric("Method order p", f"{THEORETICAL_ORDER_1D.get(method_1d,'—')}", "theoretical")

    st.markdown("<div class='small-muted'>Diagnostics table (computed at the current n):</div>", unsafe_allow_html=True)
    st.dataframe(df_sorted, use_container_width=True, hide_index=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    eng1, diag1 = st.tabs(["Engine View", "Diagnostics"])

    with eng1:
        fig1 = plot_main_1d(expr_1d, expr_sym_1d, f1, a, b, n, show_rect)
        st.plotly_chart(fig1, use_container_width=True)

    with diag1:
        if show_conv_1d and ref_1d is not None:
            t0c = time.time()
            ns, errs = convergence_series_1d(expr_1d, a, b, method_1d, nmax_1d, step_1d)
            t1c = time.time()

            if ns.size > 0:
                p_obs = estimate_order_1d(ns, np.maximum(errs, EPS_LOG), a, b)
                figc = plot_conv(ns, errs, "Absolute Error vs n (1D)", loglog_1d)
                st.plotly_chart(figc, use_container_width=True)

                theo = THEORETICAL_ORDER_1D.get(method_1d, None)
                c1, c2, c3 = st.columns(3)
                c1.metric("Observed order (slope)", f"{p_obs:.3f}" if p_obs is not None else "n/a", "log(err) vs log(h)")
                c2.metric("Theoretical order", f"{theo}" if theo is not None else "—", "expected")
                c3.metric("Diagnostics runtime", f"{(t1c - t0c):.3f}s", f"{len(ns)} points")

                st.markdown("<div class='small-muted'>Raw convergence samples:</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"n": ns, "abs_error": errs}), use_container_width=True, hide_index=True)
            else:
                st.info("Unable to compute convergence series.")
        else:
            st.info("Convergence diagnostics requires a valid quad reference. Try a smoother function or different interval.")

    st.markdown("<div class='footer'>Unconventional Analysis Group • Quantitative Research Division</div>", unsafe_allow_html=True)


# ======================================================================
# 2D TAB
# ======================================================================
with tab_2d:
    theory_panel_2d()
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # Sidebar controls (2D)
    st.sidebar.header("Controls — 2D")

    examples_2d = {
        "Gaussian bump": "exp(-(x**2 + y**2))",
        "Polynomial": "x**2 + y**2",
        "Oscillatory": "sin(10*x)*cos(10*y)",
        "Mixed": "exp(-x**2)*sin(y)",
    }

    # ✅ Sync selectbox -> text_input (2D)
    if "ex2d" not in st.session_state:
        st.session_state.ex2d = "Gaussian bump"
    if "expr_2d" not in st.session_state:
        st.session_state.expr_2d = examples_2d[st.session_state.ex2d]

    def _apply_example_2d():
        st.session_state.expr_2d = examples_2d[st.session_state.ex2d]
        st.rerun()

    st.sidebar.selectbox(
        "Quick examples (2D)",
        list(examples_2d.keys()),
        key="ex2d",
        on_change=_apply_example_2d,
    )

    expr_2d = st.sidebar.text_input(
        "f(x,y) (SymPy syntax)",
        key="expr_2d",
    )

    st.sidebar.markdown("**Domain**  \\([a,b]\\times[c,d]\\)")
    col1, col2 = st.sidebar.columns(2)
    ax = col1.number_input("a (x-min)", value=-2.0, format="%.6f", key="ax")
    bx = col2.number_input("b (x-max)", value=2.0, format="%.6f", key="bx")
    col3, col4 = st.sidebar.columns(2)
    ay = col3.number_input("c (y-min)", value=-2.0, format="%.6f", key="ay")
    by = col4.number_input("d (y-max)", value=2.0, format="%.6f", key="by")

    if ax == bx or ay == by:
        st.error("Bounds cannot be equal (need non-zero area domain).")
        st.stop()
    if ax > bx:
        st.sidebar.warning("Swapping x-bounds because a > b.")
        ax, bx = bx, ax
    if ay > by:
        st.sidebar.warning("Swapping y-bounds because c > d.")
        ay, by = by, ay

    method_2d_name = st.sidebar.selectbox("Primary method (2D)", list(METHODS_2D.keys()), index=0, key="method_2d")
    method_2d_key = METHODS_2D[method_2d_name]

    st.sidebar.markdown("---")
    nx = st.sidebar.slider("nx (x partitions)", 10, 800, 80, step=10, key="nx")
    ny = st.sidebar.slider("ny (y partitions)", 10, 800, 80, step=10, key="ny")

    st.sidebar.markdown("---")
    show_grid = st.sidebar.checkbox("Show grid overlay (heatmap)", value=True, key="grid_overlay")
    gx = st.sidebar.slider("Resolution (x)", 40, 240, 120, step=20, key="gx")
    gy = st.sidebar.slider("Resolution (y)", 40, 240, 120, step=20, key="gy")

    st.sidebar.markdown("---")
    mc_samples = st.sidebar.slider("Monte Carlo samples (if MC chosen)", 500, 200000, 20000, step=500, key="mcN")
    mc_seed = st.sidebar.number_input("MC seed", value=42, step=1, key="mcSeed")

    st.sidebar.markdown("---")
    show_conv_2d = st.sidebar.checkbox("Show convergence diagnostics (2D)", value=True, key="conv_2d")
    nmax_2d = st.sidebar.slider("Convergence max n (n×n)", 50, 1000, 400, step=50, key="nmax_2d")
    step_2d = st.sidebar.slider("Convergence step", 10, 200, 50, step=10, key="step_2d")
    loglog_2d = st.sidebar.checkbox("Log-log view (2D)", value=True, key="loglog_2d")

    st.sidebar.markdown("---")
    st.sidebar.caption("Note: dblquad may fail for discontinuities/singularities; grid rules still run.")

    # Parse 2D
    try:
        expr_sym_2d, f2 = parse_function_2d(expr_2d)
        _ = np.array(f2(np.array([ax, (ax + bx) / 2, bx]), np.array([ay, (ay + by) / 2, by])), dtype=float)
    except Exception as e:
        st.error(f"Invalid 2D function. Parsing/evaluation failed: {e}")
        st.stop()

    # Reference dblquad
    ref_2d, referr_2d = reference_dblquad_2d(expr_2d, ax, bx, ay, by)

    # Primary approximation
    t0 = time.time()
    mc_stderr = None

    if method_2d_key == "midpoint":
        approx_2d = midpoint_2d(f2, ax, bx, ay, by, nx, ny)
    elif method_2d_key == "trapezoid":
        approx_2d = trapezoidal_2d(f2, ax, bx, ay, by, nx, ny)
    elif method_2d_key == "simpson":
        approx_2d = simpson_2d(f2, ax, bx, ay, by, nx, ny)
    else:
        approx_2d, mc_stderr = monte_carlo_2d(f2, ax, bx, ay, by, int(mc_samples), seed=int(mc_seed))

    elapsed = time.time() - t0

    # Method comparison table
    rows2 = []

    tA = time.time()
    v_mid = midpoint_2d(f2, ax, bx, ay, by, nx, ny)
    rows2.append(["Midpoint (Product)", v_mid, abs(ref_2d - v_mid) if ref_2d is not None else np.nan, time.time() - tA, "grid"])

    tA = time.time()
    v_tr = trapezoidal_2d(f2, ax, bx, ay, by, nx, ny)
    rows2.append(["Trapezoidal (Product)", v_tr, abs(ref_2d - v_tr) if ref_2d is not None else np.nan, time.time() - tA, "grid"])

    tA = time.time()
    v_si = simpson_2d(f2, ax, bx, ay, by, nx, ny)
    rows2.append(["Simpson (Product)", v_si, abs(ref_2d - v_si) if ref_2d is not None else np.nan, time.time() - tA, "grid"])

    tA = time.time()
    v_mc, se_mc = monte_carlo_2d(f2, ax, bx, ay, by, int(max(2000, min(50000, mc_samples))), seed=int(mc_seed))
    rows2.append(["Monte Carlo (2D)", v_mc, abs(ref_2d - v_mc) if ref_2d is not None else np.nan, time.time() - tA, f"SE≈{se_mc:.2e}"])

    df2 = pd.DataFrame(rows2, columns=["Method", "Approximation", "Abs Error (vs dblquad)", "Time (s)", "Notes"])
    df2s = df2.sort_values(by=["Abs Error (vs dblquad)"], ascending=True, na_position="last")

    # Metrics strip
    hx = (bx - ax) / nx
    hy = (by - ay) / ny
    M1, M2, M3, M4, M5 = st.columns([1.2, 1.1, 1.1, 1.0, 1.0])
    M1.metric("Primary Approx (2D)", f"{approx_2d:.8f}", f"hx={hx:.3g}  hy={hy:.3g}")
    if ref_2d is not None:
        M2.metric("SciPy dblquad", f"{ref_2d:.8f}", f"± {referr_2d:.2e}")
        M3.metric("Abs Error", f"{abs(ref_2d - approx_2d):.6e}", delta_color="inverse")
    else:
        M2.metric("SciPy dblquad", "n/a", "reference unavailable")
        M3.metric("Abs Error", "n/a", "—")

    if method_2d_key == "mc" and mc_stderr is not None:
        M4.metric("Monte Carlo SE", f"{mc_stderr:.2e}", f"N={int(mc_samples)}")
    else:
        M4.metric("Grid size", f"{nx}×{ny}", "product rule")

    M5.metric("Execution Time", f"{elapsed:.4f}s", method_2d_name)

    st.markdown("<div class='small-muted'>Diagnostics table (computed at the current grid):</div>", unsafe_allow_html=True)
    st.dataframe(df2s, use_container_width=True, hide_index=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    eng2, diag2 = st.tabs(["Engine View (2D)", "Diagnostics (2D)"])

    with eng2:
        # ✅ Heatmap + 3D interactive
        view_heat, view_3d = st.tabs(["Heatmap (Top View)", "3D Surface (Interactive)"])

        with view_heat:
            fig_h = plot_heatmap_2d(expr_2d, ax, bx, ay, by, gx, gy, show_grid)
            st.plotly_chart(fig_h, use_container_width=True)

        with view_3d:
            # Tip: if performance is heavy, reduce gx/gy
            fig_s = plot_surface_3d(expr_2d, ax, bx, ay, by, gx, gy)
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("<div class='small-muted'>Tip:</div>", unsafe_allow_html=True)
        st.write("- Use o 3D para enxergar a superfície (zoom/rotate/pan).")
        st.write("- Se ficar pesado, reduza **Resolution (x/y)** para 80–120.")

    with diag2:
        if show_conv_2d and ref_2d is not None:
            t0c = time.time()
            ns2, errs2 = convergence_series_2d(expr_2d, ax, bx, ay, by, method_2d_key, nmax_2d, step_2d, int(mc_seed))
            t1c = time.time()

            if ns2.size > 0:
                p_obs_2d = estimate_order_2d(ns2, np.maximum(errs2, EPS_LOG), ax, bx, ay, by)
                figc2 = plot_conv(ns2, errs2, "Absolute Error vs n (2D grid: n×n)", loglog_2d)
                st.plotly_chart(figc2, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Observed slope (order proxy)", f"{p_obs_2d:.3f}" if p_obs_2d is not None else "n/a", "log(err) vs log(h)")
                c2.metric("Reference", "dblquad", "SciPy adaptive")
                c3.metric("Diagnostics runtime", f"{(t1c - t0c):.3f}s", f"{len(ns2)} points")

                st.markdown("<div class='small-muted'>Raw convergence samples:</div>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({"n (grid)": ns2, "abs_error": errs2}), use_container_width=True, hide_index=True)
            else:
                st.info("Unable to compute convergence series.")
        else:
            st.info("2D convergence requires a valid dblquad reference. Try smoother functions or simpler bounds.")

    st.markdown("<div class='footer'>Unconventional Analysis Group • Quantitative Research Division</div>", unsafe_allow_html=True)
