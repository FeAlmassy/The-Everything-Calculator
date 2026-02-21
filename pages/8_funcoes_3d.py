# streamlit_app.py
# Function Visualizer + Analyzer — Professional Build (Streamlit + Plotly)
# ----------------------------------------------------------------------
# - Robust SymPy parsing (2D/3D) + NumPy lambdify
# - Interactive Plotly: pan/zoom/rotate (3D)
# - Dashboard metrics: critical points, maxima/minima, roots/level-set, concavity/curvature, etc.
# - Cached computations for performance
# - Clean UI architecture (tabs) + institutional styling (copied vibe from your engine)

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st


# ----------------------------
# 0) PAGE CONFIG (MUST BE FIRST)
# ----------------------------
st.set_page_config(page_title="Function Analyzer", layout="wide")


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
.kv { color: rgba(229,231,235,0.80); font-size: 0.92rem; }
.codebox {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 12px 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 2) SYMBOLS + PARSER
# ----------------------------
@st.cache_resource(show_spinner=False)
def sympy_env():
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)

    locals_map = {
        "x": x, "y": y,
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
        "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
        "Abs": sp.Abs, "abs": sp.Abs,
        "pi": sp.pi, "E": sp.E,
        "sign": sp.sign,
    }
    return x, y, locals_map


@st.cache_resource(show_spinner=False)
def parse_expr(expr_str: str, mode: str):
    x, y, locals_map = sympy_env()
    expr = sp.sympify(expr_str, locals=locals_map)

    if mode == "2D":
        f_num = sp.lambdify(x, expr, modules=["numpy"])
        return expr, f_num
    else:
        f_num = sp.lambdify((x, y), expr, modules=["numpy"])
        return expr, f_num


def _safe_np_array(v) -> np.ndarray:
    a = np.array(v, dtype=float)
    a[~np.isfinite(a)] = np.nan
    return a


# ----------------------------
# 3) CURVE/SURFACE EVALUATION (CACHE-FRIENDLY)
# ----------------------------
@st.cache_data(show_spinner=False)
def eval_curve(expr_str: str, x_min: float, x_max: float, points: int) -> Tuple[np.ndarray, np.ndarray]:
    expr, f_num = parse_expr(expr_str, "2D")
    pad = 0.10 * (x_max - x_min)
    xs = np.linspace(x_min - pad, x_max + pad, points)

    try:
        ys = _safe_np_array(f_num(xs))
    except Exception:
        ys = _safe_np_array([float(f_num(xx)) for xx in xs])

    return xs, ys


@st.cache_data(show_spinner=False)
def eval_surface(expr_str: str, x_min: float, x_max: float, y_min: float, y_max: float, res: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expr, f_num = parse_expr(expr_str, "3D")

    xs = np.linspace(x_min, x_max, res)
    ys = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    try:
        Z = _safe_np_array(f_num(X, Y))
    except Exception:
        Z = np.empty_like(X, dtype=float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = float(f_num(float(X[i, j]), float(Y[i, j])))
                except Exception:
                    Z[i, j] = np.nan
        Z[~np.isfinite(Z)] = np.nan

    return X, Y, Z


# ----------------------------
# 4) ANALYSIS UTILITIES
# ----------------------------
@dataclass
class Crit1D:
    x: float
    y: float
    kind: str   # "max" | "min" | "saddle/flat"
    details: str


def _float_or_none(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _unique_sorted(vals: List[float], tol: float = 1e-7) -> List[float]:
    if not vals:
        return []
    vals = sorted(vals)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return out


def analyze_2d(expr_str: str, x_min: float, x_max: float) -> Dict:
    x, _, _ = sympy_env()
    expr, f_num = parse_expr(expr_str, "2D")

    # Derivatives
    d1 = sp.diff(expr, x)
    d2 = sp.diff(expr, x, 2)

    # Roots
    roots = []
    try:
        sol = sp.solve(sp.Eq(expr, 0), x)
        for s in sol:
            sv = _float_or_none(sp.N(s))
            if sv is not None and (x_min <= sv <= x_max):
                roots.append(sv)
        roots = _unique_sorted(roots)
    except Exception:
        roots = []

    # Critical points (symbolic) in interval
    crits: List[Crit1D] = []
    try:
        solc = sp.solve(sp.Eq(d1, 0), x)
        xs = []
        for s in solc:
            sv = _float_or_none(sp.N(s))
            if sv is not None and (x_min <= sv <= x_max):
                xs.append(sv)
        xs = _unique_sorted(xs)

        for cx in xs:
            try:
                cy = float(f_num(np.array([cx], dtype=float))[0])
            except Exception:
                cy = float(f_num(float(cx)))

            d2v = None
            try:
                d2v = float(sp.N(d2.subs(x, cx)))
            except Exception:
                d2v = None

            if d2v is None:
                kind = "saddle/flat"
                details = "2nd derivative unavailable"
            else:
                if d2v < 0:
                    kind = "max"
                    details = f"f''(x)={d2v:.6g} < 0"
                elif d2v > 0:
                    kind = "min"
                    details = f"f''(x)={d2v:.6g} > 0"
                else:
                    kind = "saddle/flat"
                    details = f"f''(x)={d2v:.6g} ~ 0"

            crits.append(Crit1D(x=cx, y=cy, kind=kind, details=details))
    except Exception:
        crits = []

    # Interval extrema (sample + include criticals)
    xs_samp = np.linspace(x_min, x_max, 2501)
    try:
        ys_samp = _safe_np_array(f_num(xs_samp))
    except Exception:
        ys_samp = _safe_np_array([float(f_num(float(xx))) for xx in xs_samp])

    # Add critical points to sampling set for better picking
    for c in crits:
        idx = int(np.clip(round((c.x - x_min) / (x_max - x_min) * (len(xs_samp) - 1)), 0, len(xs_samp) - 1))
        ys_samp[idx] = c.y

    finite = np.isfinite(ys_samp)
    if finite.any():
        i_max = int(np.nanargmax(ys_samp))
        i_min = int(np.nanargmin(ys_samp))
        approx_max = (float(xs_samp[i_max]), float(ys_samp[i_max]))
        approx_min = (float(xs_samp[i_min]), float(ys_samp[i_min]))
    else:
        approx_max = None
        approx_min = None

    # y-intercept
    y0 = None
    if x_min <= 0 <= x_max:
        try:
            y0 = float(f_num(np.array([0.0], dtype=float))[0])
        except Exception:
            try:
                y0 = float(f_num(0.0))
            except Exception:
                y0 = None

    out = {
        "expr": expr,
        "d1": d1,
        "d2": d2,
        "roots": roots,
        "crits": crits,
        "approx_max": approx_max,
        "approx_min": approx_min,
        "y_intercept": y0,
    }
    return out


def analyze_3d(expr_str: str, x_min: float, x_max: float, y_min: float, y_max: float, res: int) -> Dict:
    x, y, _ = sympy_env()
    expr, f_num = parse_expr(expr_str, "3D")

    fx = sp.diff(expr, x)
    fy = sp.diff(expr, y)

    fxx = sp.diff(expr, x, 2)
    fyy = sp.diff(expr, y, 2)
    fxy = sp.diff(expr, x, y)

    # Stationary points (symbolic attempt)
    stationary = []
    stationary_note = "symbolic"
    try:
        sols = sp.solve([sp.Eq(fx, 0), sp.Eq(fy, 0)], [x, y], dict=True)
        pts = []
        for s in sols:
            xv = _float_or_none(sp.N(s.get(x)))
            yv = _float_or_none(sp.N(s.get(y)))
            if xv is None or yv is None:
                continue
            if (x_min <= xv <= x_max) and (y_min <= yv <= y_max):
                pts.append((xv, yv))
        # uniq
        pts_u = []
        for p in pts:
            if not any(abs(p[0]-q[0]) < 1e-7 and abs(p[1]-q[1]) < 1e-7 for q in pts_u):
                pts_u.append(p)

        for (xv, yv) in pts_u:
            try:
                zv = float(f_num(xv, yv))
            except Exception:
                zv = np.nan

            # Hessian classification (2D)
            try:
                Hxx = float(sp.N(fxx.subs({x: xv, y: yv})))
                Hyy = float(sp.N(fyy.subs({x: xv, y: yv})))
                Hxy = float(sp.N(fxy.subs({x: xv, y: yv})))
                det = Hxx * Hyy - Hxy * Hxy
                if det > 0 and Hxx > 0:
                    kind = "min"
                    details = f"det(H)={det:.6g} > 0, f_xx={Hxx:.6g} > 0"
                elif det > 0 and Hxx < 0:
                    kind = "max"
                    details = f"det(H)={det:.6g} > 0, f_xx={Hxx:.6g} < 0"
                elif det < 0:
                    kind = "saddle"
                    details = f"det(H)={det:.6g} < 0"
                else:
                    kind = "flat/undetermined"
                    details = f"det(H)={det:.6g} ~ 0"
            except Exception:
                kind = "undetermined"
                details = "Hessian eval failed"

            stationary.append({"x": xv, "y": yv, "z": zv, "kind": kind, "details": details})

    except Exception:
        stationary = []
        stationary_note = "numeric"

    # Numeric fallback: sample grid extrema
    X, Y, Z = eval_surface(expr_str, x_min, x_max, y_min, y_max, res)
    finite = np.isfinite(Z)
    approx_max = approx_min = None
    if finite.any():
        idx_max = np.nanargmax(Z)
        idx_min = np.nanargmin(Z)
        i_max, j_max = np.unravel_index(idx_max, Z.shape)
        i_min, j_min = np.unravel_index(idx_min, Z.shape)
        approx_max = (float(X[i_max, j_max]), float(Y[i_max, j_max]), float(Z[i_max, j_max]))
        approx_min = (float(X[i_min, j_min]), float(Y[i_min, j_min]), float(Z[i_min, j_min]))

    return {
        "expr": expr,
        "fx": fx, "fy": fy,
        "fxx": fxx, "fyy": fyy, "fxy": fxy,
        "stationary": stationary,
        "stationary_note": stationary_note,
        "approx_max": approx_max,
        "approx_min": approx_min,
    }


# ----------------------------
# 5) PLOTS
# ----------------------------
def plot_2d(expr_str: str, expr: sp.Expr, x_min: float, x_max: float, points: int, show_fill: bool, show_points: bool, analysis: Dict):
    xs, ys = eval_curve(expr_str, x_min, x_max, points)

    fig = go.Figure()

    # fill on [x_min, x_max]
    if show_fill:
        mask = (xs >= x_min) & (xs <= x_max)
        fig.add_trace(go.Scatter(
            x=xs[mask], y=ys[mask],
            fill="tozeroy",
            name="Area (visual)",
            fillcolor="rgba(255, 75, 75, 0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
        ))

    # glow
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip",
        showlegend=False,
    ))

    # main curve
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        name="f(x)",
        line=dict(color="#FF4B4B", width=3),
        hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>",
    ))

    if show_points:
        # roots
        roots = analysis.get("roots", [])
        if roots:
            rx = np.array(roots, dtype=float)
            ry = np.zeros_like(rx)
            fig.add_trace(go.Scatter(
                x=rx, y=ry,
                mode="markers",
                name="Roots",
                marker=dict(size=10, color="#1E90FF", line=dict(width=1, color="rgba(255,255,255,0.4)")),
                hovertemplate="root x=%{x:.6f}<extra></extra>",
            ))

        # critical points
        crits: List[Crit1D] = analysis.get("crits", [])
        if crits:
            cx = np.array([c.x for c in crits], dtype=float)
            cy = np.array([c.y for c in crits], dtype=float)
            text = [f"{c.kind} • {c.details}" for c in crits]
            fig.add_trace(go.Scatter(
                x=cx, y=cy,
                mode="markers",
                name="Critical pts",
                marker=dict(size=11, color="rgba(255,255,255,0.85)", line=dict(width=1, color="rgba(0,0,0,0.2)")),
                text=text,
                hovertemplate="x=%{x:.6f}<br>f=%{y:.6f}<br>%{text}<extra></extra>",
            ))

    fig.add_vline(x=x_min, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")
    fig.add_vline(x=x_max, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")

    fig.update_layout(
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition=dict(duration=450),
        title=f"f(x) = {sp.sstr(expr)}   |   [{x_min:.6g}, {x_max:.6g}]",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig


def plot_3d(expr_str: str, expr: sp.Expr, x_min: float, x_max: float, y_min: float, y_max: float, res: int,
            show_contour: bool, show_points: bool, analysis: Dict):
    X, Y, Z = eval_surface(expr_str, x_min, x_max, y_min, y_max, res)

    fig = go.Figure()

    surf_kwargs = dict(
        x=X, y=Y, z=Z,
        name="f(x,y)",
        hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        showscale=False,
    )

    if show_contour:
        surf_kwargs["contours"] = dict(
            z=dict(show=True, usecolormap=True, project_z=True, highlightwidth=1)
        )

    fig.add_trace(go.Surface(**surf_kwargs))

    # Stationary points (if any)
    if show_points:
        pts = analysis.get("stationary", [])
        if pts:
            px = np.array([p["x"] for p in pts], dtype=float)
            py = np.array([p["y"] for p in pts], dtype=float)
            pz = np.array([p["z"] for p in pts], dtype=float)
            text = [f'{p["kind"]} • {p["details"]}' for p in pts]
            fig.add_trace(go.Scatter3d(
                x=px, y=py, z=pz,
                mode="markers",
                name="Stationary pts",
                marker=dict(size=6, color="rgba(255,255,255,0.9)", line=dict(width=1, color="rgba(0,0,0,0.2)")),
                text=text,
                hovertemplate="x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<br>%{text}<extra></extra>",
            ))

        # Approx extrema from grid
        amax = analysis.get("approx_max")
        amin = analysis.get("approx_min")
        xs = []
        ys = []
        zs = []
        labels = []
        if amax is not None:
            xs.append(amax[0]); ys.append(amax[1]); zs.append(amax[2]); labels.append("grid max")
        if amin is not None:
            xs.append(amin[0]); ys.append(amin[1]); zs.append(amin[2]); labels.append("grid min")

        if xs:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers+text",
                name="Grid extrema",
                marker=dict(size=7, color="#FF4B4B"),
                text=labels,
                textposition="top center",
                hovertemplate="x=%{x:.5f}<br>y=%{y:.5f}<br>z=%{z:.5f}<extra></extra>",
            ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=50, b=0),
        title=f"f(x,y) = {sp.sstr(expr)}   |   x∈[{x_min:.6g},{x_max:.6g}], y∈[{y_min:.6g},{y_max:.6g}]",
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
            zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        ),
    )
    return fig


# ----------------------------
# 6) THEORY PANEL (same vibe)
# ----------------------------
def theory_panel():
    st.markdown("## 🧠 Mathematical Foundations")
    st.markdown(
        "<span class='badge'>2D Analysis</span> "
        "<span class='badge'>3D Geometry</span> "
        "<span class='badge'>Critical Points</span> "
        "<span class='badge'>Curvature</span>",
        unsafe_allow_html=True,
    )

    with st.expander("Open full theory (derivatives, extrema, roots, curvature)", expanded=False):
        st.markdown("### 1) 2D function")
        st.latex(r"f:\mathbb{R}\to\mathbb{R},\quad y=f(x)")
        st.markdown("- **Roots:** solve \(f(x)=0\).")
        st.markdown("- **Critical points:** solve \(f'(x)=0\).")
        st.markdown("- **Concavity:** sign of \(f''(x)\).")
        st.markdown("  - \(f''(x)<0\) ⇒ locally concave down (peak-like).")
        st.markdown("  - \(f''(x)>0\) ⇒ locally concave up (valley-like).")

        st.markdown("---")

        st.markdown("### 2) 3D function (surface)")
        st.latex(r"f:\mathbb{R}^2\to\mathbb{R},\quad z=f(x,y)")
        st.markdown("- **Stationary points:** solve the gradient system")
        st.latex(r"\nabla f(x,y)=\left(\frac{\partial f}{\partial x},\frac{\partial f}{\partial y}\right)=(0,0)")
        st.markdown("- **Classification (Hessian):**")
        st.latex(r"H=\begin{pmatrix}f_{xx}&f_{xy}\\f_{xy}&f_{yy}\end{pmatrix},\quad \det(H)=f_{xx}f_{yy}-f_{xy}^2")
        st.markdown("  - \(\det(H)>0\) and \(f_{xx}>0\) ⇒ local **min**")
        st.markdown("  - \(\det(H)>0\) and \(f_{xx}<0\) ⇒ local **max**")
        st.markdown("  - \(\det(H)<0\) ⇒ **saddle**")


# ----------------------------
# 7) HEADER
# ----------------------------
st.title("🧩 Function Visualizer + Analyzer")
st.caption("SymPy Parsing • Plotly Interactivity • 2D/3D Diagnostics • Clean UI")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

theory_panel()
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 8) SIDEBAR (CONTROLS)
# ----------------------------
st.sidebar.header("Controls")

mode = st.sidebar.selectbox("Mode", ["2D", "3D"], index=0)

examples_2d = {
    "Downward parabola (default)": "-x**2 + 1",
    "Sine + trend": "sin(x) - 0.1*x",
    "Exponential bump": "exp(-x**2)",
    "Abs (non-smooth)": "Abs(x) - 0.5",
}
examples_3d = {
    "Downward paraboloid (default)": "-(x**2 + y**2) + 1",
    "Saddle": "x**2 - y**2",
    "Ripple": "sin(x)*cos(y)",
    "Gaussian hill": "exp(-(x**2 + y**2))",
}

if mode == "2D":
    pick = st.sidebar.selectbox("Quick examples", list(examples_2d.keys()), index=0)
    default_expr = examples_2d[pick]
else:
    pick = st.sidebar.selectbox("Quick examples", list(examples_3d.keys()), index=0)
    default_expr = examples_3d[pick]

expr_str = st.sidebar.text_input("f(x) / f(x,y) (SymPy syntax)", value=default_expr)

st.sidebar.markdown("---")
st.sidebar.subheader("Domain / View")

if mode == "2D":
    c1, c2 = st.sidebar.columns(2)
    x_min = c1.number_input("x min", value=-2.0, format="%.6f")
    x_max = c2.number_input("x max", value=2.0, format="%.6f")
    if x_min == x_max:
        st.error("x min and x max cannot be equal.")
        st.stop()
    if x_min > x_max:
        st.sidebar.warning("Swapping bounds because x min > x max.")
        x_min, x_max = x_max, x_min

    points = st.sidebar.slider("Plot points", 300, 6000, 1400, step=100)
    show_fill = st.sidebar.checkbox("Show area fill", value=True)
    show_points = st.sidebar.checkbox("Show roots & critical points", value=True)

else:
    c1, c2 = st.sidebar.columns(2)
    x_min = c1.number_input("x min", value=-2.0, format="%.6f")
    x_max = c2.number_input("x max", value=2.0, format="%.6f")
    c3, c4 = st.sidebar.columns(2)
    y_min = c3.number_input("y min", value=-2.0, format="%.6f")
    y_max = c4.number_input("y max", value=2.0, format="%.6f")

    if x_min == x_max or y_min == y_max:
        st.error("Bounds cannot be equal.")
        st.stop()

    if x_min > x_max:
        st.sidebar.warning("Swapping x bounds because x min > x max.")
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        st.sidebar.warning("Swapping y bounds because y min > y max.")
        y_min, y_max = y_max, y_min

    res = st.sidebar.slider("Surface resolution", 25, 180, 70, step=5)
    show_contour = st.sidebar.checkbox("Show contour projection", value=True)
    show_points = st.sidebar.checkbox("Show stationary/extrema markers", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: use **Plotly controls** to zoom/pan (2D) and rotate/zoom (3D).")


# ----------------------------
# 9) PARSE FUNCTION (FAIL FAST)
# ----------------------------
try:
    expr, f_num = parse_expr(expr_str, mode)
    # quick eval test
    if mode == "2D":
        test = f_num(np.array([x_min, (x_min + x_max) / 2, x_max], dtype=float))
        _ = _safe_np_array(test)
    else:
        test = f_num(0.0, 0.0)
        _ = float(test)
except Exception as e:
    st.error(f"Invalid function. Parsing/evaluation failed: {e}")
    st.stop()


# ----------------------------
# 10) ANALYZE + METRICS STRIP
# ----------------------------
t0 = time.time()
if mode == "2D":
    analysis = analyze_2d(expr_str, x_min, x_max)
else:
    analysis = analyze_3d(expr_str, x_min, x_max, y_min, y_max, res)
t1 = time.time()

if mode == "2D":
    approx_max = analysis.get("approx_max")
    approx_min = analysis.get("approx_min")
    roots = analysis.get("roots", [])
    crits: List[Crit1D] = analysis.get("crits", [])

    m1, m2, m3, m4, m5 = st.columns([1.25, 1.05, 1.05, 1.0, 1.0])

    m1.metric("Function", "2D", f"[{x_min:.3g}, {x_max:.3g}]")
    m2.metric("Roots in range", f"{len(roots)}", "f(x)=0")
    m3.metric("Critical points", f"{len(crits)}", "f'(x)=0")

    if approx_max is not None:
        m4.metric("Max (approx)", f"{approx_max[1]:.6g}", f"x≈{approx_max[0]:.6g}")
    else:
        m4.metric("Max (approx)", "n/a", "—")

    if approx_min is not None:
        m5.metric("Min (approx)", f"{approx_min[1]:.6g}", f"x≈{approx_min[0]:.6g}")
    else:
        m5.metric("Min (approx)", "n/a", "—")

else:
    approx_max = analysis.get("approx_max")
    approx_min = analysis.get("approx_min")
    stationary = analysis.get("stationary", [])
    note = analysis.get("stationary_note", "—")

    m1, m2, m3, m4, m5 = st.columns([1.25, 1.05, 1.05, 1.0, 1.0])

    m1.metric("Function", "3D", f"x∈[{x_min:.3g},{x_max:.3g}]  y∈[{y_min:.3g},{y_max:.3g}]")
    m2.metric("Stationary pts", f"{len(stationary)}", note)
    m3.metric("Grid res", f"{res}×{res}", "sampling")

    if approx_max is not None:
        m4.metric("Max (grid)", f"{approx_max[2]:.6g}", f"({approx_max[0]:.4g},{approx_max[1]:.4g})")
    else:
        m4.metric("Max (grid)", "n/a", "—")

    if approx_min is not None:
        m5.metric("Min (grid)", f"{approx_min[2]:.6g}", f"({approx_min[0]:.4g},{approx_min[1]:.4g})")
    else:
        m5.metric("Min (grid)", "n/a", "—")

st.markdown("<div class='small-muted'>Diagnostics computed on the current view window:</div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 11) TABS: VIEW / DIAGNOSTICS
# ----------------------------
tab_view, tab_diag = st.tabs(["Engine View", "Diagnostics"])


with tab_view:
    if mode == "2D":
        fig = plot_2d(expr_str, expr, x_min, x_max, points, show_fill, show_points, analysis)
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_3d(expr_str, expr, x_min, x_max, y_min, y_max, res, show_contour, show_points, analysis)
        st.plotly_chart(fig, use_container_width=True)


with tab_diag:
    if mode == "2D":
        roots = analysis.get("roots", [])
        crits: List[Crit1D] = analysis.get("crits", [])
        y0 = analysis.get("y_intercept")

        left, right = st.columns([1.15, 1.0])

        with left:
            st.markdown("### 🔎 Symbolic summary")
            st.markdown("<div class='codebox'>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b>f(x)</b> = {sp.sstr(analysis['expr'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b>f'(x)</b> = {sp.sstr(analysis['d1'])}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b>f''(x)</b> = {sp.sstr(analysis['d2'])}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 📌 Key points")
            rows = []
            for c in crits:
                rows.append([c.kind, c.x, c.y, c.details])
            dfc = pd.DataFrame(rows, columns=["Type", "x", "f(x)", "Test"])
            st.dataframe(dfc, use_container_width=True, hide_index=True)

        with right:
            st.markdown("### 🧾 Metrics & notes")
            st.markdown("<div class='codebox'>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b>Domain (view)</b>: x ∈ [{x_min:.6g}, {x_max:.6g}]</div>", unsafe_allow_html=True)
            if y0 is not None:
                st.markdown(f"<div class='kv'><b>y-intercept</b>: f(0) = {y0:.6g}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='kv'><b>y-intercept</b>: n/a (0 not in range or eval failed)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b># roots</b>: {len(roots)}</div>", unsafe_allow_html=True)
            if roots:
                st.markdown(f"<div class='kv'><b>roots</b>: {', '.join([f'{r:.6g}' for r in roots[:12]])}{' …' if len(roots)>12 else ''}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kv'><b>analysis runtime</b>: {(t1 - t0):.3f}s</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        stationary = analysis.get("stationary", [])
        st.markdown("### 🔎 Symbolic summary")
        st.markdown("<div class='codebox'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f(x,y)</b> = {sp.sstr(analysis['expr'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f_x</b> = {sp.sstr(analysis['fx'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f_y</b> = {sp.sstr(analysis['fy'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f_xx</b> = {sp.sstr(analysis['fxx'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f_yy</b> = {sp.sstr(analysis['fyy'])}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='kv'><b>f_xy</b> = {sp.sstr(analysis['fxy'])}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📌 Stationary points (if solved)")
        if stationary:
            dfp = pd.DataFrame(
                [[p["kind"], p["x"], p["y"], p["z"], p["details"]] for p in stationary],
                columns=["Type", "x", "y", "f(x,y)", "Hessian test"]
            )
            st.dataframe(dfp, use_container_width=True, hide_index=True)
        else:
            st.info("No stationary points found symbolically in the current window (or solver could not resolve). Grid extrema are still shown (if enabled).")

        st.markdown("### 🧾 Runtime")
        st.markdown(f"<div class='small-muted'>Diagnostics runtime: {(t1 - t0):.3f}s</div>", unsafe_allow_html=True)


# ----------------------------
# 12) FOOTER
# ----------------------------
st.markdown("<div class='footer'>Unconventional Analysis Group - Fellipe Almassy • </div>", unsafe_allow_html=True)
