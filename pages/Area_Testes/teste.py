# pages/Area_Testes/markov.py
# Cadeias de Markov — The Everything Calculator
# ------------------------------------------------------------
# - Simulação interativa com animação em tempo real
# - Grafo da cadeia com Plotly (arestas + pesos)
# - Distribuição estacionária (autovetor)
# - Evolução temporal da distribuição
# - Múltiplos exemplos práticos
# - Diagnósticos de convergência
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# 0) CONFIG DA PÁGINA
# ----------------------------
st.set_page_config(page_title="Cadeias de Markov", layout="wide")


# ----------------------------
# 1) CSS — mesmo estilo do TEC
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
# 2) EXEMPLOS PRÁTICOS
# ----------------------------
EXEMPLOS_MARKOV = {
    "Clima (Sol / Nublado / Chuva)": {
        "estados": ["Sol", "Nublado", "Chuva"],
        "matriz": np.array([
            [0.70, 0.20, 0.10],
            [0.30, 0.40, 0.30],
            [0.20, 0.30, 0.50],
        ]),
        "descricao": "Previsão do tempo: transições diárias entre condições climáticas.",
        "emoji": "☁️",
    },
    "Mercado de Ações (Bull / Bear / Lateral)": {
        "estados": ["Bull", "Bear", "Lateral"],
        "matriz": np.array([
            [0.60, 0.20, 0.20],
            [0.25, 0.55, 0.20],
            [0.30, 0.30, 0.40],
        ]),
        "descricao": "Regimes de mercado: transições semanais entre tendências.",
        "emoji": "📈",
    },
    "Passeio Aleatório (3 estados)": {
        "estados": ["Esq.", "Centro", "Dir."],
        "matriz": np.array([
            [0.50, 0.50, 0.00],
            [0.25, 0.50, 0.25],
            [0.00, 0.50, 0.50],
        ]),
        "descricao": "Passeio aleatório com fronteiras refletivas.",
        "emoji": "🎲",
    },
    "Saúde do Cliente (Ativo / Risco / Churned)": {
        "estados": ["Ativo", "Em Risco", "Churned"],
        "matriz": np.array([
            [0.80, 0.15, 0.05],
            [0.30, 0.50, 0.20],
            [0.10, 0.10, 0.80],
        ]),
        "descricao": "Customer Success: evolução do estado de clientes ao longo do tempo.",
        "emoji": "👤",
    },
    "Personalizado": {
        "estados": ["A", "B", "C"],
        "matriz": np.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.3, 0.3, 0.4],
        ]),
        "descricao": "Defina sua própria cadeia de Markov.",
        "emoji": "✏️",
    },
}

CORES_ESTADOS = ["#FF4B4B", "#1E90FF", "#2ECC71", "#F39C12", "#9B59B6"]


# ----------------------------
# 3) FUNÇÕES DE COMPUTAÇÃO
# ----------------------------
@st.cache_data(show_spinner=False)
def validar_matriz(mat: tuple) -> tuple[bool, str]:
    P = np.array(mat)
    n = P.shape[0]
    if np.any(P < 0):
        return False, "Existem probabilidades negativas."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        erros = [f"Linha {i+1}: soma={s:.4f}" for i, s in enumerate(row_sums) if not np.isclose(s, 1.0, atol=1e-6)]
        return False, "Linhas não somam 1: " + ", ".join(erros)
    return True, "OK"


@st.cache_data(show_spinner=False)
def calcular_estacionaria(mat: tuple) -> Optional[np.ndarray]:
    P = np.array(mat)
    try:
        vals, vecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        pi = np.real(vecs[:, idx])
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def evolucao_distribuicao(mat: tuple, dist_inicial: tuple, n_passos: int) -> np.ndarray:
    P = np.array(mat)
    v = np.array(dist_inicial, dtype=float)
    v /= v.sum()
    historico = [v.copy()]
    for _ in range(n_passos):
        v = v @ P
        historico.append(v.copy())
    return np.array(historico)


@st.cache_data(show_spinner=False)
def simular_cadeia(mat: tuple, estado_inicial: int, n_passos: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    P = np.array(mat)
    n = P.shape[0]
    trajetoria = [estado_inicial]
    estado = estado_inicial
    for _ in range(n_passos):
        estado = int(rng.choice(n, p=P[estado]))
        trajetoria.append(estado)
    return trajetoria


# ----------------------------
# 4) GRÁFICOS
# ----------------------------
def make_grafo_markov(estados: list[str], P: np.ndarray, pi: Optional[np.ndarray]) -> go.Figure:
    n = len(estados)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    xs = np.cos(angulos)
    ys = np.sin(angulos)

    fig = go.Figure()

    for i in range(n):
        for j in range(n):
            p_ij = float(P[i, j])
            if p_ij < 0.01:
                continue

            xi, yi = float(xs[i]), float(ys[i])
            xj, yj = float(xs[j]), float(ys[j])

            if i == j:
                loop_x = xi + 0.30 * float(np.cos(angulos[i]))
                loop_y = yi + 0.30 * float(np.sin(angulos[i]))
                fig.add_trace(go.Scatter(
                    x=[loop_x], y=[loop_y],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=30,
                        color=CORES_ESTADOS[i % len(CORES_ESTADOS)],
                        line=dict(width=max(1.5, p_ij * 5), color=CORES_ESTADOS[i % len(CORES_ESTADOS)]),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_annotation(
                    x=loop_x, y=loop_y,
                    text=f"<b>{p_ij:.2f}</b>",
                    showarrow=False,
                    font=dict(size=10, color="rgba(229,231,235,0.95)"),
                    bgcolor="rgba(14,17,23,0.85)",
                )
            else:
                dx, dy = xj - xi, yj - yi
                mid_x = (xi + xj) / 2 - dy * 0.20
                mid_y = (yi + yj) / 2 + dx * 0.20
                width = max(1.0, p_ij * 7)
                alpha = 0.30 + 0.65 * p_ij
                cor = CORES_ESTADOS[i % len(CORES_ESTADOS)]

                fig.add_trace(go.Scatter(
                    x=[xi, mid_x, xj],
                    y=[yi, mid_y, yj],
                    mode="lines",
                    line=dict(width=width, color=cor),
                    opacity=alpha,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"<b>{p_ij:.2f}</b>",
                    showarrow=False,
                    font=dict(size=9, color="rgba(229,231,235,0.85)"),
                    bgcolor="rgba(14,17,23,0.80)",
                    borderpad=2,
                )

    # Nós
    tamanhos = [30 + 60 * float(pi[i]) for i in range(n)] if pi is not None else [45] * n
    hover = [
        f"<b>{est}</b><br>π = {float(pi[i]):.4f}" if pi is not None else f"<b>{est}</b>"
        for i, est in enumerate(estados)
    ]

    fig.add_trace(go.Scatter(
        x=list(xs), y=list(ys),
        mode="markers+text",
        marker=dict(
            size=tamanhos,
            color=CORES_ESTADOS[:n],
            line=dict(color="rgba(255,255,255,0.25)", width=1.5),
        ),
        text=estados,
        textposition="middle center",
        textfont=dict(size=11, color="white", family="monospace"),
        showlegend=False,
        hovertemplate=[h + "<extra></extra>" for h in hover],
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[-1.65, 1.65], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.65, 1.65], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def make_evolucao_plot(historico: np.ndarray, estados: list[str], pi: Optional[np.ndarray]) -> go.Figure:
    fig = go.Figure()
    passos = np.arange(historico.shape[0])

    for i, estado in enumerate(estados):
        cor = CORES_ESTADOS[i % len(CORES_ESTADOS)]
        fig.add_trace(go.Scatter(
            x=passos, y=historico[:, i],
            mode="lines",
            name=estado,
            line=dict(color=cor, width=2.5),
            hovertemplate=f"{estado}: %{{y:.4f}}<extra></extra>",
        ))
        if pi is not None:
            fig.add_hline(
                y=float(pi[i]),
                line_dash="dot",
                line_color=cor,
                line_width=1,
                opacity=0.4,
                annotation_text=f"π({estado})={float(pi[i]):.3f}",
                annotation_position="right",
                annotation_font=dict(size=10, color=cor),
            )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=100, t=30, b=0),
        title="Evolução da Distribuição P(estado | t)",
        xaxis_title="Passo t",
        yaxis_title="Probabilidade",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_trajetoria_plot(trajetoria: list[int], estados: list[str]) -> go.Figure:
    n = len(trajetoria)
    cores_traj = [CORES_ESTADOS[s % len(CORES_ESTADOS)] for s in trajetoria]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(n)),
        y=trajetoria,
        mode="lines+markers",
        line=dict(color="rgba(30,144,255,0.55)", width=1.5, shape="hv"),
        marker=dict(size=5, color=cores_traj, line=dict(color="rgba(255,255,255,0.3)", width=0.5)),
        name="Trajetória",
        hovertemplate="t=%{x}<br>Estado=%{text}<extra></extra>",
        text=[estados[s] for s in trajetoria],
    ))

    freq = np.bincount(trajetoria, minlength=len(estados)) / n
    for i, (est, f) in enumerate(zip(estados, freq)):
        fig.add_annotation(
            x=n * 1.01, y=i,
            text=f"{est}: {f:.3f}",
            showarrow=False,
            font=dict(size=9, color=CORES_ESTADOS[i % len(CORES_ESTADOS)]),
            xanchor="left",
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=100, t=30, b=0),
        title="Trajetória Simulada",
        xaxis_title="Passo t",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(estados))),
            ticktext=estados,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
    )
    return fig


def make_heatmap_transicao(P: np.ndarray, estados: list[str]) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=P,
        x=estados,
        y=estados,
        colorscale=[
            [0.0, "rgba(14,17,23,1)"],
            [0.5, "rgba(30,144,255,0.6)"],
            [1.0, "rgba(255,75,75,1)"],
        ],
        zmin=0, zmax=1,
        text=[[f"{P[i,j]:.3f}" for j in range(len(estados))] for i in range(len(estados))],
        texttemplate="%{text}",
        textfont=dict(size=13),
        hovertemplate="De: %{y}<br>Para: %{x}<br>P = %{z:.4f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="P(i→j)", thickness=14),
    ))
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        title="Matriz de Transição P",
        xaxis_title="Estado Destino",
        yaxis_title="Estado Origem",
        height=320,
    )
    return fig


def make_convergencia_markov(historico: np.ndarray, pi: np.ndarray) -> go.Figure:
    passos = np.arange(historico.shape[0])
    tv = 0.5 * np.sum(np.abs(historico - pi[None, :]), axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=passos, y=tv,
        mode="lines",
        name="Distância TV",
        line=dict(color="#FF4B4B", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,75,75,0.08)",
        hovertemplate="t=%{x}<br>TV=%{y:.6f}<extra></extra>",
    ))
    use_log = bool(tv[1:].max() > 1e-10 and tv[1:].min() > 0)
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        title="Convergência: Distância de Variação Total ||μₜ − π||_TV",
        xaxis_title="Passo t",
        yaxis_title="||μₜ − π||_TV",
        yaxis_type="log" if use_log else "linear",
    )
    return fig


# ----------------------------
# 5) EDITOR DE MATRIZ (modo personalizado)
# ----------------------------
def editor_matriz(estados: list[str], P_default: np.ndarray) -> tuple[list[str], np.ndarray]:
    n = len(estados)
    cols_nomes = st.columns(n)
    novos_estados = []
    for i, col in enumerate(cols_nomes):
        nome = col.text_input(f"Estado {i+1}", value=estados[i], key=f"est_nome_{i}", label_visibility="collapsed")
        novos_estados.append(nome.strip() if nome.strip() else estados[i])

    P_nova = np.zeros((n, n))
    st.markdown("<div class='small-muted'>Matriz de transição — cada linha deve somar 1 (normalização automática):</div>", unsafe_allow_html=True)

    header = st.columns([1.2] + [1] * n)
    header[0].markdown("**De \\ Para**")
    for j, est in enumerate(novos_estados):
        header[j + 1].markdown(f"**{est}**")

    for i in range(n):
        row_cols = st.columns([1.2] + [1] * n)
        row_cols[0].markdown(f"**{novos_estados[i]}**")
        vals_row = []
        for j in range(n):
            val = row_cols[j + 1].number_input(
                f"P[{i},{j}]",
                min_value=0.0, max_value=1.0,
                value=float(P_default[i, j]),
                step=0.05, format="%.2f",
                key=f"p_{i}_{j}",
                label_visibility="collapsed",
            )
            vals_row.append(val)
        soma = sum(vals_row)
        P_nova[i] = [v / soma for v in vals_row] if soma > 0 else [1.0 / n] * n

    return novos_estados, P_nova


# ----------------------------
# 6) CABEÇALHO
# ----------------------------
st.title("Cadeias de Markov")
st.caption("Processos estocásticos com a propriedade de Markov")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 7) TEORIA COLAPSÁVEL
# ----------------------------
with st.expander("Teoria: o que é uma Cadeia de Markov?", expanded=False):
    st.markdown(
        "<span class='badge'>Propriedade de Markov</span> "
        "<span class='badge'>Matriz de Transição</span> "
        "<span class='badge'>Distribuição Estacionária</span> "
        "<span class='badge'>Convergência</span>",
        unsafe_allow_html=True,
    )
    st.markdown("Uma cadeia de Markov é um processo estocástico com a **propriedade de Markov**:")
    st.latex(r"\Pr(X_{t+1} = j \mid X_t = i,\, X_{t-1},\, \ldots) = \Pr(X_{t+1} = j \mid X_t = i) = P_{ij}")
    st.markdown("O futuro depende **apenas do estado presente**, não da história completa.")
    st.markdown("---")
    st.markdown("**Matriz de Transição** — cada linha soma 1 (matriz estocástica por linhas):")
    st.latex(r"P = \begin{bmatrix} P_{11} & P_{12} & \cdots \\ P_{21} & P_{22} & \cdots \\ \vdots & & \ddots \end{bmatrix}, \qquad \sum_j P_{ij} = 1 \; \forall\, i")
    st.markdown("**Distribuição no passo t** — basta multiplicar à esquerda repetidamente:")
    st.latex(r"\mu_t = \mu_0 \cdot P^t")
    st.markdown("**Distribuição Estacionária** — vetor π tal que:")
    st.latex(r"\pi P = \pi \qquad \Longleftrightarrow \qquad \pi \text{ é autovetor esquerdo de } P \text{ com autovalor } 1")
    st.markdown("**Teorema Ergódico** — para cadeias irredutíveis e aperiódicas:")
    st.latex(r"\lim_{t \to \infty} \mu_t = \pi \qquad \text{(independente de } \mu_0\text{)}")
    st.markdown("A velocidade de convergência é controlada pelo **spectral gap**: $1 - |\\lambda_2|$, onde $\\lambda_2$ é o segundo maior autovalor em módulo.")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 8) SIDEBAR — CONTROLES
# ----------------------------
st.sidebar.header("Controles")

exemplo_nome = st.sidebar.selectbox("Exemplo", list(EXEMPLOS_MARKOV.keys()), index=0)
exemplo = EXEMPLOS_MARKOV[exemplo_nome]

if exemplo_nome == "Personalizado":
    n_estados = st.sidebar.slider("Nº de estados", 2, 5, 3, key="n_estados")
    estados_default = [chr(65 + i) for i in range(n_estados)]
    P_default = np.full((n_estados, n_estados), 1.0 / n_estados)
    estados_edit = estados_default
    P_edit = P_default
else:
    estados_edit = list(exemplo["estados"])
    P_edit = exemplo["matriz"].copy()
    n_estados = len(estados_edit)

n_passos_evolucao = st.sidebar.slider("Passos de evolução", 10, 200, 60, step=5)
n_passos_sim      = st.sidebar.slider("Passos da simulação", 50, 2000, 300, step=50)
seed_sim          = st.sidebar.number_input("Seed da simulação", value=42, min_value=0, max_value=9999, step=1)

st.sidebar.markdown("**Distribuição Inicial μ₀**")
dist_cols = st.sidebar.columns(n_estados)
dist_inicial_raw = []
for i, col in enumerate(dist_cols):
    v = col.number_input(
        estados_edit[i],
        min_value=0.0, max_value=1.0,
        value=round(1.0 / n_estados, 2),
        step=0.1, format="%.2f",
        key=f"dist0_{i}",
    )
    dist_inicial_raw.append(v)

s = sum(dist_inicial_raw)
dist_inicial = tuple(v / s for v in dist_inicial_raw) if s > 0 else tuple(1.0 / n_estados for _ in range(n_estados))
st.sidebar.caption(f"μ₀ normalizado: {[f'{v:.3f}' for v in dist_inicial]}")

estado_inicial_nome = st.sidebar.selectbox("Estado inicial (simulação)", estados_edit)
estado_inicial_idx  = estados_edit.index(estado_inicial_nome)

st.sidebar.markdown("---")
st.sidebar.caption("Dica: matrizes com |λ₂| próximo de 1 convergem lentamente — veja nos Diagnósticos.")


# ----------------------------
# 9) EDITOR (modo personalizado) OU DESCRIÇÃO
# ----------------------------
if exemplo_nome == "Personalizado":
    st.markdown("#### Definir Cadeia")
    estados_edit, P_edit = editor_matriz(estados_edit, P_edit)
else:
    st.markdown(f"**{exemplo['emoji']} {exemplo_nome}** — {exemplo['descricao']}")

P      = P_edit
estados = estados_edit
mat_tuple = tuple(map(tuple, P))


# ----------------------------
# 10) VALIDAÇÃO
# ----------------------------
valido, msg = validar_matriz(mat_tuple)
if not valido:
    st.error(f"Matriz inválida: {msg}")
    st.stop()


# ----------------------------
# 11) COMPUTAÇÕES
# ----------------------------
pi        = calcular_estacionaria(mat_tuple)
historico = evolucao_distribuicao(mat_tuple, dist_inicial, n_passos_evolucao)
trajetoria = simular_cadeia(mat_tuple, estado_inicial_idx, n_passos_sim, int(seed_sim))

n_est = len(estados)
autovalores  = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
lambda2      = float(autovalores[1]) if len(autovalores) > 1 else 0.0
spectral_gap = 1.0 - lambda2
mixing_time  = int(np.ceil(1.0 / spectral_gap)) if spectral_gap > 1e-10 else 9999


# ----------------------------
# 12) MÉTRICAS PRINCIPAIS
# ----------------------------
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Nº de estados",       str(n_est))
mc2.metric("λ₂ (2º autovalor)",   f"{lambda2:.4f}",       "espectral")
mc3.metric("Spectral gap",         f"{spectral_gap:.4f}",  "1 − |λ₂|")
mc4.metric("Mixing time (est.)",   f"~{mixing_time} passos", "1 / gap")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 13) DISTRIBUIÇÃO ESTACIONÁRIA
# ----------------------------
if pi is not None:
    st.markdown("**Distribuição Estacionária π**")
    cols_pi = st.columns(n_est)
    for i, col in enumerate(cols_pi):
        col.metric(estados[i], f"{float(pi[i]):.4f}", f"{float(pi[i])*100:.1f}%")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# ----------------------------
# 14) ABAS
# ----------------------------
tab_grafo, tab_evolucao, tab_sim, tab_diag = st.tabs([
    "🔵 Grafo da Cadeia",
    "📊 Evolução Temporal",
    "🎲 Simulação",
    "🔬 Diagnósticos",
])

with tab_grafo:
    col_g, col_h = st.columns([1.2, 1])
    with col_g:
        st.plotly_chart(make_grafo_markov(estados, P, pi), use_container_width=True)
    with col_h:
        st.plotly_chart(make_heatmap_transicao(P, estados), use_container_width=True)
        df_P = pd.DataFrame(P, index=estados, columns=estados)
        st.markdown("<div class='small-muted'>Matriz P (numérica):</div>", unsafe_allow_html=True)
        st.dataframe(
            df_P.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None, vmin=0, vmax=1),
            use_container_width=True,
        )

with tab_evolucao:
    st.plotly_chart(make_evolucao_plot(historico, estados, pi), use_container_width=True)

    if pi is not None:
        tv_final = 0.5 * float(np.sum(np.abs(historico[-1] - pi)))
        st.markdown(
            f"<div class='small-muted'>Distância TV após {n_passos_evolucao} passos: <b>{tv_final:.6f}</b></div>",
            unsafe_allow_html=True,
        )

    df_hist = pd.DataFrame(
        historico[::max(1, n_passos_evolucao // 20)],
        columns=estados,
    )
    df_hist.index = df_hist.index * max(1, n_passos_evolucao // 20)
    df_hist.index.name = "Passo"
    st.markdown("<div class='small-muted'>Amostras da distribuição μₜ:</div>", unsafe_allow_html=True)
    st.dataframe(df_hist.style.format("{:.6f}"), use_container_width=True)

with tab_sim:
    st.plotly_chart(make_trajetoria_plot(trajetoria, estados), use_container_width=True)

    freq_emp = np.bincount(trajetoria, minlength=n_est) / len(trajetoria)
    df_freq = pd.DataFrame({
        "Estado":          estados,
        "Freq. Empírica":  [f"{v:.4f}" for v in freq_emp],
        "π (Teórica)":     [f"{float(pi[i]):.4f}" for i in range(n_est)] if pi is not None else ["—"] * n_est,
        "Erro |emp − π|":  [f"{abs(freq_emp[i] - float(pi[i])):.4f}" for i in range(n_est)] if pi is not None else ["—"] * n_est,
    })
    st.markdown("<div class='small-muted'>Frequência empírica vs distribuição estacionária:</div>", unsafe_allow_html=True)
    st.dataframe(df_freq, use_container_width=True, hide_index=True)
    st.caption(f"Trajetória: {n_passos_sim} passos | Seed: {int(seed_sim)} | Estado inicial: {estados[estado_inicial_idx]}")

with tab_diag:
    if pi is not None:
        st.plotly_chart(make_convergencia_markov(historico, pi), use_container_width=True)

        autovalores_full = np.linalg.eigvals(P)
        df_autos = pd.DataFrame({
            "Autovalor (Re)": [f"{v.real:.6f}" for v in sorted(autovalores_full, key=abs, reverse=True)],
            "Autovalor (Im)": [f"{v.imag:.6f}" for v in sorted(autovalores_full, key=abs, reverse=True)],
            "|λ|":            [f"{abs(v):.6f}"  for v in sorted(autovalores_full, key=abs, reverse=True)],
        })
        st.markdown("<div class='small-muted'>Espectro de P (autovalores):</div>", unsafe_allow_html=True)
        st.dataframe(df_autos, use_container_width=True, hide_index=True)

        st.markdown("<div class='small-muted'>Potências de P — convergência a π:</div>", unsafe_allow_html=True)
        linhas = []
        for k in [1, 2, 5, 10, 20, 50]:
            Pk = np.linalg.matrix_power(P, k)
            tv_k = 0.5 * float(np.max(np.abs(Pk - pi[None, :])))
            linhas.append({"k": k, "max TV(Pᵏ, π)": f"{tv_k:.8f}"})
        st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
    else:
        st.info("Distribuição estacionária não pôde ser calculada para esta cadeia.")


# ----------------------------
# 15) RODAPÉ
# ----------------------------
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>Cadeias de Markov — The Everything Calculator • Fellipe Almässy</div>",
    unsafe_allow_html=True,
)
