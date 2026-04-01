# markov_tab.py
# Aba de Cadeias de Markov — The Everything Calculator
# ------------------------------------------------------------
# - Simulação interativa com animação em tempo real
# - Grafo da cadeia com Plotly (arestas + pesos)
# - Distribuição estacionária (autovetor)
# - Evolução temporal da distribuição
# - Múltiplos exemplos práticos
# - Diagnósticos de convergência
# ------------------------------------------------------------

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ----------------------------
# EXEMPLOS PRÁTICOS
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
# FUNÇÕES DE COMPUTAÇÃO
# ----------------------------
@st.cache_data(show_spinner=False)
def validar_matriz(mat: tuple) -> tuple[bool, str]:
    """Valida se é uma matriz de transição estocástica."""
    P = np.array(mat)
    n = P.shape[0]
    if P.shape != (n, n):
        return False, "Matriz não é quadrada."
    if np.any(P < 0):
        return False, "Existem probabilidades negativas."
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        erros = [f"Linha {i+1}: soma={s:.4f}" for i, s in enumerate(row_sums) if not np.isclose(s, 1.0, atol=1e-6)]
        return False, "Linhas não somam 1: " + ", ".join(erros)
    return True, "OK"


@st.cache_data(show_spinner=False)
def calcular_estacionaria(mat: tuple) -> Optional[np.ndarray]:
    """Calcula a distribuição estacionária via autovetor esquerdo."""
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
    """Computa P^t aplicado à distribuição inicial ao longo de n_passos."""
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
    """Simula uma trajetória da cadeia de Markov."""
    rng = np.random.default_rng(seed)
    P = np.array(mat)
    n = P.shape[0]
    trajetoria = [estado_inicial]
    estado = estado_inicial
    for _ in range(n_passos):
        estado = rng.choice(n, p=P[estado])
        trajetoria.append(estado)
    return trajetoria


# ----------------------------
# GRÁFICOS
# ----------------------------
def make_grafo_markov(estados: list[str], P: np.ndarray, pi: Optional[np.ndarray]) -> go.Figure:
    """Cria grafo da cadeia com layout circular e arestas ponderadas."""
    n = len(estados)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
    raio = 1.0

    xs = raio * np.cos(angulos)
    ys = raio * np.sin(angulos)

    fig = go.Figure()

    # Arestas
    for i in range(n):
        for j in range(n):
            p_ij = P[i, j]
            if p_ij < 0.01:
                continue

            xi, yi = xs[i], ys[i]
            xj, yj = xs[j], yj = xs[j], ys[j]

            if i == j:
                # Self-loop: pequeno arco anotado
                loop_x = xi + 0.28 * np.cos(angulos[i])
                loop_y = yi + 0.28 * np.sin(angulos[i])
                fig.add_trace(go.Scatter(
                    x=[loop_x], y=[loop_y],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=28,
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
                    font=dict(size=9, color="rgba(229,231,235,0.9)"),
                    bgcolor="rgba(14,17,23,0.85)",
                )
            else:
                # Aresta direta com desvio para evitar sobreposição
                dx, dy = xj - xi, yj - yi
                mid_x = (xi + xj) / 2 - dy * 0.18
                mid_y = (yi + yj) / 2 + dx * 0.18

                width = max(1.0, p_ij * 6)
                alpha = 0.3 + 0.6 * p_ij
                color_src = CORES_ESTADOS[i % len(CORES_ESTADOS)]

                # Curva via pontos intermediários
                fig.add_trace(go.Scatter(
                    x=[xi, mid_x, xj],
                    y=[yi, mid_y, yj],
                    mode="lines",
                    line=dict(width=width, color=color_src.replace("#", "rgba(").replace("FF4B4B", "255,75,75") + f",{alpha:.2f})"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

                # Anotação no meio da aresta
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"<b>{p_ij:.2f}</b>",
                    showarrow=False,
                    font=dict(size=9, color="rgba(229,231,235,0.85)"),
                    bgcolor="rgba(14,17,23,0.80)",
                    borderpad=2,
                )

    # Nós (tamanho proporcional à dist. estacionária)
    tamanhos = [45] * n
    if pi is not None:
        tamanhos = [30 + 60 * float(pi[i]) for i in range(n)]

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
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
        hovertemplate=[
            f"<b>{est}</b><br>π = {float(pi[i]):.4f}" if pi is not None else f"<b>{est}</b>"
            for i, est in enumerate(estados)
        ],
    ))

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.6, 1.6], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def make_evolucao_plot(historico: np.ndarray, estados: list[str], pi: Optional[np.ndarray]) -> go.Figure:
    """Evolução da distribuição ao longo do tempo com linhas estacionárias."""
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
    """Plota a trajetória da simulação como step chart."""
    fig = go.Figure()

    n = len(trajetoria)
    cores_traj = [CORES_ESTADOS[s % len(CORES_ESTADOS)] for s in trajetoria]

    fig.add_trace(go.Scatter(
        x=list(range(n)),
        y=trajetoria,
        mode="lines+markers",
        line=dict(color="rgba(30,144,255,0.6)", width=1.5, shape="hv"),
        marker=dict(
            size=6,
            color=cores_traj,
            line=dict(color="rgba(255,255,255,0.3)", width=0.5),
        ),
        name="Trajetória",
        hovertemplate="t=%{x}<br>Estado=%{text}<extra></extra>",
        text=[estados[s] for s in trajetoria],
    ))

    # Frequência empírica
    freq = np.bincount(trajetoria, minlength=len(estados)) / n
    for i, (est, f) in enumerate(zip(estados, freq)):
        fig.add_annotation(
            x=n * 0.98, y=i,
            text=f"  {est}: {f:.3f}",
            showarrow=False,
            font=dict(size=9, color=CORES_ESTADOS[i % len(CORES_ESTADOS)]),
            xanchor="right",
        )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
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
    """Heatmap da matriz de transição."""
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
    """Distância total de variação até a estacionária."""
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
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        title="Convergência: Distância de Variação Total ||μₜ − π||_TV",
        xaxis_title="Passo t",
        yaxis_title="||μₜ − π||_TV",
        yaxis_type="log" if tv[1:].max() > 0 else "linear",
    )
    return fig


# ----------------------------
# EDITOR DE MATRIZ
# ----------------------------
def editor_matriz(estados: list[str], P_default: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Interface para editar estados e matriz de transição."""
    n = len(estados)

    # Editar nomes dos estados
    cols_nomes = st.columns(n)
    novos_estados = []
    for i, col in enumerate(cols_nomes):
        nome = col.text_input(f"Estado {i+1}", value=estados[i], key=f"est_{i}_{id(estados)}", label_visibility="collapsed")
        novos_estados.append(nome if nome.strip() else estados[i])

    # Editar linhas da matriz
    P_nova = np.zeros((n, n))
    st.markdown("<div class='small-muted'>Matriz de transição (linha i → coluna j):</div>", unsafe_allow_html=True)

    cols_header = st.columns([1.2] + [1] * n)
    cols_header[0].markdown("**De \\ Para**")
    for j, est in enumerate(novos_estados):
        cols_header[j + 1].markdown(f"**{est}**")

    for i in range(n):
        cols_row = st.columns([1.2] + [1] * n)
        cols_row[0].markdown(f"**{novos_estados[i]}**")
        soma = 0.0
        vals_row = []
        for j in range(n):
            val = cols_row[j + 1].number_input(
                f"P[{i},{j}]",
                min_value=0.0, max_value=1.0,
                value=float(P_default[i, j]),
                step=0.05, format="%.2f",
                key=f"p_{i}_{j}_{id(estados)}",
                label_visibility="collapsed",
            )
            vals_row.append(val)
            soma += val
        # Normalização automática por linha
        if soma > 0:
            P_nova[i] = [v / soma for v in vals_row]
        else:
            P_nova[i] = [1.0 / n] * n

    return novos_estados, P_nova


# ----------------------------
# ABA PRINCIPAL
# ----------------------------
def render_markov_tab():
    """Renderiza a aba de Cadeias de Markov no TEC."""

    st.markdown("### Cadeias de Markov")
    st.caption("Processos estocásticos com memória sem história")
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ---------- TEORIA COLAPSÁVEL ----------
    with st.expander("Teoria: o que é uma Cadeia de Markov?", expanded=False):
        st.markdown("Uma cadeia de Markov é um processo estocástico com a **propriedade de Markov**:")
        st.latex(r"\Pr(X_{t+1} = j \mid X_t = i,\, X_{t-1},\, \ldots) = \Pr(X_{t+1} = j \mid X_t = i) = P_{ij}")
        st.markdown("O futuro depende **apenas do estado presente**, não da história.")
        st.markdown("---")
        st.markdown("**Matriz de Transição**")
        st.latex(r"P = \begin{bmatrix} P_{11} & P_{12} & \cdots \\ P_{21} & P_{22} & \cdots \\ \vdots & & \ddots \end{bmatrix}, \quad \sum_j P_{ij} = 1 \; \forall i")
        st.markdown("**Distribuição no passo t**")
        st.latex(r"\mu_t = \mu_0 \cdot P^t")
        st.markdown("**Distribuição Estacionária** — satisfaz")
        st.latex(r"\pi = \pi P \quad \Longleftrightarrow \quad \pi \text{ é autovetor esquerdo de } P \text{ com autovalor } 1")
        st.markdown("**Convergência** — para cadeias ergódicas:")
        st.latex(r"\lim_{t \to \infty} \mu_t = \pi \quad \text{(independente de } \mu_0\text{)}")
        st.markdown("A velocidade de convergência é ditada pelo **segundo maior autovalor em módulo** (spectral gap).")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ---------- SIDEBAR: CONTROLES ----------
    st.sidebar.markdown("---")
    st.sidebar.header("Markov — Controles")

    exemplo_nome = st.sidebar.selectbox(
        "Exemplo",
        list(EXEMPLOS_MARKOV.keys()),
        index=0,
    )
    exemplo = EXEMPLOS_MARKOV[exemplo_nome]

    # Para o modo personalizado, usa editor; para os outros, mostra só a matriz
    if exemplo_nome == "Personalizado":
        n_estados = st.sidebar.slider("Nº de estados", 2, 5, 3, key="n_estados_markov")
        estados_default = [chr(65 + i) for i in range(n_estados)]
        P_default = np.eye(n_estados) * 0.5 + np.ones((n_estados, n_estados)) * (0.5 / n_estados)
        P_default /= P_default.sum(axis=1, keepdims=True)
        estados_edit = estados_default
        P_edit = P_default
    else:
        estados_edit = exemplo["estados"]
        P_edit = exemplo["matriz"]
        n_estados = len(estados_edit)

    n_passos_evolucao = st.sidebar.slider("Passos de evolução", 10, 200, 60, step=5, key="passos_evolucao_markov")
    n_passos_sim = st.sidebar.slider("Passos da simulação", 50, 2000, 300, step=50, key="passos_sim_markov")
    seed_sim = st.sidebar.number_input("Seed da simulação", value=42, min_value=0, max_value=9999, step=1, key="seed_markov")

    # Distribuição inicial
    st.sidebar.markdown("**Distribuição Inicial μ₀**")
    dist_cols = st.sidebar.columns(n_estados)
    dist_inicial_raw = []
    for i, col in enumerate(dist_cols):
        v = col.number_input(
            estados_edit[i],
            min_value=0.0, max_value=1.0,
            value=round(1.0 / n_estados, 2),
            step=0.1, format="%.2f",
            key=f"dist0_{i}_markov",
        )
        dist_inicial_raw.append(v)
    s = sum(dist_inicial_raw)
    if s > 0:
        dist_inicial = tuple(v / s for v in dist_inicial_raw)
    else:
        dist_inicial = tuple(1.0 / n_estados for _ in range(n_estados))
    st.sidebar.caption(f"μ₀ normalizado: {[f'{v:.3f}' for v in dist_inicial]}")

    # Estado inicial para simulação
    estado_inicial_nome = st.sidebar.selectbox("Estado inicial (simulação)", estados_edit, key="est_ini_markov")
    estado_inicial_idx = estados_edit.index(estado_inicial_nome)

    # ---------- EDITOR (modo personalizado) ----------
    if exemplo_nome == "Personalizado":
        st.markdown("#### Definir Cadeia")
        estados_edit, P_edit = editor_matriz(estados_edit, P_edit)
    else:
        st.markdown(f"**{exemplo['emoji']} {exemplo_nome}** — {exemplo['descricao']}")

    P = P_edit
    estados = estados_edit
    mat_tuple = tuple(map(tuple, P))

    # ---------- VALIDAÇÃO ----------
    valido, msg = validar_matriz(mat_tuple)
    if not valido:
        st.error(f"Matriz inválida: {msg}")
        return

    # ---------- COMPUTAÇÕES ----------
    pi = calcular_estacionaria(mat_tuple)
    historico = evolucao_distribuicao(mat_tuple, dist_inicial, n_passos_evolucao)
    trajetoria = simular_cadeia(mat_tuple, estado_inicial_idx, n_passos_sim, int(seed_sim))

    # ---------- MÉTRICAS PRINCIPAIS ----------
    n_est = len(estados)
    # Autovalores para spectral gap
    autovalores = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    lambda2 = autovalores[1] if len(autovalores) > 1 else 0.0
    spectral_gap = 1 - float(lambda2)
    mixing_time_est = int(np.ceil(1.0 / spectral_gap)) if spectral_gap > 0 else 9999

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Nº de estados", n_est)
    mc2.metric("λ₂ (2º autovalor)", f"{float(lambda2):.4f}", "espectral")
    mc3.metric("Spectral gap", f"{spectral_gap:.4f}", "1 − |λ₂|")
    mc4.metric("Mixing time (est.)", f"~{mixing_time_est} passos", "1/(1−|λ₂|)")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ---------- DISTRIBUIÇÃO ESTACIONÁRIA ----------
    if pi is not None:
        st.markdown("**Distribuição Estacionária π**")
        cols_pi = st.columns(n_est)
        for i, col in enumerate(cols_pi):
            col.metric(estados[i], f"{float(pi[i]):.4f}", f"{float(pi[i])*100:.1f}%")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ---------- ABAS INTERNAS ----------
    tab_grafo, tab_evolucao, tab_sim, tab_diag = st.tabs([
        "🔵 Grafo da Cadeia",
        "📊 Evolução Temporal",
        "🎲 Simulação",
        "🔬 Diagnósticos",
    ])

    with tab_grafo:
        col_g, col_h = st.columns([1.2, 1])
        with col_g:
            fig_grafo = make_grafo_markov(estados, P, pi)
            st.plotly_chart(fig_grafo, use_container_width=True)
        with col_h:
            fig_hm = make_heatmap_transicao(P, estados)
            st.plotly_chart(fig_hm, use_container_width=True)

            # Tabela da matriz
            df_P = pd.DataFrame(P, index=estados, columns=estados)
            st.markdown("<div class='small-muted'>Matriz P (numérica):</div>", unsafe_allow_html=True)
            st.dataframe(df_P.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn", axis=None, vmin=0, vmax=1
            ), use_container_width=True)

    with tab_evolucao:
        fig_ev = make_evolucao_plot(historico, estados, pi)
        st.plotly_chart(fig_ev, use_container_width=True)

        if pi is not None:
            # Distância TV no último passo
            tv_final = 0.5 * np.sum(np.abs(historico[-1] - pi))
            st.markdown(
                f"<div class='small-muted'>Distância TV após {n_passos_evolucao} passos: <b>{tv_final:.6f}</b></div>",
                unsafe_allow_html=True
            )

        # Tabela do histórico (amostras)
        df_hist = pd.DataFrame(
            historico[::max(1, n_passos_evolucao // 20)],
            columns=estados
        )
        df_hist.index.name = "Passo"
        st.markdown("<div class='small-muted'>Amostras da distribuição μₜ:</div>", unsafe_allow_html=True)
        st.dataframe(df_hist.style.format("{:.6f}"), use_container_width=True)

    with tab_sim:
        fig_traj = make_trajetoria_plot(trajetoria, estados)
        st.plotly_chart(fig_traj, use_container_width=True)

        # Frequência empírica vs estacionária
        freq_emp = np.bincount(trajetoria, minlength=n_est) / len(trajetoria)
        df_freq = pd.DataFrame({
            "Estado": estados,
            "Freq. Empírica": [f"{v:.4f}" for v in freq_emp],
            "π (Teórica)": [f"{float(pi[i]):.4f}" for i in range(n_est)] if pi is not None else ["—"] * n_est,
            "Erro |emp − π|": [f"{abs(freq_emp[i] - float(pi[i])):.4f}" for i in range(n_est)] if pi is not None else ["—"] * n_est,
        })
        st.markdown("<div class='small-muted'>Frequência empírica vs distribuição estacionária:</div>", unsafe_allow_html=True)
        st.dataframe(df_freq, use_container_width=True, hide_index=True)

        st.caption(f"Trajetória: {n_passos_sim} passos | Seed: {int(seed_sim)} | Estado inicial: {estados[estado_inicial_idx]}")

    with tab_diag:
        if pi is not None:
            fig_conv = make_convergencia_markov(historico, pi)
            st.plotly_chart(fig_conv, use_container_width=True)

            # Autovalores
            autovalores_completos = np.linalg.eigvals(P)
            df_autos = pd.DataFrame({
                "Autovalor (Re)": [f"{v.real:.6f}" for v in np.sort(autovalores_completos)[::-1]],
                "Autovalor (Im)": [f"{v.imag:.6f}" for v in np.sort(autovalores_completos)[::-1]],
                "|λ|": [f"{abs(v):.6f}" for v in sorted(autovalores_completos, key=abs, reverse=True)],
            })
            st.markdown("<div class='small-muted'>Espectro de P (autovalores):</div>", unsafe_allow_html=True)
            st.dataframe(df_autos, use_container_width=True, hide_index=True)

            # P^k para k crescente
            st.markdown("<div class='small-muted'>Potências de P (convergência a π):</div>", unsafe_allow_html=True)
            ks = [1, 2, 5, 10, 20, 50]
            Pk = np.eye(n_est)
            linhas = []
            for k in ks:
                potencia = np.linalg.matrix_power(P, k)
                tv_k = 0.5 * np.max(np.abs(potencia - pi[None, :]))
                linhas.append({"k": k, "max TV(P^k, π)": f"{tv_k:.8f}"})
            st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
        else:
            st.info("Distribuição estacionária não pôde ser calculada para esta cadeia.")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footer'>Cadeias de Markov — The Everything Calculator • Fellipe Almässy</div>",
        unsafe_allow_html=True
    )
