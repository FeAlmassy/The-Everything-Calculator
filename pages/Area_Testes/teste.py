import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# ──────────────────────────────────────────────
# Função MMQ
# ──────────────────────────────────────────────
def mmq(x, y, g):
    """
    Calcula os coeficientes do polinômio de grau g que melhor se ajusta
    aos pontos (x, y) pelo Método dos Mínimos Quadrados.

    Retorna coeficientes em ordem decrescente: [aₘ, ..., a₁, a₀]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    lista_potencias = [np.sum(x**k) for k in range(2 * g + 1)]
    array_potencias = np.array(lista_potencias[::-1])

    matriz_e = np.array(
        [array_potencias[i : i + g + 1] for i in range(g + 1)],
        dtype=np.float64,
    )
    matriz_d = np.array(
        [(x**i * y).sum() for i in range(g, -1, -1)],
        dtype=np.float64,
    ).reshape(-1, 1)

    solucao = np.linalg.solve(matriz_e, matriz_d)
    return solucao.flatten()


def avaliar_polinomio(coefs, x_vals):
    """Avalia o polinômio com coeficientes em ordem decrescente."""
    return np.polyval(coefs, x_vals)


# Letras para nomear coeficientes: a, b, c, ..., z, z1, z2, ...
def _nome_coef(i):
    letras = "abcdefghijklmnopqrstuvwxyz"
    if i < len(letras):
        return letras[i]
    return f"z{i - len(letras) + 1}"


_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

def _superscript(n):
    return str(n).translate(_SUPERSCRIPT)


def formatar_polinomio(coefs):
    """Retorna a string do polinômio formatada com coeficientes a, b, c, ..."""
    g = len(coefs) - 1
    termos = []
    for i, c in enumerate(coefs):
        exp = g - i
        c_r = round(float(c), 6)
        if abs(c_r) < 1e-10:
            continue
        sinal = "+" if c_r >= 0 else "-"
        valor = abs(c_r)
        if exp == 0:
            termos.append(f"{sinal} {valor:.4f}")
        elif exp == 1:
            termos.append(f"{sinal} {valor:.4f}x")
        else:
            termos.append(f"{sinal} {valor:.4f}x{_superscript(exp)}")
    if not termos:
        return "P(x) = 0"
    poly_str = " ".join(termos).lstrip("+ ").strip()
    return f"P(x) = {poly_str}"


def calcular_r2(y_obs, y_pred):
    """Calcula o coeficiente de determinação R²."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


# ──────────────────────────────────────────────
# Layout
# ──────────────────────────────────────────────
st.set_page_config(page_title="MMQ Explorer", layout="wide")
st.title("Método dos Mínimos Quadrados — Explorador Interativo")
st.markdown(
    "Insira os dados, escolha o grau do polinômio e visualize o ajuste em tempo real."
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")

    st.subheader("Dados de entrada")

    PRESETS = {
        "Ex. 1": [(1, 3), (2, 8), (3, 1), (4, 12), (5, 2), (6, 10), (7, 5), (8, 11), (9, 4)],
        "Ex. 2": [(1, 2),  (2, 10), (3, 30), (4, 68)],
        "Ex. 3": [(1, 3),  (2, 5),  (3, 7),  (4, 9),  (5, 11)],
    }

    dados_preset = st.selectbox(
        "Carregar dados de exemplo",
        options=[*PRESETS.keys(), "Personalizado"],
    )

    # Inicializa session_state com os valores do preset selecionado
    PERSONALIZADO_DEFAULT = [(1, 6), (2, 12), (3, 20)]
    preset_vals = PRESETS.get(dados_preset, PERSONALIZADO_DEFAULT)

    chave_preset = f"preset_{dados_preset}"
    if st.session_state.get("_ultimo_preset") != chave_preset:
        st.session_state["_ultimo_preset"] = chave_preset
        st.session_state["pontos"] = [{"x": float(x), "y": float(y)} for x, y in preset_vals]

    if "pontos" not in st.session_state:
        st.session_state["pontos"] = [{"x": float(x), "y": float(y)} for x, y in preset_vals]

    # Botão para adicionar linha
    if st.button("＋ Adicionar ponto") and len(st.session_state["pontos"]) < 20:
        ultimo = st.session_state["pontos"][-1]
        st.session_state["pontos"].append({"x": ultimo["x"] + 1.0, "y": 0.0})
        st.rerun()

    # Linhas de dados com botão de remoção
    st.markdown("**Insira os pontos (x , y):**")
    remover_idx = None
    for i, ponto in enumerate(st.session_state["pontos"]):
        col_x, col_y, col_del = st.columns([2, 2, 1])
        ponto["x"] = col_x.number_input(
            f"x{i+1}", value=ponto["x"], step=1.0,
            key=f"x_{i}", label_visibility="collapsed",
        )
        ponto["y"] = col_y.number_input(
            f"y{i+1}", value=ponto["y"], step=1.0,
            key=f"y_{i}", label_visibility="collapsed",
        )
        if col_del.button("✕", key=f"del_{i}") and len(st.session_state["pontos"]) > 2:
            remover_idx = i

    if remover_idx is not None:
        st.session_state["pontos"].pop(remover_idx)
        # Limpa o cache dos widgets de input para que os valores
        # sejam relidos de st.session_state["pontos"] no próximo render
        for k in list(st.session_state.keys()):
            if k.startswith(("x_", "y_", "del_")):
                del st.session_state[k]
        st.rerun()

    pontos_x = [p["x"] for p in st.session_state["pontos"]]
    pontos_y = [p["y"] for p in st.session_state["pontos"]]

    st.subheader("Polinômio")
    grau = st.slider("Grau do polinômio (g)", min_value=1, max_value=8, value=1)

    st.subheader("Visualização")
    mostrar_residuos = st.checkbox("Mostrar resíduos", value=True)
    mostrar_grade    = st.checkbox("Mostrar grade", value=False)
    n_pontos_curva   = st.slider("Resolução da curva", min_value=50, max_value=500, value=200)


# ──────────────────────────────────────────────
# Dados como arrays
# ──────────────────────────────────────────────
x_pts = np.array(pontos_x, dtype=np.float64)
y_pts = np.array(pontos_y, dtype=np.float64)

n_pontos = len(x_pts)

if n_pontos < 2:
    st.error("Insira ao menos 2 pontos para realizar o ajuste.")
    st.stop()




# ──────────────────────────────────────────────
# Cálculo
# ──────────────────────────────────────────────
try:
    coefs = mmq(x_pts, y_pts, grau)
except np.linalg.LinAlgError:
    st.error("Sistema singular — tente um grau menor ou pontos com valores x distintos.")
    st.stop()

x_curva  = np.linspace(x_pts.min(), x_pts.max(), n_pontos_curva)
y_curva  = avaliar_polinomio(coefs, x_curva)
y_ajuste = avaliar_polinomio(coefs, x_pts)
residuos = y_pts - y_ajuste
r2       = calcular_r2(y_pts, y_ajuste)


# ──────────────────────────────────────────────
# Gráfico principal
# ──────────────────────────────────────────────
col_graf, col_info = st.columns([3, 1])

with col_graf:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_pts, y_pts, color="steelblue", s=70, zorder=5, label="Dados observados")
    ax.plot(x_curva, y_curva, color="tomato", linewidth=2, label=f"Ajuste grau {grau}")

    if mostrar_residuos:
        for xi, yi, yr in zip(x_pts, y_pts, y_ajuste):
            ax.plot([xi, xi], [yi, yr], color="gray", linewidth=1,
                    linestyle="--", alpha=0.7)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ajuste por Mínimos Quadrados")
    ax.legend()
    if mostrar_grade:
        ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    st.pyplot(fig)
    plt.close(fig)

    # Polinômio e coeficientes abaixo do gráfico
    st.markdown("**Polinômio ajustado:**")
    st.code(formatar_polinomio(coefs), language="")

    st.markdown("**Coeficientes** (maior → menor grau):")
    cols_coef = st.columns(min(len(coefs), 6))
    for i, c in enumerate(coefs):
        cols_coef[i % len(cols_coef)].metric(_nome_coef(i), f"{c:.1f}")


# ──────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────
with col_info:
    st.subheader("Resultados")

    st.metric("R²", f"{r2:.6f}")
    st.metric("Resíduo máx. |e|", f"{np.max(np.abs(residuos)):.4f}")
    st.metric("RMSE", f"{np.sqrt(np.mean(residuos**2)):.4f}")


# ──────────────────────────────────────────────
# Matrizes do sistema
# ──────────────────────────────────────────────
with st.expander("Matrizes do sistema normal (E e D)"):
    lista_pot = [np.sum(x_pts**k) for k in range(2 * grau + 1)]
    arr_pot   = np.array(lista_pot[::-1])
    mat_e     = np.array([arr_pot[i : i + grau + 1] for i in range(grau + 1)], dtype=np.float64)
    mat_d     = np.array([(x_pts**i * y_pts).sum() for i in range(grau, -1, -1)],
                         dtype=np.float64).reshape(-1, 1)

    col_e, col_d = st.columns(2)
    with col_e:
        st.markdown("**Matriz E** (coeficientes)")
        st.dataframe(
            pd.DataFrame(mat_e,
                         columns=[f"a{grau - j}" for j in range(grau + 1)],
                         index=[f"eq{i}" for i in range(grau + 1)]),
            use_container_width=True,
        )
    with col_d:
        st.markdown("**Vetor D** (independente)")
        st.dataframe(
            pd.DataFrame(mat_d, columns=["valor"],
                         index=[f"eq{i}" for i in range(grau + 1)]),
            use_container_width=True,
        )
