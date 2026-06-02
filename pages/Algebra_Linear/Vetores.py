# Mecanismo de Classificação Evolutiva — Modelo CredituS (Streamlit + Plotly)
# ------------------------------------------------------------
# - Algoritmo genético para classificar Adimplência (A) x Inadimplência (I)
# - Cromossomo = Eneadecaquatérnio Q (bias b0 + 18 genes)
# - Fitness FC = PA * PI (produto força acerto balanceado nas duas classes)
# - Seleção por roleta, crossover multi-ponto, mutação, substituição elitista
# - Geração de dados sintéticos no padrão CredituS OU upload de planilha (.xlsx)
# - Diagnósticos: convergência + matriz de confusão + distribuição de Q + roleta

import io
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation


# ----------------------------
# 0) CONFIGURAÇÃO DA PÁGINA (DEVE SER A PRIMEIRA)
# ----------------------------
st.set_page_config(page_title="Algoritmo Genético", layout="wide")


# ----------------------------
# 1) ESTILO (CSS) - INCLUINDO TIPOGRAFIA WEB
# ----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');

:root {
  --bg: #09090b;
  --border: rgba(255, 255, 255, 0.05);
  --muted: #a1a1aa;
  --muted2: #52525b;
  --accent: #ef4444;
  --accent2: #3b82f6;
  --green: #10b981;
}

/* Tipografia SaaS: Inter para texto, Fira Code para dados e métricas */
html, body, [class*="css"], .stMarkdown { 
    font-family: 'Inter', sans-serif !important; 
}
div[data-testid="stMetricValue"], .stDataFrame { 
    font-family: 'Fira Code', monospace !important; 
}

.main { background-color: var(--bg); }
section[data-testid="stSidebar"] { background-color: #0b1020; border-right: 1px solid var(--border); }
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}

/* Empty State Card */
.empty-state {
    border: 1px dashed var(--muted2);
    border-radius: 12px;
    padding: 4rem 2rem;
    text-align: center;
    background-color: rgba(255,255,255,0.01);
    margin-top: 2rem;
}
.empty-state h3 { font-weight: 500; color: #f4f4f5; font-family: 'Inter', sans-serif; }
.empty-state p { color: var(--muted); font-size: 0.95rem; }

.small-muted { color: var(--muted); font-size: 0.92rem; margin-bottom: 1rem; }
.badge {
  display:inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
  color: #d4d4d8; font-size: 0.82rem; margin-right: 0.4rem; font-family: 'Fira Code', monospace;
}
.footer { text-align:center; color: var(--muted2); margin-top: 3rem; font-size: 0.85rem; }

.function-display {
    text-align: center;
    padding: 2rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------
# 2) MOTOR GENÉTICO (ENGINE)
# ----------------------------
def criar_cromossomos(qtd_cromossomos: int, qtd_genes: int) -> np.ndarray:
    return -1 + 2 * np.random.rand(qtd_cromossomos, qtd_genes)

def calcular_fitness(cromossomos: np.ndarray, array_dados_clientes: np.ndarray, array_gabarito: np.ndarray) -> np.ndarray:
    total_adimplentes = np.sum(array_gabarito == 1)
    total_inadimplentes = np.sum(array_gabarito == 0)

    fitnesses = []
    for linha in cromossomos:
        bias = linha[0]
        genes = linha[1:]

        q = np.dot(array_dados_clientes, genes) + bias
        vetor_hipotese = np.where(q >= 0, 1, 0)

        acertos_adimplentes = np.sum((vetor_hipotese == 1) & (array_gabarito == 1))
        acertos_inadimplentes = np.sum((vetor_hipotese == 0) & (array_gabarito == 0))

        pa = acertos_adimplentes / total_adimplentes if total_adimplentes else 0.0
        pi = acertos_inadimplentes / total_inadimplentes if total_inadimplentes else 0.0

        fitnesses.append(pa * pi)

    return np.array(fitnesses, dtype=float)

def fitness_percentual(vetor_fitnesses: np.ndarray) -> np.ndarray:
    soma = np.sum(vetor_fitnesses)
    if soma == 0:
        return np.ones(len(vetor_fitnesses)) / len(vetor_fitnesses)
    return vetor_fitnesses / soma

def selecionar_pais_roleta(cromossomos: np.ndarray, percentual_fitnesses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    roleta_acumulada = np.cumsum(percentual_fitnesses)
    indice_pai = min(int(np.searchsorted(roleta_acumulada, np.random.rand())), len(cromossomos) - 1)
    indice_mae = min(int(np.searchsorted(roleta_acumulada, np.random.rand())), len(cromossomos) - 1)
    return cromossomos[indice_pai], cromossomos[indice_mae]

def cruzar_pais(pai: np.ndarray, mae: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c1 = np.random.randint(1, len(pai))
    c2 = np.random.randint(1, len(pai))
    c3 = np.random.randint(1, len(pai))

    filho1 = np.concatenate([pai[:c1], mae[c1:]])
    filho2 = np.concatenate([pai[:c2], mae[c2:]])
    filho3 = np.concatenate([pai[:c3], mae[c3:]])

    return filho1, filho2, filho3

def mutar(filho1: np.ndarray, filho2: np.ndarray, filho3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for filho in [filho1, filho2, filho3]:
        indice = np.random.randint(0, len(filho))
        filho[indice] = -1 + 2 * np.random.rand()
    return filho1, filho2, filho3

def algoritmo_genetico(
    array_dados_clientes: np.ndarray, array_gabarito: np.ndarray,
    qtd_cromossomos: int = 12, geracoes: int = 500, fitness_alvo: float = 0.90,
    cruzamentos_por_geracao: int = 3, seed: int = 42,
) -> dict:
    np.random.seed(seed)
    qtd_genes = array_dados_clientes.shape[1] + 1
    populacao = criar_cromossomos(qtd_cromossomos, qtd_genes)
    fitnesses = calcular_fitness(populacao, array_dados_clientes, array_gabarito)

    melhor_cromossomo = populacao[int(np.argmax(fitnesses))].copy()
    melhor_fitness = float(np.max(fitnesses))

    hist_melhor = []
    hist_media = []
    geracao_convergencia = geracoes

    for geracao in range(geracoes):
        idx_melhor = int(np.argmax(fitnesses))
        if fitnesses[idx_melhor] > melhor_fitness:
            melhor_fitness = float(fitnesses[idx_melhor])
            melhor_cromossomo = populacao[idx_melhor].copy()

        hist_melhor.append(melhor_fitness)
        hist_media.append(float(np.mean(fitnesses)))

        if melhor_fitness >= fitness_alvo:
            geracao_convergencia = geracao + 1
            break

        percentuais = fitness_percentual(fitnesses)
        prole = []
        for _ in range(cruzamentos_por_geracao):
            pai, mae = selecionar_pais_roleta(populacao, percentuais)
            filho1, filho2, filho3 = cruzar_pais(pai, mae)
            filho1, filho2, filho3 = mutar(filho1, filho2, filho3)
            prole.extend([filho1, filho2, filho3])

        prole = np.array(prole)
        fitness_prole = calcular_fitness(prole, array_dados_clientes, array_gabarito)

        todos = np.vstack([populacao, prole])
        fitness_todos = np.concatenate([fitnesses, fitness_prole])
        ordem = np.argsort(fitness_todos)[::-1][:qtd_cromossomos]
        populacao = todos[ordem]
        fitnesses = fitness_todos[ordem]

    return {
        "melhor_cromossomo": melhor_cromossomo,
        "melhor_fitness": float(max(melhor_fitness, 0.0)),
        "hist_melhor": np.array(hist_melhor, dtype=float),
        "hist_media": np.array(hist_media, dtype=float),
        "geracao_convergencia": int(geracao_convergencia),
        "populacao_final": populacao,
        "fitness_final": fitnesses,
    }

def diagnosticar(cromossomo: np.ndarray, array_dados_clientes: np.ndarray, array_gabarito: np.ndarray) -> dict:
    bias = cromossomo[0]
    genes = cromossomo[1:]

    q = np.dot(array_dados_clientes, genes) + bias
    hipotese = np.where(q >= 0, 1, 0)

    ta = int(np.sum(array_gabarito == 1))
    ti = int(np.sum(array_gabarito == 0))
    nac = int(np.sum((hipotese == 1) & (array_gabarito == 1)))
    nic = int(np.sum((hipotese == 0) & (array_gabarito == 0)))

    pa = nac / ta if ta else 0.0
    pi = nic / ti if ti else 0.0

    return {
        "q": q, "hipotese": hipotese, "pa": pa, "pi": pi, "fc": pa * pi,
        "acuracia": (nac + nic) / len(array_gabarito),
        "nac": nac, "nic": nic, "ta": ta, "ti": ti,
        "fa": ta - nac, "fi": ti - nic,
    }


# ----------------------------
# 3) DADOS E UTILITÁRIOS
# ----------------------------
def nomes_colunas() -> list[str]:
    nomes = []
    for a in range(1, 10):
        nomes.append(f"X{a}1")
        nomes.append(f"X{a}2")
    return nomes

@st.cache_data(show_spinner=False)
def gerar_dados_sinteticos(n_clientes: int, ruido: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_clientes, 18), dtype=int)
    for i in range(9):
        nuance1 = rng.integers(0, 2, size=n_clientes)
        X[:, 2 * i] = nuance1
        X[:, 2 * i + 1] = 1 - nuance1

    pesos_true = rng.uniform(-1, 1, size=18)
    bias_true = rng.uniform(-1, 1)
    q_true = X @ pesos_true + bias_true
    rotulos = (q_true >= 0).astype(int)

    if ruido > 0:
        flip = rng.random(n_clientes) < ruido
        rotulos[flip] = 1 - rotulos[flip]

    return X.astype(float), rotulos.astype(int)

def normalizar_gabarito(coluna: pd.Series) -> np.ndarray:
    if coluna.dtype == object:
        return coluna.map(lambda v: 1 if str(v).strip().upper() in {"A", "1", "ADIMPLENTE"} else 0).to_numpy()
    return (coluna.to_numpy() >= 1).astype(int)

@st.cache_data(show_spinner=False)
def construir_template_xlsx() -> bytes:
    NAVY, LIGHT, GREEN, RED = "1F2A44", "EEF1F7", "DDF3E6", "F8E0E0"
    wb = Workbook()
    head_font = Font(name="Inter", bold=True, color="FFFFFF", size=11)
    head_fill = PatternFill("solid", fgColor=NAVY)
    body_font = Font(name="Inter", size=10)
    center = Alignment(horizontal="center", vertical="center")
    thin = Side(style="thin", color="C9D2E0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws = wb.active
    ws.title = "Dados"
    cols_x = []
    for a in range(1, 10): cols_x += [f"X{a}1", f"X{a}2"]
    headers = ["ID"] + cols_x + ["Classe (A/I)"]
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.font, cell.fill, cell.alignment, cell.border = head_font, head_fill, center, border

    exemplos = [
        [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, "A"],
        [2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, "A"],
        [3, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, "I"],
        [4, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, "I"],
        [5, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, "I"],
    ]
    for row in exemplos: ws.append(row)
    for r in range(2, 2 + len(exemplos)):
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=r, column=c)
            cell.font, cell.alignment, cell.border = body_font, center, border
            if c == len(headers):
                cell.fill = PatternFill("solid", fgColor=GREEN if cell.value == "A" else RED)

    dv01 = DataValidation(type="list", formula1='"0,1"', allow_blank=True, showErrorMessage=True, error="Use apenas 0 ou 1.")
    ws.add_data_validation(dv01)
    dv01.add("B2:S1000")
    dvAI = DataValidation(type="list", formula1='"A,I"', allow_blank=True, showErrorMessage=True, error="Use A (Adimplente) ou I (Inadimplente).")
    ws.add_data_validation(dvAI)
    dvAI.add("T2:T1000")

    ws.column_dimensions["A"].width = 6
    for i in range(18): ws.column_dimensions[chr(ord("B") + i)].width = 6
    ws.column_dimensions["T"].width = 14
    ws.freeze_panes = "A2"

    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


# ----------------------------
# 4) PAINEL TEÓRICO
# ----------------------------
def theory_panel():
    st.markdown("### Teoria Explicada")
    st.markdown(
        "<span class='badge'>Eneadecaquatérnio</span> "
        "<span class='badge'>Fitness Produto</span> "
        "<span class='badge'>Seleção por Roleta</span> "
        "<span class='badge'>Crossover & Mutação</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    with st.expander("Abrir teoria completa (cromossomo, fitness, roleta, evolução)", expanded=False):
        st.markdown("Este motor resolve o problema **CredituS**: evoluir um classificador que separe **Adimplentes (A)** de **Inadimplentes (I)**.")
        st.write("")
        st.markdown("### 1) O Cromossomo: Eneadecaquatérnio $Q$")
        st.latex(r"Q = b_0 + b_{11}X_{11} + b_{12}X_{12} + \cdots + b_{91}X_{91} + b_{92}X_{92}")
        st.latex(r"Q \geq 0 \;\Rightarrow\; \text{Adimplente (A)} \qquad Q < 0 \;\Rightarrow\; \text{Inadimplente (I)}")
        st.write("")
        st.markdown("### 2) Fitness do Cromossomo")
        st.latex(r"FC = PA \cdot PI = \frac{NAC}{TA}\cdot\frac{NIC}{TI}")
        st.markdown("O produto é proposital: só obtém fitness alto quem acerta as duas classes.")
        st.write("")
        st.markdown("### 3) Convergência e Mutação")
        st.markdown("O esquema elitista garante que o melhor da prole e dos pais avance. Converge quando $FC \to 1$.")


# ----------------------------
# 5) GRÁFICOS
# ----------------------------
def make_convergence_plot(hist_melhor: np.ndarray, hist_media: np.ndarray) -> go.Figure:
    geracoes = np.arange(1, len(hist_melhor) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=geracoes, y=hist_media, mode="lines", name="Fitness médio da população",
        line=dict(color="#3b82f6", width=2, dash="dot"), hovertemplate="geração=%{x}<br>média=%{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=geracoes, y=hist_melhor, mode="lines", name="Melhor fitness (elite)",
        line=dict(color="#ef4444", width=3), hovertemplate="geração=%{x}<br>melhor=%{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20),
        title="Evolução do Fitness por Geração", xaxis_title="Geração", yaxis_title="Fitness (FC = PA · PI)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), transition=dict(duration=450),
        font=dict(family="Inter")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", range=[0, 1.02])
    return fig

def make_chromosome_plot(cromossomo: np.ndarray) -> go.Figure:
    rotulos = ["b<sub>0</sub>"]
    for a in range(1, 10):
        rotulos.append(f"b<sub>{a}1</sub>")
        rotulos.append(f"b<sub>{a}2</sub>")
    rotulos = rotulos[:len(cromossomo)]
    cores = ["#10b981" if v >= 0 else "#ef4444" for v in cromossomo]
    fig = go.Figure(go.Bar(
        x=rotulos, y=cromossomo, marker=dict(color=cores, line=dict(color="rgba(255,255,255,0.25)", width=0.5)),
        hovertemplate="gene=%{x}<br>peso=%{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color="rgba(229,231,235,0.45)")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20),
        title="Melhor Cromossomo — Eneadecaquatérnio Treinado", xaxis_title="Genes", yaxis_title="Valor do gene", transition=dict(duration=450), font=dict(family="Inter")
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig

def make_q_distribution_plot(q: np.ndarray, hipotese: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=q[hipotese == 1], name="Classificado A (Q ≥ 0)", marker_color="#10b981", opacity=0.7))
    fig.add_trace(go.Histogram(x=q[hipotese == 0], name="Classificado I (Q < 0)", marker_color="#ef4444", opacity=0.7))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="rgba(229,231,235,0.85)", annotation_text="Threshold = 0", annotation_position="top")
    fig.update_layout(
        barmode="overlay", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20),
        title="Distribuição dos Valores de Q", xaxis_title="Q (eneadecaquatérnio)", yaxis_title="Nº de clientes",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), font=dict(family="Inter")
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
    return fig

def make_confusion_plot(diag: dict) -> go.Figure:
    z = [[diag["nac"], diag["fa"]], [diag["fi"], diag["nic"]]]
    texto = [[f"NAC<br>{diag['nac']}", f"erro<br>{diag['fa']}"], [f"erro<br>{diag['fi']}", f"NIC<br>{diag['nic']}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predito A", "Predito I"], y=["Real A", "Real I"], text=texto, texttemplate="%{text}",
        colorscale=[[0, "#09090b"], [1, "#ef4444"]], showscale=False, hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20), title="Matriz de Confusão", font=dict(family="Inter"))
    return fig

def make_roleta_plot(fitness_final: np.ndarray) -> go.Figure:
    pct = fitness_percentual(fitness_final)
    rotulos = [f"C{i + 1}" for i in range(len(fitness_final))]
    fig = go.Figure(go.Pie(
        labels=rotulos, values=pct, hole=0.45, textinfo="label+percent", marker=dict(line=dict(color="#09090b", width=2)), hovertemplate="%{label}<br>fatia=%{percent}<extra></extra>",
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=60, b=20), title="Roda de Seleção", showlegend=False, font=dict(family="Inter"))
    return fig


# ----------------------------
# 6) CABEÇALHO
# ----------------------------
st.title("Algoritmo Genético")
st.caption("Classificador Evolutivo CredituS — Adimplência (A) × Inadimplência (I)")
st.write("")

theory_panel()
st.write("")


# ----------------------------
# 7) SIDEBAR (BARRA LATERAL COM TOOLTIPS)
# ----------------------------
st.sidebar.header("Controles")

st.sidebar.download_button(
    "⬇ Baixar template (.xlsx)", data=construir_template_xlsx(), file_name="CredituS_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True,
    help="Utilize esta planilha estruturada para inserir seus dados reais e fazer o upload."
)

st.sidebar.divider()

fonte = st.sidebar.radio("Fonte de dados", ["Gerar dados sintéticos", "Enviar planilha (.xlsx)"], index=0)

dados_ok = False
features = None
gabarito = None
nomes = nomes_colunas()

if fonte.startswith("Enviar"):
    up = st.sidebar.file_uploader(
        "Planilha de clientes", type=["xlsx", "xls"],
        help="A primeira coluna é ignorada (ID). Da 2ª à 19ª são as variáveis binárias X. A última coluna é o gabarito."
    )
    if up is not None:
        try:
            df = pd.read_excel(up)
            df_features = df.iloc[:, 1:-1].apply(pd.to_numeric, errors="coerce")
            serie_gab = df.iloc[:, -1]
            mascara = df_features.notna().all(axis=1) & serie_gab.notna()
            n_descartadas = int((~mascara).sum())
            df_features = df_features[mascara]
            serie_gab = serie_gab[mascara]

            features = df_features.to_numpy(dtype=float)
            gabarito = normalizar_gabarito(serie_gab)
            nomes = list(df.iloc[:, 1:-1].columns.astype(str))
            dados_ok = len(gabarito) > 0
            
            if n_descartadas: st.sidebar.warning(f"{n_descartadas} linha(s) incompleta(s) descartada(s).")
            if not dados_ok: st.sidebar.error("Nenhuma linha completa encontrada.")
        except Exception as e:
            st.sidebar.error(f"Falha ao ler a planilha: {e}")
else:
    n_clientes = st.sidebar.slider("Nº de clientes", 50, 1000, 300, step=50, help="Tamanho da população simulada para treino.")
    ruido = st.sidebar.slider("Ruído dos rótulos", 0.0, 0.30, 0.05, step=0.01, help="Fração de dados propositalmente invertida para testar resiliência.")
    seed_dados = st.sidebar.number_input("Seed dos dados", value=7, step=1, help="Garante que os dados sintéticos gerados sejam reproduzíveis.")
    features, gabarito = gerar_dados_sinteticos(n_clientes, ruido, int(seed_dados))
    dados_ok = True

st.sidebar.divider()
st.sidebar.subheader("Evolução")

geracoes = st.sidebar.slider("Máx. de gerações", 50, 2000, 500, step=50, help="Teto máximo de iterações do algoritmo se não atingir o fitness alvo.")
fitness_alvo = st.sidebar.slider("Fitness alvo", 0.50, 0.99, 0.85, step=0.01, help="A execução encerra antecipadamente se atingir este valor (FC).")

with st.sidebar.expander("⚙️ Parâmetros Avançados (Motor Genético)"):
    qtd_cromossomos = st.slider("Tamanho da população", 4, 50, 12, step=1, help="Número de indivíduos/soluções candidatas evoluindo simultaneamente.")
    cruzamentos = st.slider("Cruzamentos por geração", 1, 20, 3, step=1, help="Cada cruzamento gera 3 filhos. Aumenta a velocidade de convergência.")
    seed_ga = st.number_input("Seed da evolução", value=42, step=1, help="Semente para as roletas, cortes de crossover e mutações aleatórias.")

st.sidebar.divider()
rodar = st.sidebar.button("▶ Rodar evolução", use_container_width=True, type="primary")


# ----------------------------
# 8) EMPTY STATE OU PRÉVIA (TABELAS RICAS)
# ----------------------------
if not dados_ok:
    st.markdown("""
    <div class='empty-state'>
        <h3>Aguardando Fonte de Dados</h3>
        <p>Utilize a barra lateral para gerar um banco sintético padronizado CredituS<br>ou faça o upload de uma planilha válida para iniciar o motor genético.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='footer'>The Everything Calculator - Fellipe Almässy • </div>", unsafe_allow_html=True)
    st.stop()

ta_total = int(np.sum(gabarito == 1))
ti_total = int(np.sum(gabarito == 0))

st.markdown(
    f"<div class='small-muted'>Banco carregado: <b>{len(gabarito)}</b> clientes • <b>{features.shape[1]}</b> variáveis • <b>{ta_total}</b> (A) • <b>{ti_total}</b> (I).</div>",
    unsafe_allow_html=True,
)

df_preview = pd.DataFrame(features.astype(int), columns=nomes[:features.shape[1]])
df_preview.insert(0, "Classe Real", np.where(gabarito == 1, "A", "I"))

with st.expander("Ver amostra do banco de dados (Rich Table)", expanded=False):
    st.dataframe(
        df_preview.head(15),
        column_config={
            "Classe Real": st.column_config.TextColumn(
                "Classe", help="A = Adimplente, I = Inadimplente", width="small"
            ),
        },
        use_container_width=True, hide_index=True
    )
st.write("")


# ----------------------------
# 9) RODAR ALGORITMO COM st.status
# ----------------------------
if "ga_result" not in st.session_state:
    st.session_state.ga_result = None

if rodar:
    with st.status("Iniciando motor evolutivo...", expanded=True) as status:
        st.write("Configurando o banco de dados e gerando eneadecaquatérnios...")
        t0 = time.time()
        
        st.write("Rodando cruzamentos, mutações e seleção elitista...")
        res = algoritmo_genetico(
            features, gabarito, qtd_cromossomos=qtd_cromossomos, geracoes=geracoes,
            fitness_alvo=fitness_alvo, cruzamentos_por_geracao=cruzamentos, seed=int(seed_ga)
        )
        
        st.write("Calculando matriz de confusão e diagnósticos finais...")
        res["runtime"] = time.time() - t0
        res["diag"] = diagnosticar(res["melhor_cromossomo"], features, gabarito)
        
        status.update(label=f"Evolução concluída em {res['runtime']:.2f}s!", state="complete", expanded=False)
    st.session_state.ga_result = res

res = st.session_state.ga_result
if res is None:
    st.info("Parâmetros configurados. Clique em **Rodar evolução** na barra lateral para iniciar o treino.")
    st.markdown("<div class='footer'>The Everything Calculator - Fellipe Almässy • </div>", unsafe_allow_html=True)
    st.stop()

diag = res["diag"]


# ----------------------------
# 10) MÉTRICAS
# ----------------------------
st.markdown("<div class='function-display'>", unsafe_allow_html=True)
st.latex(rf"FC_{{\text{{melhor}}}} = {res['melhor_fitness']:.4f}")
st.markdown("</div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns([1.2, 1.0, 1.0, 1.1, 1.1])
m1.metric("Melhor Fitness", f"{res['melhor_fitness']:.4f}", "FC = PA · PI")
m2.metric("PA (acerto A)", f"{diag['pa'] * 100:.1f}%", f"{diag['nac']}/{diag['ta']}")
m3.metric("PI (acerto I)", f"{diag['pi'] * 100:.1f}%", f"{diag['nic']}/{diag['ti']}")
m4.metric("Acurácia Global", f"{diag['acuracia'] * 100:.1f}%", f"{diag['nac'] + diag['nic']}/{len(gabarito)}")
m5.metric("Convergiu em", f"{res['geracao_convergencia']} ger.", f"{res['runtime']:.2f}s")
st.write("")


# ----------------------------
# 11) ABAS: MOTOR / DIAGNÓSTICOS
# ----------------------------
tab_motor, tab_diag = st.tabs(["Visão do Motor", "Diagnósticos"])

with tab_motor:
    fig_conv = make_convergence_plot(res["hist_melhor"], res["hist_media"])
    st.plotly_chart(fig_conv, use_container_width=True, config={'displayModeBar': False})

    fig_chromo = make_chromosome_plot(res["melhor_cromossomo"])
    st.plotly_chart(fig_chromo, use_container_width=True, config={'displayModeBar': False})
    st.caption("Barras verdes = peso positivo (Adimplente) | Barras vermelhas = peso negativo (Inadimplente).")

    rotulos_genes = ["b0 (bias)"] + [f"b{a}{n}" for a in range(1, 10) for n in (1, 2)]
    df_chromo = pd.DataFrame({"gene": rotulos_genes[:len(res["melhor_cromossomo"])], "peso": res["melhor_cromossomo"]})
    st.download_button("⬇ Baixar cromossomo treinado (.csv)", data=df_chromo.to_csv(index=False).encode("utf-8"), file_name="cromossomo_treinado.csv", mime="text/csv")

with tab_diag:
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(make_confusion_plot(diag), use_container_width=True, config={'displayModeBar': False})
    with c2: st.plotly_chart(make_roleta_plot(res["fitness_final"]), use_container_width=True, config={'displayModeBar': False})

    st.plotly_chart(make_q_distribution_plot(diag["q"], diag["hipotese"]), use_container_width=True, config={'displayModeBar': False})

    st.markdown("<div class='small-muted'>Classificação individual (Rich Table):</div>", unsafe_allow_html=True)
    df_result = pd.DataFrame({
        "Classe Real": np.where(gabarito == 1, "A", "I"),
        "Q": np.round(diag["q"], 4),
        "Predito": np.where(diag["hipotese"] == 1, "A", "I"),
        "Status": np.where(diag["hipotese"] == gabarito, "✅ Acerto", "❌ Erro"),
    })
    st.dataframe(
        df_result.head(25),
        column_config={
            "Classe Real": st.column_config.TextColumn("Real"),
            "Predito": st.column_config.TextColumn("Predito"),
            "Q": st.column_config.NumberColumn("Valor de Q", format="%.4f"),
            "Status": st.column_config.TextColumn("Status do Algoritmo")
        },
        use_container_width=True, hide_index=True
    )


# ----------------------------
# 12) RODAPÉ
# ----------------------------
st.markdown("<div class='footer'>The Everything Calculator - Fellipe Almässy • </div>", unsafe_allow_html=True)
