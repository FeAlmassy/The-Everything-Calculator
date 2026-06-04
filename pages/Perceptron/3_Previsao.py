"""Página de previsão: aplica um modelo treinado a uma base de clientes novos."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import ga_perceptron as ga


st.set_page_config(page_title="Previsão", page_icon="🔮", layout="wide")
st.title("🔮 Previsão de clientes novos")
st.caption("Aplica um modelo treinado a uma base sem rótulos.")

# ---------------------------------------------------------------------------
# 1. Upload do modelo
# ---------------------------------------------------------------------------
st.header("1. Modelo treinado")

arquivo_modelo = st.file_uploader(
    "Arraste o arquivo .json do modelo",
    type=["json"],
    help="Arquivo gerado na página de treino.",
    key="modelo",
)

if arquivo_modelo is None:
    st.info("⬆️ Faça upload do modelo treinado (.json) para continuar.")
    st.stop()

try:
    modelo = ga.desserializar_modelo(arquivo_modelo.getvalue())
except Exception as e:
    st.error(f"❌ Erro ao ler o modelo: {e}")
    st.stop()

# resumo do modelo
metadados = modelo.get("metadados", {})
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Features esperadas", len(modelo["nomes_features"]))
col_m2.metric(
    "G-mean (treino)",
    f"{metadados.get('fitness_treino', 0):.4f}" if metadados.get("fitness_treino") is not None else "—",
)
ft = metadados.get("fitness_teste")
col_m3.metric("G-mean (teste)", f"{ft:.4f}" if ft is not None else "—")
col_m4.metric("Treinado em", metadados.get("data_treino", "—")[:10] if metadados.get("data_treino") else "—")

with st.expander("👁️ Detalhes do modelo"):
    st.markdown(f"**Features na ordem esperada** ({len(modelo['nomes_features'])}):")
    st.code(", ".join(modelo["nomes_features"]))

    st.markdown("**Metadados do treino:**")
    st.json(metadados)

# ---------------------------------------------------------------------------
# 2. Upload da base de previsão
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Base de clientes novos")

arquivo_base = st.file_uploader(
    "Arraste o arquivo .xlsx no formato do template_prever",
    type=["xlsx"],
    help="A base deve conter exatamente as mesmas features do treino (sem a coluna 'alvo').",
    key="base",
)

if arquivo_base is None:
    st.info("⬆️ Faça upload da base de clientes novos para gerar as previsões.")
    st.stop()

try:
    X_novo, nomes_novo, ids = ga.carregar_base_prever(arquivo_base)
except ValueError as e:
    st.error(f"❌ Erro de validação: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao ler o arquivo: {e}")
    st.stop()

st.success(f"Base carregada: **{len(ids)}** clientes a serem classificados.")

# ---------------------------------------------------------------------------
# 3. Predição
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Resultado")

try:
    predicoes, scores = ga.prever(modelo, X_novo, nomes_novo)
except ValueError as e:
    st.error(f"❌ Incompatibilidade entre o modelo e a base: {e}")
    st.stop()

# resumo geral
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
col_p1.metric("Total previsto", len(predicoes))
col_p2.metric("Classe 0", int((predicoes == 0).sum()), f"{(predicoes == 0).mean():.1%}")
col_p3.metric("Classe 1", int((predicoes == 1).sum()), f"{(predicoes == 1).mean():.1%}")
col_p4.metric("Score q (médio)", f"{scores.mean():+.3f}")

# tabela de resultados — com classificação de confiança
df_resultado = pd.DataFrame({
    "id": ids,
    "predicao": predicoes,
    "score_q": scores,
})


def classificar_confianca(q):
    """Classifica |q| em níveis de confiança baseado em quantis empíricos."""
    abs_q = abs(q)
    if abs_q < 0.5:
        return "limítrofe"
    elif abs_q < 1.5:
        return "moderada"
    else:
        return "alta"


df_resultado["confianca"] = df_resultado["score_q"].apply(classificar_confianca)
df_resultado["score_q"] = df_resultado["score_q"].round(4)

# adiciona as features originais ao lado, para inspeção
df_features = pd.DataFrame(X_novo, columns=modelo["nomes_features"])
df_completo = pd.concat([df_resultado.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

# UI: filtros
st.markdown("##### Filtros")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    filtro_classe = st.multiselect(
        "Filtrar por predição",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: f"Classe {x}",
    )
with col_f2:
    filtro_confianca = st.multiselect(
        "Filtrar por confiança",
        options=["alta", "moderada", "limítrofe"],
        default=["alta", "moderada", "limítrofe"],
    )
with col_f3:
    ordenar_por = st.selectbox(
        "Ordenar por",
        options=["score_q (decrescente)", "score_q (crescente)", "|score_q| (decrescente)", "id"],
    )

df_filtrado = df_completo[
    df_completo["predicao"].isin(filtro_classe) & df_completo["confianca"].isin(filtro_confianca)
].copy()

if ordenar_por == "score_q (decrescente)":
    df_filtrado = df_filtrado.sort_values("score_q", ascending=False)
elif ordenar_por == "score_q (crescente)":
    df_filtrado = df_filtrado.sort_values("score_q", ascending=True)
elif ordenar_por == "|score_q| (decrescente)":
    df_filtrado = df_filtrado.assign(_abs=df_filtrado["score_q"].abs()).sort_values("_abs", ascending=False).drop(columns=["_abs"])
else:
    df_filtrado = df_filtrado.sort_values("id")

# estilização: cor de fundo na coluna de predição e barra no score_q
def estilo_predicao(val):
    if val == 1:
        return "background-color: #d4edda; color: #155724; font-weight: 600;"
    elif val == 0:
        return "background-color: #f8d7da; color: #721c24; font-weight: 600;"
    return ""


def estilo_confianca(val):
    cores = {
        "alta": "background-color: #d1ecf1; color: #0c5460;",
        "moderada": "background-color: #fff3cd; color: #856404;",
        "limítrofe": "background-color: #f5c6cb; color: #721c24;",
    }
    return cores.get(val, "")


styled = (
    df_filtrado.style
    .map(estilo_predicao, subset=["predicao"])
    .map(estilo_confianca, subset=["confianca"])
    .background_gradient(subset=["score_q"], cmap="RdYlGn", vmin=-3, vmax=3)
    .format({"score_q": "{:+.4f}"})
)

st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

st.caption(
    "**Como ler o score q:** valor positivo → classe 1, negativo → classe 0. "
    "|q| < 0.5 = limítrofe (próximo da fronteira de decisão), 0.5 ≤ |q| < 1.5 = moderada, "
    "|q| ≥ 1.5 = alta confiança. Quanto maior |q|, mais distante do hiperplano de separação."
)

# ---------------------------------------------------------------------------
# 4. Download
# ---------------------------------------------------------------------------
st.divider()
st.markdown("### 💾 Exportar previsões")

col_d1, col_d2 = st.columns(2)

with col_d1:
    csv_bytes = df_completo.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Baixar CSV (com features)",
        data=csv_bytes,
        file_name="previsoes.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    # versão enxuta para Excel
    from io import BytesIO
    from openpyxl import Workbook

    buffer = BytesIO()
    df_xlsx = df_completo[["id", "predicao", "score_q", "confianca"]].copy()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_xlsx.to_excel(writer, sheet_name="previsoes", index=False)
    buffer.seek(0)
    st.download_button(
        label="⬇️ Baixar XLSX (resumo)",
        data=buffer.getvalue(),
        file_name="previsoes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
