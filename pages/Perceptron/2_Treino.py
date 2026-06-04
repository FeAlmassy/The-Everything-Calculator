"""Página de treino: upload da base, parâmetros do GA, treino, download do modelo."""
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

# garante import do módulo na raiz do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))
import ga_perceptron as ga


st.set_page_config(page_title="Treino", page_icon="🧬", layout="wide")
st.title("🧬 Treino do modelo")
st.caption("Otimização dos pesos do perceptron via algoritmo genético.")

# ---------------------------------------------------------------------------
# 1. Upload da base
# ---------------------------------------------------------------------------
st.header("1. Base de treino")

arquivo = st.file_uploader(
    "Arraste o arquivo .xlsx no formato do template",
    type=["xlsx"],
    help="O arquivo deve ter uma aba 'dados' com colunas 'id', 'alvo' e features numéricas.",
)

if arquivo is None:
    st.info("⬆️ Faça upload da base para continuar. Baixe o template na página inicial se precisar.")
    st.stop()

try:
    X, y, nomes_features = ga.carregar_base_treino(arquivo)
except ValueError as e:
    st.error(f"❌ Erro de validação: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao ler o arquivo: {e}")
    st.stop()

# resumo da base
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Registros", X.shape[0])
col_b.metric("Features", X.shape[1])
col_c.metric("Genes (bias + pesos)", X.shape[1] + 1)
col_d.metric("Classe 1 / Total", f"{(y == 1).sum()} / {len(y)} ({(y == 1).mean():.1%})")

with st.expander("👀 Ver features detectadas"):
    df_features = pd.DataFrame({
        "feature": nomes_features,
        "média": X.mean(axis=0).round(4),
        "desvio": X.std(axis=0).round(4),
        "mínimo": X.min(axis=0).round(4),
        "máximo": X.max(axis=0).round(4),
    })
    st.dataframe(df_features, use_container_width=True, hide_index=True)

if X.shape[0] < 50:
    st.warning(f"⚠️ A base tem só {X.shape[0]} registros. Recomenda-se pelo menos 50.")

# ---------------------------------------------------------------------------
# 2. Parâmetros do GA
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Parâmetros do algoritmo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### População e reprodução")
    qtd_cromossomos = st.slider(
        "Quantidade de cromossomos (tamanho da população)",
        min_value=20, max_value=500, value=200, step=10,
        help="Mais cromossomos → mais diversidade, mais lento por geração.",
    )
    n_filhos_por_geracao = st.slider(
        "Filhos gerados por geração",
        min_value=2, max_value=min(100, qtd_cromossomos - 2), value=min(20, qtd_cromossomos - 2), step=2,
        help="Os N piores cromossomos são substituídos pelos N filhos a cada geração.",
    )
    taxa_mutacao = st.slider(
        "Taxa de mutação ρ (probabilidade por gene)",
        min_value=0.0, max_value=0.30, value=0.05, step=0.01,
        format="%.2f",
        help=(
            f"Cada gene de cada filho muta independentemente com probabilidade ρ. "
            f"Como o cromossomo tem {X.shape[1] + 1} genes, o número esperado de "
            f"mutações por filho é ρ × {X.shape[1] + 1}. Típico: 0.01 a 0.10."
        ),
    )

with col2:
    st.markdown("##### Treino e parada")
    n_geracoes_max = st.number_input(
        "Máximo de gerações", min_value=10, max_value=20000, value=500, step=50,
        help="Limite superior. O treino pode parar antes por atingir o alvo ou por estagnação.",
    )
    fitness_alvo = st.slider(
        "Fitness alvo (G-mean)", min_value=0.50, max_value=1.00, value=0.95, step=0.01,
        format="%.2f",
        help="Se o melhor cromossomo atingir esse fitness, o treino para imediatamente.",
    )
    paciencia = st.number_input(
        "Paciência (gerações sem melhora até parar)",
        min_value=5, max_value=1000, value=50, step=5,
        help="Se o melhor fitness não melhorar por N gerações, o treino para (estagnação).",
    )

st.markdown("##### Divisão treino/teste e reprodutibilidade")
col_s1, col_s2 = st.columns(2)
with col_s1:
    frac_treino = st.slider(
        "Fração para treino", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        format="%.2f",
        help="O restante vira conjunto de teste, usado só para avaliar generalização. "
             "Use 1.00 para treinar com a base inteira (sem conjunto de teste).",
    )
with col_s2:
    seed = st.number_input(
        "Seed (semente aleatória)", min_value=0, max_value=2**31 - 1, value=42, step=1,
        help="Fixa toda a aleatoriedade: split, inicialização, seleção, cruzamento, mutação. "
             "Mesma seed + mesmos parâmetros = resultado idêntico.",
    )

with st.expander("ℹ️ Resumo dos parâmetros"):
    st.markdown(
        f"""
- População de **{qtd_cromossomos}** cromossomos com **{X.shape[1] + 1}** genes cada.
- A cada geração, **{n_filhos_por_geracao}** filhos são gerados e substituem os piores.
- Cada gene de cada filho tem **{taxa_mutacao:.0%}** de chance de mutar (esperado:
  ~{taxa_mutacao * (X.shape[1] + 1):.1f} genes por filho).
- O treino para no primeiro dos critérios: G-mean ≥ **{fitness_alvo:.2f}**,
  **{n_geracoes_max}** gerações executadas, ou **{paciencia}** gerações consecutivas sem melhora.
- Divisão: **{frac_treino:.0%}** treino / **{1 - frac_treino:.0%}** teste, estratificada.
- Seed = **{seed}** (reprodutibilidade total).
"""
    )

# ---------------------------------------------------------------------------
# 3. Treino
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Treinar")

if st.button("🚀 Iniciar treino", type="primary", use_container_width=True):

    # padronização (calcula scaler na base TODA, depois aplica)
    X_pad, scaler_media, scaler_desvio = ga.padronizar_treino(X)

    # split estratificado
    rng_split = np.random.default_rng(seed)
    if frac_treino < 1.0:
        X_tr, X_te, y_tr, y_te = ga.dividir_treino_teste(X_pad, y, frac_treino, rng_split)
    else:
        X_tr, y_tr = X_pad, y
        X_te, y_te = None, None

    st.info(
        f"Treinando com {len(y_tr)} registros · Teste com {len(y_te) if y_te is not None else 0} registros · "
        f"Seed = {seed}"
    )

    # UI de progresso
    barra = st.progress(0.0, text="Geração 0")
    placeholder_grafico = st.empty()
    placeholder_metricas = st.empty()

    # buffer para o gráfico em tempo real
    historico_gen = []
    historico_melhor = []
    historico_medio = []

    # callback chamado a cada geração pelo módulo do GA
    def on_geracao(h: ga.HistoricoGeracao):
        historico_gen.append(h.geracao)
        historico_melhor.append(h.melhor_fitness)
        historico_medio.append(h.fitness_medio)
        barra.progress(
            min(h.geracao / n_geracoes_max, 1.0),
            text=f"Geração {h.geracao}/{n_geracoes_max} · melhor fitness = {h.melhor_fitness:.4f}",
        )
        # atualiza o gráfico a cada 5 gerações pra não ficar pesado
        if h.geracao % 5 == 0 or h.geracao == 1:
            df_h = pd.DataFrame({
                "geração": historico_gen,
                "melhor": historico_melhor,
                "médio": historico_medio,
            }).set_index("geração")
            placeholder_grafico.line_chart(df_h, height=300)

    # roda o GA
    resultado = ga.treinar(
        X_tr, y_tr,
        qtd_cromossomos=qtd_cromossomos,
        n_filhos_por_geracao=n_filhos_por_geracao,
        taxa_mutacao=taxa_mutacao,
        n_geracoes_max=int(n_geracoes_max),
        fitness_alvo=fitness_alvo,
        paciencia=int(paciencia),
        seed=int(seed),
        X_teste=X_te, y_teste=y_te,
        callback=on_geracao,
    )

    # gráfico final completo
    df_h = pd.DataFrame({
        "geração": historico_gen,
        "melhor": historico_melhor,
        "médio": historico_medio,
    }).set_index("geração")
    placeholder_grafico.line_chart(df_h, height=300)
    barra.progress(1.0, text=f"✅ Concluído · {resultado.geracoes_executadas} gerações")

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    st.success(f"Treino concluído! Motivo da parada: **{resultado.motivo_parada}**")

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("G-mean (treino)", f"{resultado.fitness_treino:.4f}")
    if resultado.fitness_teste is not None:
        gap = resultado.fitness_treino - resultado.fitness_teste
        col_r2.metric(
            "G-mean (teste)", f"{resultado.fitness_teste:.4f}",
            delta=f"{-gap:.4f} vs treino",
            delta_color="inverse",
        )
    else:
        col_r2.metric("G-mean (teste)", "—", help="Treinou com 100% da base (sem teste).")
    col_r3.metric("Gerações executadas", resultado.geracoes_executadas)

    # matriz de confusão no teste (se houver)
    if X_te is not None and y_te is not None and len(X_te) > 0:
        with st.expander("📊 Matriz de confusão (conjunto de teste)"):
            bias = resultado.cromossomo[0]
            pesos = resultado.cromossomo[1:]
            q_te = X_te @ pesos + bias
            pred_te = (q_te >= 0).astype(int)

            vp = int(((pred_te == 1) & (y_te == 1)).sum())
            vn = int(((pred_te == 0) & (y_te == 0)).sum())
            fp = int(((pred_te == 1) & (y_te == 0)).sum())
            fn = int(((pred_te == 0) & (y_te == 1)).sum())

            cm = pd.DataFrame(
                [[vn, fp], [fn, vp]],
                index=["real 0", "real 1"],
                columns=["pred 0", "pred 1"],
            )
            st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)

            tpr = vp / max(vp + fn, 1)
            tnr = vn / max(vn + fp, 1)
            acuracia = (vp + vn) / len(y_te)
            st.markdown(
                f"""
                - **Sensibilidade (TPR)** = {tpr:.4f} — acertos na classe 1.
                - **Especificidade (TNR)** = {tnr:.4f} — acertos na classe 0.
                - **G-mean** = √(TPR · TNR) = {(tpr * tnr) ** 0.5:.4f}.
                - Acurácia simples = {acuracia:.4f} (referência apenas; não usada como fitness).
                """
            )

    # ------------------------------------------------------------------
    # Pesos aprendidos
    # ------------------------------------------------------------------
    with st.expander("⚖️ Pesos aprendidos pelo modelo"):
        df_pesos = pd.DataFrame({
            "feature": ["(bias)"] + nomes_features,
            "peso": resultado.cromossomo.round(4),
        })
        df_pesos["|peso|"] = df_pesos["peso"].abs()
        df_pesos = df_pesos.sort_values("|peso|", ascending=False)
        st.dataframe(df_pesos[["feature", "peso"]], use_container_width=True, hide_index=True)
        st.caption(
            "Como as features foram padronizadas (z-score), o |peso| é uma medida razoável de "
            "importância relativa. Peso positivo empurra a classificação para 1, negativo para 0."
        )

    # ------------------------------------------------------------------
    # Download do modelo
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### 💾 Salvar modelo")

    metadados = {
        "fitness_treino": resultado.fitness_treino,
        "fitness_teste": resultado.fitness_teste,
        "geracoes_executadas": resultado.geracoes_executadas,
        "motivo_parada": resultado.motivo_parada,
        "qtd_cromossomos": qtd_cromossomos,
        "n_filhos_por_geracao": n_filhos_por_geracao,
        "taxa_mutacao": taxa_mutacao,
        "n_geracoes_max": int(n_geracoes_max),
        "fitness_alvo": fitness_alvo,
        "paciencia": int(paciencia),
        "frac_treino": frac_treino,
        "seed": int(seed),
        "n_registros_treino": len(y_tr),
        "n_registros_teste": int(len(y_te)) if y_te is not None else 0,
        "data_treino": datetime.now().isoformat(timespec="seconds"),
    }

    modelo_json = ga.serializar_modelo(
        resultado.cromossomo,
        nomes_features,
        scaler_media,
        scaler_desvio,
        metadados,
    )

    st.download_button(
        label="⬇️ Baixar modelo (.json)",
        data=modelo_json.encode("utf-8"),
        file_name=f"modelo_ga_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
        use_container_width=True,
        help="O arquivo contém os pesos, os parâmetros do scaler e os metadados do treino.",
    )

    with st.expander("👁️ Pré-visualizar conteúdo do modelo"):
        st.code(modelo_json[:2000] + ("\n... (truncado)" if len(modelo_json) > 2000 else ""), language="json")
