"""Página de teoria: formulação matemática do perceptron, da G-mean e do AG."""
import streamlit as st


st.set_page_config(page_title="Teoria", page_icon="📚", layout="wide")

st.title("📚 Teoria")
st.caption("Formulação matemática do perceptron, da métrica G-mean e do algoritmo genético.")

# ---------------------------------------------------------------------------
st.header("1. O problema")

st.markdown(
    r"""
Considere uma base de $N$ registros, cada um com $F$ atributos numéricos
(*features*) e um rótulo binário:

$$
\mathcal{D} = \{(\mathbf{x}_n, y_n)\}_{n=1}^{N},
\quad \mathbf{x}_n \in \mathbb{R}^F,
\quad y_n \in \{0, 1\}.
$$

Buscamos uma função $h : \mathbb{R}^F \to \{0, 1\}$ que generalize bem
para registros não observados — o **classificador**.
"""
)

# ---------------------------------------------------------------------------
st.header("2. O perceptron como classificador linear")

st.markdown(
    r"""
O perceptron é o classificador linear mais simples. Ele combina linearmente
as features com pesos $\mathbf{w} \in \mathbb{R}^F$, adiciona um *bias*
$b \in \mathbb{R}$ e aplica uma função degrau:

$$
q(\mathbf{x}) \;=\; \mathbf{w}^\top \mathbf{x} + b,
\qquad
h(\mathbf{x}) \;=\;
\begin{cases}
1, & \text{se } q(\mathbf{x}) \geq 0 \\
0, & \text{caso contrário.}
\end{cases}
$$

A equação $q(\mathbf{x}) = 0$ define um **hiperplano** em $\mathbb{R}^F$ —
a **fronteira de decisão**. Tudo de um lado é classificado como 1, tudo do
outro como 0. O valor $q(\mathbf{x})$ pode ser lido como uma **distância
sinalizada** à fronteira: $|q|$ grande significa alta confiança;
$|q| \approx 0$ significa caso limítrofe.
"""
)

with st.expander("🔍 Por que padronizar as features antes do treino?"):
    st.markdown(
        r"""
Sem padronização, features de escalas muito diferentes (ex.: idade em $[0, 100]$
e renda em $[0, 50000]$) tornam o espaço de pesos extremamente desbalanceado.
O peso que multiplica a renda só consegue ter efeito significativo se for
muito pequeno, enquanto o peso da idade pode ser grande. Como inicializamos
os pesos uniformemente em $[-1, 1]$, o algoritmo desperdiça gerações inteiras
ajustando ordens de grandeza em vez de aprender padrões.

Aplicamos **z-score** em cada feature:

$$
\tilde{x}_j \;=\; \frac{x_j - \mu_j}{\sigma_j},
\quad j = 1, \dots, F,
$$

onde $\mu_j$ e $\sigma_j$ são a média e o desvio-padrão da feature $j$ na
base de treino. Após a transformação, todas as features têm média $0$ e
desvio $1$ — o GA explora o espaço de pesos de forma uniforme e converge
muito mais rápido.

⚠️ **Crítico:** os parâmetros $\mu_j$ e $\sigma_j$ devem ser salvos junto
com o modelo. Na hora de prever clientes novos, aplica-se exatamente a
mesma transformação. Caso contrário, o modelo está vendo dados em uma
escala diferente da que foi treinado.
"""
    )

# ---------------------------------------------------------------------------
st.header("3. A métrica de fitness: G-mean")

st.markdown(
    r"""
Para avaliar quão bom é um conjunto de pesos $(\mathbf{w}, b)$, precisamos
de uma métrica. **Acurácia simples não serve** quando as classes estão
desbalanceadas: em uma base com $95\%$ de adimplentes, classificar todos
como 1 dá $95\%$ de acurácia, sem ter aprendido nada.

A **média geométrica de sensibilidade e especificidade** (G-mean) resolve isso:

$$
\text{TPR} = \frac{\text{acertos na classe 1}}{\text{total da classe 1}},
\qquad
\text{TNR} = \frac{\text{acertos na classe 0}}{\text{total da classe 0}},
$$

$$
\boxed{\;\;
\text{G-mean} \;=\; \sqrt{\,\text{TPR} \cdot \text{TNR}\,}
\;\;}
$$

A propriedade central da G-mean é que **se uma das taxas cai a zero, a
métrica inteira cai a zero**. Não há como compensar péssimo desempenho
em uma classe acertando muito a outra — o produto sob a raiz penaliza
o desequilíbrio. Por isso ela é o padrão em problemas desbalanceados
(risco de crédito, detecção de fraude, diagnóstico médico).
"""
)

with st.expander("🔍 Relação com a métrica original do projeto"):
    st.markdown(
        r"""
O código original calculava $\text{TPR} \cdot \text{TNR}$ (sem a raiz).
A ordenação dos cromossomos é idêntica — o cromossomo de maior
$\text{TPR} \cdot \text{TNR}$ também tem maior $\sqrt{\text{TPR} \cdot \text{TNR}}$ —
mas a versão com raiz é a **definição canônica** na literatura.

A vantagem da raiz é que os valores são diretamente interpretáveis na
mesma escala que TPR e TNR. Uma G-mean de $0{,}9$ significa, no mínimo,
$0{,}9$ tanto em TPR quanto em TNR (limite atingido quando ambas são iguais).
"""
    )

# ---------------------------------------------------------------------------
st.header("4. O algoritmo genético")

st.markdown(
    r"""
A G-mean **não é diferenciável** com relação aos pesos (a função degrau
introduz uma descontinuidade), então o gradiente descendente clássico
não se aplica diretamente. O **algoritmo genético** contorna isso: ele
otimiza usando apenas avaliações da função, sem precisar de derivadas.

### Cromossomo

Cada candidato à solução é um **cromossomo** — um vetor de $F + 1$ números
reais que codifica os pesos e o bias:

$$
\mathbf{c} \;=\; [\,b, \; w_1, \; w_2, \; \dots, \; w_F\,] \in \mathbb{R}^{F+1}.
$$

A **população** é uma matriz $P \in \mathbb{R}^{C \times (F+1)}$ com $C$
cromossomos por linha. Inicializa-se cada gene uniformemente em $[-1, 1]$.

### Ciclo evolutivo

A cada geração:

1. **Avaliação** — calcular o fitness (G-mean) de cada cromossomo.
2. **Seleção** — escolher dois pais com probabilidade proporcional ao fitness (roleta).
3. **Cruzamento** — combinar os pais em pontos de corte para gerar filhos.
4. **Mutação** — perturbar genes dos filhos com pequena probabilidade.
5. **Substituição** — os $k$ piores cromossomos são substituídos pelos $k$ filhos.

### Seleção por roleta

Cada cromossomo $c_i$ tem probabilidade de ser escolhido igual a

$$
p_i \;=\; \frac{f_i}{\sum_{j=1}^{C} f_j},
$$

onde $f_i$ é seu fitness. Pais melhores têm mais chance de se reproduzir,
mas pais ruins não estão excluídos — preserva-se diversidade genética.
Implementação: ordenar acumulado e fazer busca binária com um número aleatório
em $[0, 1]$. Garante-se que pai $\neq$ mãe (seleção sem reposição).

### Cruzamento de um ponto

Sorteia-se um ponto de corte $k \in \{1, \dots, F\}$. Os filhos herdam um
prefixo de um pai e um sufixo do outro:

$$
\mathbf{f}_a = [\,b_p, w_{p,1}, \dots, w_{p,k-1}, \; w_{m,k}, \dots, w_{m,F}\,],
$$

$$
\mathbf{f}_b = [\,b_m, w_{m,1}, \dots, w_{m,k-1}, \; w_{p,k}, \dots, w_{p,F}\,].
$$

### Mutação probabilística

Cada gene de cada filho tem probabilidade $\rho$ (a **taxa de mutação**)
de ser reamostrado uniformemente em $[-1, 1]$, independentemente dos outros.
Esta é a formulação canônica — a quantidade esperada de genes mutados por
filho é $\rho \cdot (F + 1)$.

A mutação cumpre o papel de **exploração**: previne o algoritmo de ficar
preso em ótimos locais ao injetar variação aleatória. Taxa típica: $\rho \in [0{,}01,\, 0{,}10]$.

### Elitismo implícito

Como a cada geração substituímos os **$k$ piores** cromossomos pelos
filhos, os melhores nunca são descartados. O fitness do melhor cromossomo
é uma sequência monótona não-decrescente ao longo das gerações.

### Critérios de parada

O treino encerra quando o **primeiro** de três critérios é atingido:

1. **Fitness alvo** — o melhor cromossomo atingiu o nível desejado de qualidade.
2. **Máximo de gerações** — orçamento computacional esgotado.
3. **Estagnação** — o melhor fitness não melhora há $\pi$ gerações (a *paciência*).
   Indica convergência (atingiu um ótimo, possivelmente local) e evita
   queimar tempo de máquina sem ganho.
"""
)

# ---------------------------------------------------------------------------
st.header("5. Generalização: treino × teste")

st.markdown(
    r"""
Avaliar o fitness em $100\%$ da base **não diz nada** sobre quão bem o modelo
classifica registros novos — pode estar simplesmente **decorando** a base de
treino. Para medir generalização, dividimos a base aleatoriamente em duas
partes disjuntas:

- **Treino** ($\sim 70$–$80\%$): o GA otimiza fitness aqui.
- **Teste** ($\sim 20$–$30\%$): o modelo final é avaliado aqui, **sem ter
  visto esses dados em nenhum momento**.

O *split* é **estratificado**: a proporção de classes 0 e 1 é preservada nos
dois conjuntos. Isso evita que uma base muito desbalanceada gere um conjunto
de teste com uma única classe.

### Como interpretar a diferença

- **Treino ≈ teste** ✓ — o modelo generaliza bem.
- **Treino ≫ teste** ✗ — *overfitting*: decorou a base de treino e perdeu
  poder de generalização. Soluções: simplificar (menos features), regularizar,
  conseguir mais dados.
- **Treino baixo** ✗ — *underfitting*: o modelo é simples demais para o problema.
  Soluções: features mais informativas, modelo mais complexo (perceptron linear
  pode não dar conta de relações não-lineares).
"""
)

# ---------------------------------------------------------------------------
st.header("6. Sobre a base de dados")

st.markdown(
    r"""
O **contrato de dados** deste projeto impõe três regras:

1. Uma coluna chamada `id` — identificador único, descartada no treino.
2. Uma coluna chamada `alvo` — variável binária (0/1) a ser prevista.
3. Todas as demais colunas são features numéricas.

Essa restrição existe porque tentar adivinhar qual coluna é o alvo, qual é
o id, e quais são as features é uma fonte infinita de bugs. Ao impor a
estrutura via template, podemos validar de forma estrita e dar mensagens
de erro precisas.

### Variáveis categóricas

O perceptron só aceita features **numéricas**. Variáveis categóricas devem
ser convertidas antes do upload, usando *one-hot encoding*:

| cor (categórica) | cor_vermelho | cor_azul | cor_verde |
|------------------|--------------|----------|-----------|
| vermelho         | 1            | 0        | 0         |
| azul             | 0            | 1        | 0         |
| verde            | 0            | 0        | 1         |

Cada categoria vira uma feature binária independente. Para evitar
**multicolinearidade**, costuma-se omitir uma categoria de referência
(no exemplo, manter só duas das três colunas).
"""
)

# ---------------------------------------------------------------------------
st.header("7. Limitações")

st.markdown(
    r"""
O perceptron é um classificador **linear**. Ele só consegue separar as
classes se existir um hiperplano que as divida razoavelmente bem (ou seja,
se elas forem **linearmente separáveis**, ou aproximadamente).

Para problemas com fronteiras de decisão não-lineares — como o clássico
XOR — o perceptron não converge para uma solução boa, **independentemente
do otimizador**. Isso não é uma falha do algoritmo genético; é uma
limitação da própria classe de modelos. Para esses casos, são necessários:

- **Engenharia de features**: criar interações ($x_1 \cdot x_2$, $x_1^2$, etc.)
  manualmente, dando ao perceptron um espaço onde a separação seja linear.
- **Redes neurais multicamadas**: empilhar perceptrons com não-linearidades
  intermediárias. Cada camada adiciona poder de representação.
- **Métodos de kernel** (SVM): operar implicitamente em um espaço de alta
  dimensão sem materializá-lo.

Este projeto serve como **fundação conceitual**. Os mesmos blocos (split,
padronização, otimização baseada em métrica, persistência de modelo)
escalam para modelos mais sofisticados.
"""
)

st.divider()
st.caption(
    "Referências canônicas: Goldberg, *Genetic Algorithms in Search, Optimization "
    "and Machine Learning*, 1989. Bishop, *Pattern Recognition and Machine Learning*, "
    "2006. Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2009."
)
