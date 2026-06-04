import streamlit as st

st.set_page_config(
    page_title="TEC – The Everything Calculator",
    page_icon="🔢",
    layout="wide",
)

pg = st.navigation({
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
    ],
    "Cálculo": [
        st.Page("pages/Calculo/integral_3d.py", title="Integral 3D", icon="📐"),
        st.Page("pages/Calculo/limite.py",      title="Limites",     icon="🎯"),
        st.Page("pages/Calculo/limiteV2.py",    title="Limites V2",  icon="✨"),
        st.Page("pages/Calculo/limiteV3.py",    title="Limites V3",  icon="✨"),
    ],
    "Machine Learning": [
        # Mantendo os arquivos que você já tinha cadastrados nessa categoria
        st.Page("pages/MachineLearning/Perceptron.py",        title="Perceptron – Teoria",   icon="📘"),
        st.Page("pages/MachineLearning/Perceptron_Treino.py",  title="Perceptron – Treino",   icon="🍬"),
        st.Page("pages/MachineLearning/Perceptron_Previsao.py", title="Perceptron – Previsão", icon="🔮"),
    ],
    "Perceptron": [  # <-- Nova categoria criada no menu esquerdo
        st.Page("pages/Perceptron/1_Teoria.py", title="Teoria", icon="📘"),
        st.Page("pages/Perceptron/3_Previsao.py", title="Previsão", icon="🔮"),
    ],
    "AREA DE TESTES": [
        st.Page("pages/Area_Testes/teste.py", title="teste", icon="📊"),
    ],
})

pg.run()
