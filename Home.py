import streamlit as st

pg = st.navigation({
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
    ],
    "Cálculo": [
        st.Page("pages/Calculo/integral_3d.py", title="Integral 3D", icon="📐"),
        st.Page("pages/Calculo/limite.py",       title="Limites",     icon="🎯"),
        st.Page("pages/Calculo/limiteV2.py",    title="Limites V2",  icon="✨"),
        st.Page("pages/Calculo/limiteV3.py",    title="Limites V3",  icon="✨"),
    ],
    "Machine Learning": [
        st.Page("pages/MachineLearning/Perceptron.py", title="Perceptron", icon="📊"),

    ],
    "AREA DE TESTES": [
        st.Page("pages/Area_Testes/teste.py", title="teste", icon="📊"),
    ],
})

pg.run()



