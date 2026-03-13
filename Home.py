import streamlit as st

pg = st.navigation({
    "": [
        st.Page("pages/home.py", title="Home", icon="🏠"),
    ],
    "Cálculo": [
        st.Page("pages/Calculo/integral_3d.py", title="Integral 3D", icon="📐"),
        st.Page("pages/Calculo/limite.py",       title="Limites",     icon="🎯"),
    ],
    "Álgebra Linear": [
        st.Page("pages/Algebra_Linear/Vetores.py", title="Vetores", icon="📊"),
    ],
})

pg.run()
