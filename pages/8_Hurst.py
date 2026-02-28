import streamlit as st
import sympy as sp

st.set_page_config(page_title="TEC • Puzzle Integrator", layout="centered")

st.title("⌨️ Puzzle Function Builder")
st.caption("Monte sua função clicando nos blocos de construção.")

if 'buffer' not in st.session_state:
    st.session_state.buffer = ""

# Layout do Teclado
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    if c1.button(" x "): st.session_state.buffer += "x"
    if c2.button(" + "): st.session_state.buffer += "+"
    if c3.button(" - "): st.session_state.buffer += "-"
    if c4.button(" * "): st.session_state.buffer += "*"
    
    c1, c2, c3, c4 = st.columns(4)
    if c1.button(" sin( "): st.session_state.buffer += "sin("
    if c2.button(" cos( "): st.session_state.buffer += "cos("
    if c3.button(" exp( "): st.session_state.buffer += "exp("
    if c4.button(" log( "): st.session_state.buffer += "log("

    c1, c2, c3, c4 = st.columns(4)
    if c1.button(" ^2 "): st.session_state.buffer += "**2"
    if c2.button(" ^3 "): st.session_state.buffer += "**3"
    if c3.button(" ( "): st.session_state.buffer += "("
    if c4.button(" ) "): st.session_state.buffer += ")"

# Input manual para ajustes finos
st.session_state.buffer = st.text_input("Expressão Atual:", value=st.session_state.buffer)

col_ctrl1, col_ctrl2 = st.columns(2)
if col_ctrl1.button("Apagar Tudo", use_container_width=True):
    st.session_state.buffer = ""
    st.rerun()

# Preview Matemático
try:
    if st.session_state.buffer:
        expr = sp.sympify(st.session_state.buffer)
        st.markdown("### Visualização Simbólica:")
        st.latex(rf"\int f(x) \, dx \implies f(x) = {sp.latex(expr)}")
        st.success("Sintaxe Válida!")
except Exception:
    st.warning("Sintaxe incompleta ou inválida. Continue montando...")

st.divider()
st.info("Ideal para uso em tablets ou smartphones onde digitar '**' é incômodo.")
