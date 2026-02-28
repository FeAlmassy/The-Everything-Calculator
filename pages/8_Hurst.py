import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

st.set_page_config(page_title="TEC • Smart Parser", layout="wide")

st.title("🧠 Tradutor Matemático Inteligente")
st.caption("Digite como você fala. O motor interpreta multiplicação implícita e termos usuais.")

# Configuração do Parser para aceitar '2x' como '2*x'
transformations = (standard_transformations + (implicit_multiplication_application,))

user_input = st.text_input("Insira sua função (ex: 3x^2 + sen(2x) + e^x):", value="2x sen(x)")

def smart_interpreter(text):
    # Tradução de termos comuns antes de enviar ao parser
    prepared = text.replace("^", "**").replace("sen", "sin").replace("tg", "tan")
    try:
        return parse_expr(prepared, transformations=transformations)
    except:
        return None

parsed_expr = smart_interpreter(user_input)

if parsed_expr:
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### O que você digitou:")
        st.code(user_input)
    
    with c2:
        st.markdown("### Como o TEC interpretou:")
        st.latex(sp.latex(parsed_expr))
        
    st.success("Motor pronto para integração!")
    
    # Exemplo de derivada automática só para mostrar poder de showroom
    st.markdown("---")
    st.markdown("### Showroom de Cálculo Simbólico (Extra):")
    derivada = sp.diff(parsed_expr, sp.Symbol('x'))
    st.write("Derivada da sua função:")
    st.latex(rf"\frac{{df}}{{dx}} = {sp.latex(derivada)}")

else:
    st.error("Erro de Parsing. Verifique se esqueceu algum parêntese ou caractere especial.")

st.sidebar.info("Este módulo demonstra capacidade de processamento de linguagem formal, essencial para o ITA.")
