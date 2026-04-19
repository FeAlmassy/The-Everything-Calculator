# The Everything Calculator (TEC)

> A multi-tool interactive web app for applied mathematics — built with Python and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)]()

---

## About

TEC is a growing collection of interactive mathematical tools, designed to make abstract concepts from Calculus, Linear Algebra, and Probability tangible and visual.

Each tool is a standalone page with interactive inputs and real-time Plotly visualizations — you change the parameters, the math updates instantly.

The project is built as a practical portfolio for applied mathematics and data science, and is actively expanded as new topics are studied.

---

## Current Pages

### Cálculo
| Page | Description |
|------|-------------|
| **Integral 3D** | Visualizes triple integrals over 3D regions with interactive bounds |
| **Limites** | Computes and displays limits of real functions, step by step |
| **Limites V2** | Extended limit solver with one-sided limit support |
| **Limites V3** | Graphical limit explorer with zoom and tolerance controls |

### Álgebra Linear
| Page | Description |
|------|-------------|
| **Vetores** | Vector operations in 2D/3D: addition, dot product, cross product, magnitude |

### Probabilidade & Processos Estocásticos
| Page | Description |
|------|-------------|
| **Cadeias de Markov** | Interactive Markov chain simulator — define transition matrices, visualize stationary distributions and state evolution |

---

## Tech Stack

- **[Streamlit](https://streamlit.io/)** — multi-page app framework
- **[Plotly](https://plotly.com/python/)** — interactive visualizations
- **[SymPy](https://www.sympy.org/)** — symbolic mathematics (limits, integrals)
- **[NumPy](https://numpy.org/)** — numerical computation

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/FeAlmassy/The-Everything-Calculator.git
cd The-Everything-Calculator

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run Home.py
```

The app will open at `http://localhost:8501`.

---

## Project Structure

```
The-Everything-Calculator/
│
├── Home.py                  # Main entry point — navigation config
├── requirements.txt
│
└── pages/
    ├── home.py              # Landing page
    ├── Calculo/
    │   ├── integral_3d.py
    │   ├── limite.py
    │   ├── limiteV2.py
    │   └── limiteV3.py
    └── Algebra_Linear/
        └── Vetores.py
```

---

## Roadmap

- [ ] Statistics module — distributions, hypothesis testing, CLT simulator
- [ ] Numerical methods — Newton-Raphson, bisection, Euler method
- [ ] Linear regression from scratch with visual residual analysis
- [ ] Matrix decompositions (LU, SVD, eigendecomposition)
- [ ] Time series explorer (ARIMA, moving averages)

---

## Author

**Fellipe Almässy** — Computer Engineering and Applied Maths student at Prandiano Mathematics Museum in São Paulo.
Focused on applied mathematics and data science.

[![GitHub](https://img.shields.io/badge/GitHub-FeAlmassy-black?logo=github)](https://github.com/FeAlmassy)
