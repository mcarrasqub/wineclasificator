import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

# Configuración de la página
st.set_page_config(page_title="Wine Classifier", layout="wide")
st.title("🍷 Wine Classification Explorer")

# 1. Carga de datos
@st.cache_data
def get_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = get_data()

# Barra lateral para parámetros
st.sidebar.header("Configuración del Modelo")
tipo_modelo = st.sidebar.selectbox("Selecciona técnica:", ["Árbol Único", "Bagging", "Boosting (AdaBoost)"])
max_depth = st.sidebar.slider("Profundidad del Árbol (max_depth):", 1, 10, 3)

# 2. Definición del Modelo Base
base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

if tipo_modelo == "Árbol Único":
    model = base_tree
elif tipo_modelo == "Bagging":
    n_estimators = st.sidebar.number_input("Número de árboles (Estimadores):", 5, 100, 10)
    model = BaggingClassifier(estimator=base_tree, n_estimators=n_estimators, random_state=42)
else: # Boosting
    n_estimators = st.sidebar.number_input("Número de iteraciones:", 5, 100, 50)
    model = AdaBoostClassifier(estimator=base_tree, n_estimators=n_estimators, random_state=42)

# 3. Validación Cruzada
st.subheader(f"Evaluación del Modelo: {tipo_modelo}")

if st.button("Entrenar y Validar"):
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro')
    }
    
    cv_results = cross_validate(model, df.drop('target', axis=1), df['target'], cv=5, scoring=scoring)
    
    # Mostrar Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy Media", f"{cv_results['test_accuracy'].mean():.2%}")
    col2.metric("Precisión Media", f"{cv_results['test_precision'].mean():.2%}")
    col3.metric("Recall Medio", f"{cv_results['test_recall'].mean():.2%}")

    # Mostrar dataframe de los datos
    st.write("### Datos de entrenamiento (Muestra)")
    st.dataframe(df.head())
else:
    st.info("Ajusta los parámetros en la barra lateral y presiona 'Entrenar'.")