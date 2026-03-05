import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score

# 1. Configuración de la página (Limpia para evitar errores de renderizado)
st.set_page_config(page_title="Wine Classifier ML", layout="wide")

# Título simple sin caracteres especiales al inicio para evitar errores de slugify
st.title("Wine Classification Machine Learning Project")

# 2. Carga de datos con Cache para rendimiento
@st.cache_data
def get_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = get_data()

# 3. Interfaz de Usuario (Sidebar)
st.sidebar.header("Parametros del Modelo")
metodo = st.sidebar.selectbox(
    "Seleccione la tecnica:", 
    ["Arbol de Decision Unico", "Bagging", "Boosting (AdaBoost)"]
)

max_depth = st.sidebar.slider("Profundidad del Arbol (max_depth):", 1, 20, 3)

# 4. Construcción del Modelo según selección
base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

if metodo == "Arbol de Decision Unico":
    model = base_tree
elif metodo == "Bagging":
    n_estimators = st.sidebar.number_input("Numero de estimadores (Arboles):", 5, 100, 10)
    model = BaggingClassifier(estimator=base_tree, n_estimators=n_estimators, random_state=42)
else: # Boosting
    n_estimators = st.sidebar.number_input("Numero de iteraciones:", 5, 100, 50)
    model = AdaBoostClassifier(estimator=base_tree, n_estimators=n_estimators, random_state=42)

# 5. Ejecución y Métricas
st.write(f"### Evaluacion mediante Validacion Cruzada ({metodo})")

if st.button("Entrenar y Validar Modelo"):
    # Definimos métricas para multiclase (macro para balancear las clases)
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0)
    }
    
    # Validación Cruzada (K-Fold CV = 5)
    cv_results = cross_validate(
        model, 
        df.drop('target', axis=1), 
        df['target'], 
        cv=5, 
        scoring=scoring
    )
    
    # Mostrar resultados en columnas
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy (Promedio)", f"{cv_results['test_accuracy'].mean():.2%}")
    col2.metric("Precision (Promedio)", f"{cv_results['test_precision'].mean():.2%}")
    col3.metric("Recall (Promedio)", f"{cv_results['test_recall'].mean():.2%}")
    
    st.success("Analisis completado exitosamente.")
    
    # Mostrar tabla de datos para referencia
    st.write("#### Vista previa de los datos")
    st.dataframe(df.head(10))
else:
    st.info("Haga clic en el boton para iniciar el entrenamiento con los parametros seleccionados.")

# Pie de página técnico
st.markdown("---")
st.caption("Proyecto desarrollado con Scikit-Learn y Streamlit.")