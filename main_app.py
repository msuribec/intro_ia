import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, 
                             RocCurveDisplay, ConfusionMatrixDisplay)

# Modelos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="ML Prototyper V2", layout="wide")

st.sidebar.title("⚙️ Panel de Control")

# 1. Datos
dataset_name = st.sidebar.selectbox("Dataset", ["Breast Cancer", "Iris", "Wine"])

@st.cache_data
def load_and_prep(name):
    if name == "Breast Cancer": data = datasets.load_breast_cancer()
    elif name == "Iris": data = datasets.load_iris()
    else: data = datasets.load_wine()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names

X_raw, y, target_names = load_and_prep(dataset_name)

# 2. Preproceso y PCA por Varianza
apply_scaling = st.sidebar.toggle("Escalar Datos", value=True)
apply_pca = st.sidebar.toggle("Aplicar PCA", value=False)

X = X_raw.copy()
if apply_scaling:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

if apply_pca:
    var_target = st.sidebar.slider("Varianza Explicada", 0.50, 0.99, 0.95)
    pca_temp = PCA().fit(X)
    n_components = np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= var_target) + 1
    X = PCA(n_components=n_components).fit_transform(X)
    st.sidebar.info(f"Componentes usados: {n_components}")

# 3. Validación y Métricas
cv_type = st.sidebar.selectbox("Estrategia", ["Hold-out", "K-Fold"])
main_metric = st.sidebar.selectbox("Métrica Objetivo", ["Accuracy", "F1-Score"])

# 4. Modelo
clf_key = st.sidebar.selectbox("Algoritmo", ["KNN", "Decision Tree", "Naive Bayes", "LDA"])
model = {"KNN": KNeighborsClassifier(), "Decision Tree": DecisionTreeClassifier(), 
         "Naive Bayes": GaussianNB(), "LDA": LDA()}[clf_key]

# --- EJECUCIÓN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Reporte Superior
acc = accuracy_score(y_test, y_pred)
st.title(f"🚀 Resultado: {clf_key}")
col_m1, col_m2 = st.columns(2)
col_m1.metric("Exactitud (Test)", f"{acc:.2%}")

if acc > 0.85:
    st.success("✅ ¡Desempeño Alto! Modelo listo para despliegue.")
else:
    st.warning("⚠️ Desempeño medio. Intenta ajustar el preproceso.")

# --- VISUALIZACIONES (SIN SCIKIT-PLOT) ---
tab1, tab2 = st.tabs(["📊 Matriz y Reporte", "📈 Curvas y Estabilidad"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Matriz de Confusión")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=target_names, cmap='Blues', ax=ax)
        st.pyplot(fig)
    with c2:
        st.write("### Reporte Detallado")
        st.code(classification_report(y_test, y_pred, target_names=target_names))

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        st.write("### Curva ROC (One-vs-Rest)")
        fig_roc, ax_roc = plt.subplots()
        # Para multiclase o binario usando el método nativo de sklearn
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
        st.pyplot(fig_roc)
    with c4:
        if cv_type == "K-Fold":
            scores = cross_val_score(model, X, y, cv=5)
            st.write("### Estabilidad (K-Fold)")
            fig_cv, ax_cv = plt.subplots()
            sns.boxplot(x=scores, ax=ax_cv, color='lightgreen')
            st.pyplot(fig_cv)
            st.write(f"Promedio CV: {scores.mean():.2%}")
