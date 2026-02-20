import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="AutoML Classifier Prototype", layout="wide")

st.title("🤖 Prototipo de Clasificación Automatizada")
st.markdown("""
Esta aplicación permite cargar un dataset de Scikit-Learn, procesar los datos, 
extraer características y evaluar modelos de Machine Learning dinámicamente.
""")

# --- 1. Carga de Datos ---
st.sidebar.header("Configuración")
data_choice = st.sidebar.selectbox("Selecciona el Dataset", ["Breast Cancer", "Iris", "Wine"])

def load_data(choice):
    if choice == "Breast Cancer":
        data = datasets.load_breast_cancer()
    elif choice == "Iris":
        data = datasets.load_iris()
    else:
        data = datasets.load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

df, target_names = load_data(data_choice)
st.write(f"### Dataset: {data_choice}")
st.dataframe(df.head(), use_container_width=True)

# --- 2. Preproceso y 3. Feature Extraction ---
st.sidebar.subheader("Preprocesamiento")
do_scaling = st.sidebar.checkbox("Escalar Datos (StandardScaler)", value=True)
use_pca = st.sidebar.checkbox("Extraer características (PCA)", value=False)
n_components = st.sidebar.slider("Componentes PCA", 1, min(10, df.shape[1]-1), 2) if use_pca else None

# --- 4. Selección de Modelos ---
st.sidebar.subheader("Modelado")
classifier_name = st.sidebar.selectbox(
    "Selecciona el Algoritmo", 
    ("KNN", "Decision Tree", "Naive Bayes", "LDA")
)

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K (Vecinos)", 1, 15, 3)
        params["K"] = K
    elif clf_name == "Decision Tree":
        max_depth = st.sidebar.slider("Profundidad Máxima", 2, 15, 5)
        params["max_depth"] = max_depth
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        return KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=params["max_depth"])
    elif clf_name == "Naive Bayes":
        return GaussianNB()
    else:
        return LDA()

clf = get_classifier(classifier_name, params)

# --- Ejecución del Pipeline ---
X = df.drop("target", axis=1)
y = df["target"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado
if do_scaling:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# PCA
if use_pca:
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

# Entrenamiento y Predicción
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- 5 & 6. Validación e Informe de Despliegue ---
col1, col2 = st.columns(2)

with col1:
    st.write(f"#### Resultados: {classifier_name}")
    st.metric("Accuracy (Exactitud)", f"{acc:.2%}")
    
    if acc > 0.85:
        st.success("✅ **Modelo con desempeño Alto.** Listo para despliegue en producción.")
    elif acc > 0.70:
        st.warning("⚠️ **Desempeño Medio.** Se recomienda ajustar hiperparámetros.")
    else:
        st.error("❌ **Desempeño Bajo.** No apto para despliegue.")

with col2:
    # --- 7. Gráficas de Desempeño ---
    st.write("#### Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig)

# Visualización de Clústeres (si hay PCA)
if use_pca and n_components >= 2:
    st.write("#### Visualización de Características (PCA)")
    pca_df = pd.DataFrame(X_train[:, :2], columns=["PC1", "PC2"])
    pca_df['target'] = y_train.values
    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color=pca_df['target'].astype(str),
                         title="Espacio de Características (Top 2 PCs)")
    st.plotly_chart(fig_pca)
