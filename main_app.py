import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scikitplot as skplt

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_curve, auc)

# Modelos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Advanced ML Prototyper", layout="wide")

# --- UI LATERAL ---
st.sidebar.title("⚙️ Configuración Avanzada")

# 1. Selección de Datos
dataset_name = st.sidebar.selectbox("Dataset", ["Breast Cancer", "Iris", "Wine"])

# 2. Preproceso y PCA por Varianza
st.sidebar.subheader("Preprocesamiento")
apply_scaling = st.sidebar.toggle("Escalar Datos", value=True)
apply_pca = st.sidebar.toggle("Aplicar PCA", value=False)

variance_threshold = 0.95
if apply_pca:
    variance_threshold = st.sidebar.slider("Varianza Explicada Deseada", 0.50, 0.99, 0.95, 0.05)

# 3. Validación Cruzada
st.sidebar.subheader("Estrategia de Validación")
cv_type = st.sidebar.selectbox("Tipo de CV", ["Hold-out (Simple)", "K-Fold Cross Validation"])
k_folds = st.sidebar.number_input("Número de Folds (K)", 2, 10, 5) if cv_type == "K-Fold Cross Validation" else None

# 4. Métrica Objetivo
main_metric = st.sidebar.selectbox("Métrica Principal para Reporte", ["Accuracy", "Precision", "Recall", "F1-Score"])

# --- LÓGICA DE DATOS ---
@st.cache_data
def load_and_prep(name):
    if name == "Breast Cancer": data = datasets.load_breast_cancer()
    elif name == "Iris": data = datasets.load_iris()
    else: data = datasets.load_wine()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names

X_raw, y, target_names = load_and_prep(dataset_name)

# --- PIPELINE DE PROCESAMIENTO ---
X = X_raw.copy()
if apply_scaling:
    X = StandardScaler().fit_transform(X)

if apply_pca:
    pca_full = PCA().fit(X)
    # Calcular cuántos componentes se necesitan para la varianza pedida
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    st.sidebar.info(f"PCA redujo a {n_components} componentes para cubrir el {variance_threshold*100:.0f}% de varianza.")

# --- SELECCIÓN DE MODELO ---
st.sidebar.subheader("Modelo")
clf_key = st.sidebar.selectbox("Algoritmo", ["KNN", "Decision Tree", "Naive Bayes", "LDA"])

def get_model(name):
    if name == "KNN": return KNeighborsClassifier(n_neighbors=5)
    if name == "Decision Tree": return DecisionTreeClassifier(max_depth=5)
    if name == "Naive Bayes": return GaussianNB()
    return LDA()

model = get_model(clf_key)

# --- EJECUCIÓN ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)

# Cálculo de métricas
metrics_dict = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average='weighted'),
    "Recall": recall_score(y_test, y_pred, average='weighted'),
    "F1-Score": f1_score(y_test, y_pred, average='weighted')
}

# --- DESPLIEGUE DE RESULTADOS ---
st.title(f"🚀 Dashboard de Clasificación: {clf_key}")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Accuracy", f"{metrics_dict['Accuracy']:.2%}")
col_m2.metric("Precision", f"{metrics_dict['Precision']:.2%}")
col_m3.metric("Recall", f"{metrics_dict['Recall']:.2%}")
col_m4.metric("F1-Score", f"{metrics_dict['F1-Score']:.2%}")

# Verificación de "Despliegue"
performance = metrics_dict[main_metric]
if performance > 0.90:
    st.success(f"🌟 **Modelo Excelente ({main_metric}: {performance:.2%})**. El prototipo está listo para ser exportado a producción.")
elif performance > 0.75:
    st.warning("⚠️ **Desempeño Aceptable.** Considere optimizar hiperparámetros.")
else:
    st.error("❌ **Desempeño Insuficiente.** Revise la calidad de los datos o cambie de modelo.")

# --- VISUALIZACIONES ---
tab1, tab2, tab3 = st.tabs(["📊 Análisis de Error", "📈 Curvas de Rendimiento", "🔍 Estructura de Datos"])

with tab1:
    st.subheader("Matriz de Confusión y Reporte")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        fig_cm, ax_cm = plt.subplots()
        skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, ax=ax_cm)
        st.pyplot(fig_cm)
    with c2:
        st.text("Reporte de Clasificación:")
        st.code(classification_report(y_test, y_pred, target_names=target_names))

with tab2:
    st.subheader("Curvas ROC y Validación")
    c3, c4 = st.columns(2)
    with c3:
        fig_roc, ax_roc = plt.subplots()
        skplt.metrics.plot_roc(y_test, y_probas, ax=ax_roc)
        st.pyplot(fig_roc)
    with c4:
        if cv_type == "K-Fold Cross Validation":
            skf = StratifiedKFold(n_splits=k_folds)
            scores = cross_val_score(model, X, y, cv=skf)
            st.write(f"**Resultados de CV ({k_folds} Folds):**")
            fig_cv, ax_cv = plt.subplots()
            sns.boxplot(data=scores, ax=ax_cv, orient='h', color='skyblue')
            ax_cv.set_title("Estabilidad del Accuracy en CV")
            st.pyplot(fig_cv)
            st.info(f"Media CV: {scores.mean():.2%} (+/- {scores.std():.2%})")

with tab3:
    st.subheader("Distribución de Clases (PCA)")
    if X.shape[1] >= 2:
        pca_df = pd.DataFrame(X[:, :2], columns=["Comp 1", "Comp 2"])
        pca_df['Clase'] = [target_names[i] for i in y]
        fig_scatter = px.scatter(pca_df, x="Comp 1", y="Comp 2", color="Clase", template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("Se requieren al menos 2 componentes para visualizar el espacio.")
