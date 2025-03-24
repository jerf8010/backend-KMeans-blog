import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# Función para reiniciar la aplicación
def reset():
    st.session_state.iteration = 0
    st.session_state.kmeans = None
    st.session_state.prev_centers = None
    st.session_state.converged = False
    st.rerun()


# Título del dashboard
st.title('Visualización del Algoritmo KMeans Paso a Paso')

# Inicializar variables en session_state si es la primera vez
if 'iteration' not in st.session_state:
    st.session_state.iteration = 0
    st.session_state.kmeans = None
    st.session_state.prev_centers = None
    st.session_state.converged = False

# Determinar si los controles deben estar deshabilitados
controls_disabled = st.session_state.iteration > 0

# Sidebar para configurar el dataset
st.sidebar.header('Parámetros del Dataset')
n_samples = st.sidebar.slider('Número de muestras', 100, 1000, 500,
                              disabled=controls_disabled)
variabilidad = st.sidebar.slider('Variabilidad del dataset', 1, 5, 2,
                                 disabled=controls_disabled)
n_clusters = st.sidebar.slider('Número de clusters', 2, 10, 4,
                               disabled=controls_disabled)
random_state = st.sidebar.slider('Random State: ', 0, 100, 42,
                                 disabled=controls_disabled)

# Generar datos sintéticos
X, _ = make_blobs(n_samples=n_samples, centers=4,
                  cluster_std=variabilidad, random_state=random_state)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicializar el modelo KMeans si es la primera vez
if st.session_state.kmeans is None:
    st.session_state.kmeans = KMeans(n_clusters=n_clusters, init='random',
                                     n_init=1, max_iter=1,
                                     random_state=random_state)
    st.session_state.kmeans.fit(X_scaled)
    st.session_state.prev_centers = None  # Aún no se guarda el primer centroide  # noqa: E501

# Obtener el modelo y los centros actuales
kmeans = st.session_state.kmeans
centers = kmeans.cluster_centers_

# Crear el gráfico
fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1],
                     c=kmeans.labels_ if st.session_state.iteration > 0
                     else 'lightgray',
                     cmap='viridis', s=50, label='Puntos')

# Mostrar los centros de los clusters
ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200,
           alpha=0.5, marker='x', label='Centros')

ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.set_title(f'Iteración {st.session_state.iteration}: Centros de Clusters')
ax.legend(loc='upper left')

# Mostrar la gráfica en Streamlit
st.pyplot(fig)

# Verificar convergencia solo después de la segunda iteración
if st.session_state.iteration > 1 and st.session_state.prev_centers is not None:  # noqa: E501
    if np.allclose(st.session_state.prev_centers, centers):
        st.session_state.converged = True
        st.success(f"✅ El algoritmo ha convergido en {st.session_state.iteration} iteraciones.")  # noqa: E501

# Crear columnas para los botones
col1, col2 = st.columns([1, 2])  # Dos columnas de diferente tamaño

with col1:
    if st.button('Siguiente Iteración', disabled=st.session_state.converged):
        st.session_state.prev_centers = np.copy(centers)  # Guardar centros anteriores # noqa: E501
        st.session_state.iteration += 1
        st.session_state.kmeans.max_iter = st.session_state.iteration
        st.session_state.kmeans.fit(X_scaled)  # Ajustar con más iteraciones
        st.rerun()  # Reiniciar el script para actualizar la gráfica y bloquear los controles # noqa: E501

with col2:
    if st.button('Reiniciar'):
        reset()
