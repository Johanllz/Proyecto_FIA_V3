import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from preprocess import load_and_preprocess_data

def visualize_knn(data_dir, model_path, img_size=(128, 128), samples_per_class=120):
    # Cargar datos de prueba
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_dir, img_size, samples_per_class)

    # Cargar modelo entrenado
    knn, scaler, _, pca = joblib.load(model_path)
    X_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test)
    X_train_pca = pca.transform(scaler.transform(X_train))

    # Seleccionar una muestra de prueba
    test_sample = X_test_pca[0].reshape(1, -1)
    y_test_sample = y_test[0]

    # Encontrar los k vecinos más cercanos
    distances, indices = knn.kneighbors(test_sample)

    # Visualización
    plt.figure(figsize=(10, 8))

    # Colores para las clases
    cmap = ListedColormap(plt.get_cmap('viridis', len(np.unique(y_train))).colors)
    scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap=cmap, alpha=0.5)
    plt.colorbar(scatter, label='Class labels')

    # Dibujar la muestra de prueba
    plt.scatter(test_sample[0, 0], test_sample[0, 1], c='red', edgecolor='black', s=200, label='Test sample', marker='X')

    # Dibujar los vecinos más cercanos
    for i in range(len(indices[0])):
        if indices[0][i] < len(X_train_pca):
            plt.plot([test_sample[0, 0], X_train_pca[indices[0][i], 0]], [test_sample[0, 1], X_train_pca[indices[0][i], 1]], 'k--', alpha=0.6)
            plt.scatter(X_train_pca[indices[0][i], 0], X_train_pca[indices[0][i], 1], c='yellow', edgecolor='black', s=100, label='Neighbor' if i == 0 else '')

    plt.title('KNN Visualization with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_dir = 'F:/Proyecto_FIA/Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation'
    model_path = 'knn_model.pkl'
    visualize_knn(data_dir, model_path)
