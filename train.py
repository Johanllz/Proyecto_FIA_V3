import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from preprocess import load_and_preprocess_data

def train_knn_model(data_dir, model_path, img_size=(128, 128), n_components=30, k=4, samples_per_class=600):
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_dir, img_size, samples_per_class)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    joblib.dump((knn, scaler, label_encoder, pca), model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    data_dir = 'F:/Proyecto_FIA/Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation'  # Cambia esta línea con la ruta correcta a tu dataset
    model_path = 'knn_model.pkl'
    train_knn_model(data_dir, model_path)
