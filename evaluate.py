import joblib
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_and_preprocess_data

def evaluate_knn_model(data_dir, model_path, img_size=(128, 128), samples_per_class=120):
    _, X_test, _, y_test, label_encoder = load_and_preprocess_data(data_dir, img_size, samples_per_class)

    knn, scaler, _, pca = joblib.load(model_path)
    X_test = scaler.transform(X_test)
    X_test = pca.transform(X_test)

    y_pred = knn.predict(X_test)

   

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    data_dir = 'F:/Proyecto_FIA/Plant_leaf_diseases_dataset_with_augmentation/Plant_leave_diseases_dataset_with_augmentation'
    model_path = 'knn_model.pkl'
    evaluate_knn_model(data_dir, model_path)
