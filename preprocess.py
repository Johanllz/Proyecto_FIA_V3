import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(data_dir, img_size=(128, 128), samples_per_class=600):
    data = []
    labels = []
    label_encoder = LabelEncoder()
    classes = os.listdir(data_dir)
    classes = [cls for cls in classes if os.path.isdir(os.path.join(data_dir, cls))]
    print(f"Classes found: {classes}")

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        cls_images = os.listdir(cls_dir)
        cls_images = cls_images[:samples_per_class]

        for img_name in cls_images:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, img_size)
                data.append(img)
                labels.append(cls)
            else:
                print(f"Warning: Could not read image {img_path}")

    data = np.array(data)
    labels = np.array(labels)

    # Normalizar datos
    data = data.astype('float32') / 255.0

    # Codificar etiquetas
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)

    # Aplanar imágenes
    data = data.reshape(data.shape[0], -1)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder
