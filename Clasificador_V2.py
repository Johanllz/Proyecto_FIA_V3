import joblib
import cv2
import numpy as np

# Consejos para cada enfermedad
disease_tips = {
    "Apple___Apple_scab": "La sarna del manzano, también conocida como sarna del manzano, es una enfermedad común de los manzanos causada por el hongo Venturia inaequalis. Los síntomas incluyen manchas oscuras en las hojas y frutas. Para controlar la sarna del manzano, se recomienda la aplicación de fungicidas a base de cobre y la eliminación de hojas y frutas infectadas.",
    "Apple___Black_rot": "La podredumbre negra es una enfermedad fúngica que afecta a los manzanos. Los síntomas incluyen manchas circulares en las hojas y frutas, que luego se vuelven negras y podridas. Para controlar la podredumbre negra, se deben eliminar y destruir las frutas y hojas infectadas y se pueden aplicar fungicidas específicos.",
    "Apple___Cedar_apple_rust": "La roya del manzano es una enfermedad causada por el hongo Gymnosporangium juniperi-virginianae, que afecta a los manzanos y los cedros. Los síntomas incluyen manchas de óxido en las hojas y frutas. Para controlar la roya del manzano, se deben usar fungicidas y eliminar los árboles de enebro cercanos.",
    "Apple___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como el riego y la fertilización, para mantenerla saludable.",
    "Blueberry___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la poda y la gestión de malezas, para mantenerla saludable.",
    "Cherry___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la poda y la gestión de plagas, para mantenerla saludable.",
    "Cherry___Powdery_mildew": "El mildiu polvoriento es una enfermedad fúngica que afecta a las cerezas. Los síntomas incluyen un polvo blanco en las hojas y frutas. Para controlar el mildiu polvoriento, se pueden aplicar fungicidas específicos y mejorar la circulación de aire alrededor de la planta.",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "La mancha foliar de Cercospora, también conocida como mancha gris, es una enfermedad común del maíz causada por el hongo Cercospora zeae-maydis. Los síntomas incluyen manchas foliares grises en las hojas. Para controlar la mancha foliar de Cercospora, se recomienda la rotación de cultivos y la aplicación de fungicidas.",
    "Corn___Common_rust": "La roya común del maíz es una enfermedad fúngica causada por el hongo Puccinia sorghi. Los síntomas incluyen pústulas de color naranja en las hojas y tallos. Para controlar la roya común, se pueden usar variedades resistentes y aplicar fungicidas específicos.",
    "Corn___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la rotación de cultivos y la gestión de malezas, para mantenerla saludable.",
    "Corn___Northern_Leaf_Blight": "La mancha foliar del maíz del norte es una enfermedad fúngica causada por el hongo Exserohilum turcicum. Los síntomas incluyen manchas foliares marrones en las hojas. Para controlar la mancha foliar del maíz del norte, se recomienda la rotación de cultivos y el uso de fungicidas específicos.",
    "Grape___Black_rot": "La podredumbre negra de la uva es una enfermedad fúngica causada por el hongo Guignardia bidwellii. Los síntomas incluyen manchas circulares en las hojas y frutas, que luego se vuelven negras y podridas. Para controlar la podredumbre negra, se deben eliminar partes afectadas y aplicar fungicidas específicos.",
    "Grape___Esca_(Black_Measles)": "La esca, también conocida como sarna negra, es una enfermedad fúngica que afecta a las uvas. Los síntomas incluyen manchas negras en las hojas y troncos. Para controlar la esca, se deben eliminar y destruir las partes afectadas y se pueden aplicar fungicidas específicos.",
    "Grape___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la poda y la gestión del riego, para mantenerla saludable.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "La mancha foliar de Isariopsis, también conocida como mancha foliar de hollín, es una enfermedad fúngica común de las uvas. Los síntomas incluyen manchas foliares marrones en las hojas. Para controlar la mancha foliar de Isariopsis, se recomienda la aplicación de fungicidas y la eliminación de hojas afectadas.",
    "Orange___Haunglongbing_(Citrus_greening)": "La enfermedad del huanglongbing, también conocida como enverdecimiento de los cítricos, es una enfermedad bacteriana devastadora de los cítricos. Los síntomas incluyen hojas amarillentas y frutas verdes y deformadas. Para controlar el huanglongbing, se deben eliminar los árboles infectados y controlar los vectores de la enfermedad.",
    "Peach___Bacterial_spot": "La mancha bacteriana del durazno es una enfermedad bacteriana que afecta a los duraznos. Los síntomas incluyen manchas circulares en las hojas y frutas. Para controlar la mancha bacteriana del durazno, se pueden aplicar tratamientos a base de cobre y mejorar la circulación de aire alrededor de la planta.",
    "Peach___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la poda y la gestión de plagas, para mantenerla saludable.",
    "Pepper,_bell___Bacterial_spot": "La mancha bacteriana del pimiento, también conocida como mancha foliar bacteriana, es una enfermedad bacteriana que afecta a los pimientos. Los síntomas incluyen manchas circulares en las hojas y frutas. Para controlar la mancha bacteriana del pimiento, se pueden usar semillas libres de patógenos y aplicar bactericidas específicos.",
    "Pepper,_bell___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la fertilización y la irrigación, para mantenerla saludable.",
    "Potato___Early_blight": "La mancha temprana de la patata es una enfermedad fúngica común de las patatas causada por el hongo Alternaria solani. Los síntomas incluyen manchas oscuras en las hojas. Para controlar la mancha temprana de la patata, se recomienda la rotación de cultivos y la aplicación de fungicidas específicos.",
    "Potato___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la rotación de cultivos y la gestión de malezas, para mantenerla saludable.",
    "Potato___Late_blight": "La gota de la patata es una enfermedad fúngica destructiva causada por el oomycete Phytophthora infestans. Los síntomas incluyen manchas marrones en las hojas y tubérculos. Para controlar la gota de la patata, se deben eliminar y destruir las plantas afectadas y se pueden aplicar fungicidas específicos.",
    "Raspberry___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la poda y la gestión del suelo, para mantenerla saludable.",
    "Soybean___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la rotación de cultivos y la gestión de malezas, para mantenerla saludable.",
    "Squash___Powdery_mildew": "El mildiu polvoriento es una enfermedad fúngica común de las cucurbitáceas, como las calabazas. Los síntomas incluyen un polvo blanco en las hojas y tallos. Para controlar el mildiu polvoriento, se recomienda la aplicación de fungicidas específicos y la mejora de la circulación de aire alrededor de la planta.",
    "Strawberry___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la irrigación y la poda, para mantenerla saludable.",
    "Strawberry___Leaf_scorch": "El marchitamiento de la hoja de la fresa es una enfermedad fúngica que afecta a las fresas. Los síntomas incluyen bordes marrones en las hojas. Para controlar el marchitamiento de la hoja de la fresa, se pueden eliminar las hojas afectadas y mejorar la circulación de aire alrededor de la planta.",
    "Tomato___Bacterial_spot": "La mancha bacteriana del tomate es una enfermedad bacteriana que afecta a los tomates. Los síntomas incluyen manchas circulares en las hojas y frutas. Para controlar la mancha bacteriana del tomate, se pueden usar semillas libres de patógenos y aplicar bactericidas específicos.",
    "Tomato___Early_blight": "La mancha temprana del tomate es una enfermedad fúngica común de los tomates causada por el hongo Alternaria solani. Los síntomas incluyen manchas marrones en las hojas. Para controlar la mancha temprana del tomate, se recomienda la rotación de cultivos y la aplicación de fungicidas específicos.",
    "Tomato___healthy": "La planta está sana. Asegúrate de mantener prácticas de cultivo adecuadas, como la rotación de cultivos y la poda, para mantenerla saludable.",
    "Tomato___Late_blight": "La gota tardía del tomate es una enfermedad fúngica destructiva causada por el oomycete Phytophthora infestans. Los síntomas incluyen manchas marrones en las hojas y frutas. Para controlar la gota tardía del tomate, se deben eliminar y destruir las plantas afectadas y se pueden aplicar fungicidas específicos.",
    "Tomato___Leaf_Mold": "El moho de la hoja del tomate es una enfermedad fúngica que afecta a los tomates. Los síntomas incluyen manchas amarillas en las hojas. Para controlar el moho de la hoja del tomate, se recomienda mejorar la circulación de aire y aplicar fungicidas específicos.",
    "Tomato___Septoria_leaf_spot": "La mancha foliar de Septoria es una enfermedad fúngica común de los tomates causada por el hongo Septoria lycopersici. Los síntomas incluyen manchas circulares con un centro blanco en las hojas. Para controlar la mancha foliar de Septoria, se pueden eliminar las hojas afectadas y aplicar fungicidas específicos.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Los ácaros araña, como el ácaro araña de dos manchas, son plagas comunes de los tomates. Los síntomas incluyen manchas amarillas en las hojas y telarañas. Para controlar los ácaros araña, se pueden usar acaricidas específicos y mejorar la circulación de aire alrededor de la planta.",
    "Tomato___Target_Spot": "La mancha objetivo del tomate es una enfermedad fúngica común de los tomates causada por el hongo Corynespora cassiicola. Los síntomas incluyen manchas circulares con un centro oscuro en las hojas. Para controlar la mancha objetivo del tomate, se pueden eliminar las hojas afectadas y aplicar fungicidas específicos.",
    "Tomato___Tomato_mosaic_virus": "El virus del mosaico del tomate es una enfermedad viral común de los tomates. Los síntomas incluyen manchas amarillas y deformaciones en las hojas y frutas. Para controlar el virus del mosaico del tomate, se deben eliminar las plantas infectadas y controlar los vectores de la enfermedad.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "El virus del rizado amarillo de las hojas del tomate es una enfermedad viral devastadora de los tomates. Los síntomas incluyen hojas amarillentas y rizadas. Para controlar el virus del rizado amarillo de las hojas del tomate, se deben eliminar las plantas infectadas y controlar los vectores de la enfermedad."
}


def load_model(model_path):
    return joblib.load(model_path)

def preprocess_image(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        img = img.reshape(1, -1)
        return img
    else:
        raise ValueError(f"Could not read image {image_path}")

def predict_disease(model, image_path):
    knn, scaler, label_encoder, pca = model
    img = preprocess_image(image_path)
    img = scaler.transform(img)
    img = pca.transform(img)
    prediction = knn.predict(img)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class, disease_tips[predicted_class]

if __name__ == "__main__":
    model_path = 'knn_model.pkl'
    image_paths = [
        'pruebas/Potato_LB3.jpg',
        #'image2.jpg'
    ]

    model = load_model(model_path)

    for image_path in image_paths:
        try:
            predicted_class, tips = predict_disease(model, image_path)
            print(f"Image: {image_path}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Tips: {tips}\n")
        except ValueError as e:
            print(e)
