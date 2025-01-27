import os
import cv2
import pandas as pd
import numpy as np
import pickle
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

class ImageFeatureExtractor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
    
    def extract_features(self, image_path):
        """
        Extrae características de una imagen usando GLCM y características morfológicas
        """
        try:
            # Leer y preprocesar la imagen
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Error al cargar la imagen: {image_path}")
            
            img = cv2.resize(img, self.target_size)
            
            # Extraer características GLCM
            glcm = graycomatrix(img, [1], [0, 45, 90, 135], 
                              symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            energy = graycoprops(glcm, 'energy').mean()
            
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            area = np.sum(binary > 0)
            perimeter = sum(cv2.arcLength(cont, True) for cont in contours)
            
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            
            return [area, perimeter, contrast, homogeneity, 
                    correlation, energy, mean_intensity, std_intensity]
            
        except Exception as e:
            print(f"Error procesando {image_path}: {str(e)}")
            return None
    
    def get_feature_names(self):
        return ['area', 'perimeter', 'contrast', 'homogeneity', 
                'correlation', 'energy', 'mean_intensity', 'std_intensity']

class ImageClassifier:
    def __init__(self, dataset_dir, class_names):
        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.label_map = {i: name for i, name in enumerate(class_names)}
        self.feature_extractor = ImageFeatureExtractor()
        self.model = None

    def load_dataset(self):
        """
        Carga y procesa todas las imágenes del dataset
        """
        features = []
        labels = []

        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(self.dataset_dir, class_name)
            print(f"Procesando clase: {class_name}")

            if not os.path.exists(class_folder):
                raise ValueError(f"No se encontró el directorio: {class_folder}")

            images = [f for f in os.listdir(class_folder)
                      if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

            for image_name in images:
                img_path = os.path.join(class_folder, image_name)
                feature = self.feature_extractor.extract_features(img_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(label)

        columns = self.feature_extractor.get_feature_names()
        self.df = pd.DataFrame(features, columns=columns)
        self.df['label'] = pd.Series(labels).map(self.label_map)

        return self.df

    def train(self, n_neighbors=3, test_size=0.2, random_state=42):
        """
        Entrena el modelo KNN con validación cruzada
        """
        X = self.df.drop(columns=['label'])
        y = self.df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Crear pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

        # Entrenar y evaluar
        self.model.fit(X_train, y_train)

        # Validación cruzada
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        # Predicciones y métricas
        y_pred = self.model.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred)
        }

        return results

    def train_without_mix(self, n_neighbors=3, test_size=0.2, random_state=42):
        """
        Entrena el modelo KNN excluyendo la clase `mix`.
        """
        # Filtrar dataset excluyendo `mix`
        self.df = self.df[self.df['label'] != 'mix']

        # Preparar datos
        X = self.df.drop(columns=['label'])
        y = self.df['label']

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Crear pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
        ])

        # Entrenar y evaluar
        self.model.fit(X_train, y_train)

        # Validación cruzada
        cv_scores = cross_val_score(self.model, X, y, cv=5)

        # Predicciones y métricas
        y_pred = self.model.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                y_test, y_pred, labels=self.df['label'].unique())
        }

        return results
    
def predict(self, image_path):
        """
        Realiza predicción para una nueva imagen
        """
        if not self.model:
            raise ValueError("El modelo no ha sido entrenado")
            
        features = self.feature_extractor.extract_features(image_path)
        if features is None:
            return None
            
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'predicted_class': prediction,
            'probabilities': {
                class_name: prob 
                for class_name, prob in zip(self.class_names, probabilities)
            }
        }

if __name__ == "__main__":
    dataset_dir = 'dataset'
    class_names = ['perfumes', 'tenis', 'gorras', 'playeras', 'funkos', 'mix']
    
    classifier = ImageClassifier(dataset_dir, class_names)
    
    print("Cargando dataset...")
    df = classifier.load_dataset()
    print("\nResumen del dataset:")
    print(df['label'].value_counts())
    
print("\nEntrenando modelo excluyendo la clase 'mix'...")
results = classifier.train_without_mix(n_neighbors=3)

print("\nResultados del entrenamiento (sin 'mix'):")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Cross-validation mean: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
print("\nReporte de clasificación:")
print(results['classification_report'])

with open('model.pkl', 'wb') as file:
    pickle.dump(classifier.model, file)