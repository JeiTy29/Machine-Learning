import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# === 1. Cargar dataset Iris ===
iris = load_iris()
X = iris.data
y = iris.target

# === 2. One-hot encoding ===
encoder = OneHotEncoder(sparse_output=False)  
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# === 3. Train/Test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.3, random_state=42)

# === 4. Entrenar modelo ===
model = LinearRegression()
model.fit(X_train, y_train)

# === 5. Predicciones ===
y_pred_continuo = model.predict(X_test)
y_pred_clases = np.argmax(y_pred_continuo, axis=1)
y_test_clases = np.argmax(y_test, axis=1)

# === 6. Métricas ===
acc = accuracy_score(y_test_clases, y_pred_clases)
print(f"Precisión del modelo: {acc:.2f}\n")

# Reporte detallado
print("=== Reporte de clasificación ===")
print(classification_report(y_test_clases, y_pred_clases, target_names=iris.target_names))

# === 7. Matriz de confusión ===
cm = confusion_matrix(y_test_clases, y_pred_clases)
print("=== Matriz de confusión (en consola) ===")
print(cm)

# También mostrar como gráfico
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusión - Regresión Lineal en Iris")
plt.show()

# === Mostrar predicciones en test ===
print("\n=== Predicciones de prueba (primeros 10 casos) ===")
for i in range(10):
    real = iris.target_names[y_test_clases[i]]
    pred = iris.target_names[y_pred_clases[i]]
    print(f"Real: {real:10s} -> Predicho: {pred}")
