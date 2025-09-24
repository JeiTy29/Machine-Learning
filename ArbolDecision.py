import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# =============================
# 1. Cargar dataset
# =============================
data = pd.read_csv("C:/Users/thoma/Documents/Archivos/Universidad/Octavo Semestre/Machine Learning/Trabajos/Datasets/emails_dataset.csv")

# Features y etiqueta
X = data.drop("label", axis=1)
y = data["label"]

# Convertimos la etiqueta a binaria (spam=1, ham=0)
y = y.map({"ham": 0, "spam": 1})

# =============================
# 2. Entrenamiento en 50 ejecuciones
# =============================
n_runs = 50
accuracy_list = []
f1_list = []
conf_matrices = []

for i in range(n_runs):
    # Dividir dataset (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=i
    )
    
    # Árbol de decisión
    clf = DecisionTreeClassifier(random_state=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    accuracy_list.append(acc)
    f1_list.append(f1)
    
    # Guardamos matriz de confusión
    conf_matrices.append(confusion_matrix(y_test, y_pred))

# =============================
# 3. Cálculo de Z-scores
# =============================
accuracy_array = np.array(accuracy_list)
f1_array = np.array(f1_list)

mean_acc = np.mean(accuracy_array)
std_acc = np.std(accuracy_array)
z_scores = (accuracy_array - mean_acc) / std_acc

# =============================
# 4. Resultados en consola
# =============================
print("\n===== RESULTADOS DE 50 EJECUCIONES =====\n")

print(f"Accuracy promedio: {mean_acc:.4f}")
print(f"Accuracy desviación estándar: {std_acc:.4f}")
print(f"Mejor Accuracy: {np.max(accuracy_array):.4f} (Ejecución {np.argmax(accuracy_array)+1})")
print(f"Peor Accuracy: {np.min(accuracy_array):.4f} (Ejecución {np.argmin(accuracy_array)+1})")

print("\n-----------------------------------------\n")

print(f"F1-score promedio: {np.mean(f1_array):.4f}")
print(f"F1-score desviación estándar: {np.std(f1_array):.4f}")
print(f"Mejor F1-score: {np.max(f1_array):.4f} (Ejecución {np.argmax(f1_array)+1})")
print(f"Peor F1-score: {np.min(f1_array):.4f} (Ejecución {np.argmin(f1_array)+1})")

print("\n-----------------------------------------\n")

print(f"Z-score promedio: {np.mean(z_scores):.4f} (debe ser ≈ 0)")
print(f"Rango de Z-scores: {np.min(z_scores):.2f} a {np.max(z_scores):.2f}")

# =============================
# 5. Gráficas
# =============================

# Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_runs+1), accuracy_array, marker='o', label="Accuracy")
plt.axhline(mean_acc, color='red', linestyle='--', label=f"Media={mean_acc:.2f}")
plt.title("Accuracy en 50 ejecuciones")
plt.xlabel("Ejecución")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# F1-score
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_runs+1), f1_array, marker='s', color="green", label="F1-score")
plt.axhline(np.mean(f1_array), color='red', linestyle='--', label=f"Media={np.mean(f1_array):.2f}")
plt.title("F1-score en 50 ejecuciones")
plt.xlabel("Ejecución")
plt.ylabel("F1-score")
plt.legend()
plt.show()

# Z-scores
plt.figure(figsize=(10, 5))
plt.bar(range(1, n_runs+1), z_scores, color="purple")
plt.axhline(0, color="black", linestyle="--")
plt.title("Z-scores de Accuracy en 50 ejecuciones")
plt.xlabel("Ejecución")
plt.ylabel("Z-score")
plt.show()

# =============================
# 6. Matriz de confusión promedio
# =============================
mean_conf_matrix = np.mean(conf_matrices, axis=0)

plt.figure(figsize=(6, 5))
sns.heatmap(mean_conf_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Matriz de confusión promedio (50 ejecuciones)")
plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.show()
