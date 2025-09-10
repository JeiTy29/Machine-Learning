import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score

df = pd.read_csv("C:/Users/thoma/Documents/Archivos/Universidad/Octavo Semestre/Machine Learning/Trabajos/CorreoSpam.csv")

# Features y target
X = df.drop(columns=["Destinatario", "Remitente", "Asunto", "Mensaje", "Clase"])  
y = df["Clase"].map({"ham": 0, "spam":1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_probs = model.predict_proba(X_test)[:, 1]  # prob de ser SPAM

# Evaluamos F1-score para distintos umbrales
thresholds = np.linspace(0.1, 0.9, 9)
scores = []

for t in thresholds:
    y_pred_t = (y_probs >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    scores.append((t, f1))

best_t, best_f1 = max(scores, key=lambda x: x[1])
print(f"\nMejor umbral: {best_t:.2f} con F1-score = {best_f1:.4f}")

# Usamos el mejor umbral
y_pred = (y_probs >= best_t).astype(int)

print("\nClasificación:")
print(classification_report(y_test, y_pred, target_names=["No Spam", "Spam"]))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(pd.DataFrame(cm, index=["Real: No Spam","Real: Spam"], columns=["Pred: No Spam","Pred: Spam"]))

coef = model.coef_[0]
features = X.columns

# Importancia relativa en porcentaje
importance = np.abs(coef) / np.sum(np.abs(coef)) * 100
feature_importance = pd.DataFrame({"Feature": features, "Importancia (%)": importance})
feature_importance = feature_importance.sort_values(by="Importancia (%)", ascending=False)

print("\nTop 10 Features más influyentes (%):")
print(feature_importance.head(10).to_string(index=False))

corr = pd.DataFrame(X_scaled, columns=features).corr().round(2)
print("\nMatriz de correlación (primeras 10 filas):")
print(corr.head(10))
