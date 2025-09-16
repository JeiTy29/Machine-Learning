Este repositorio contiene diferentes proyectos y prácticas desarrolladas en el marco de la materia Machine Learning. 
Autores: Thomas Cristancho, Andres Melo

Contenido:

Dataset: 'CorreosSpam.csv': Contiene correos clasificados como spam o ham, con características extraídas de los mensajes (cantidad de enlaces, palabras sospechosas, mayúsculas, longitud del mensaje, dominio sospechoso, etc.).

Codigo 'RegresionLogistica.py': Implementa un modelo de regresión logística para clasificar correos como spam o no spam. El script incluye: Preprocesamiento y normalización de datos. Entrenamiento del modelo. Evaluación usando F1-score y matriz de confusión. Identificación de las características más importantes.

Codigo 'RegresionLinealIris.py': Implementa un modelo de Regresión Lineal en Python utilizando `scikit-learn` para clasificar las tres especies de flores del dataset Iris (setosa, versicolor y virginica).  Aunque la regresión lineal está diseñada para predicción de variables continuas, aquí se adapta como clasificador mediante One-Hot Encoding de las etiquetas y la selección de clases con `argmax`.  El código entrena el modelo, evalúa su desempeño con métricas de clasificación, genera la **matriz de confusión** (en consola y en gráfico) y muestra ejemplos de predicciones individuales.
