import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Usamos Classifier porque es detección de fraude (0 o 1)
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Cargar los datos
file_path = 'creditcard.csv'
data = pd.read_csv(file_path)

# 2. Elegir la variable objetivo (target) y las características (features)
# En el dataset creditcard, la columna objetivo suele llamarse 'Class' (Fraude o No Fraude)
y = data['Class']
X = data.drop(columns=['Class'])

# 3. Hacer el split (División de datos para entrenamiento y validación)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# 4. Definir el modelo
modelo = DecisionTreeClassifier(random_state=1)

# 5. Entrenar el modelo (Fit)
modelo.fit(train_X, train_y)

# 6. Predecir
predicciones = modelo.predict(val_X)

# 7. Validación del modelo (MAE)
mae = mean_absolute_error(val_y, predicciones)
print(f"Error Absoluto Medio (MAE): {mae}")
