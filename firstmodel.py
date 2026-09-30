import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('creditcard.csv').dropna(axis=0)

X = df.drop(columns=['Class'])
y = df['Class']

print("Filas y columnas en X:", X.shape)
print("Target (y) muestreo:")
print(y.head())

model = DecisionTreeRegressor(random_state=1)
print("Entrenando modelo...")
model.fit(X, y)
print("¡Entrenamiento completado!")

print("\nPredicciones para los primeros 5 registros:")
print(model.predict(X.head()))
print("Valores reales:")
print(y.head().values)
