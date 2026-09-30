import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('creditcard.csv').dropna(axis=0)

X = df.drop(columns=['Class'])
y = df['Class']

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

predictions = model.predict(X)
mae_in_sample = mean_absolute_error(y, predictions)
print("MAE In-Sample (Entrenamiento):", mae_in_sample)
