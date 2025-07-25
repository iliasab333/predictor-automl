import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar los datos
df = pd.read_csv("C:/Users/Ilias/Desktop/TFG/ventas_mensuales.csv")

# Asegurar que la columna de fechas está en formato datetime
df['Year-Month'] = pd.to_datetime(df['Year-Month'])

# Ordenar cronológicamente y crear una columna numérica del tiempo
df = df.sort_values("Year-Month")
df["Mes_ordinal"] = np.arange(len(df))

# Variables
X = df[["Mes_ordinal"]]
y = df["Total_Ventas"]

# Separar datos en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Calcular métricas
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Mostrar métricas
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualización: puntos reales + línea de tendencia
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label="Valores reales")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Línea de tendencia")
plt.title("Predicción de ventas con Regresión Lineal")
plt.xlabel("Mes ordinal")
plt.ylabel("Total de ventas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

