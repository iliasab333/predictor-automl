import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Puedes cambiarlo por otro modelo
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Cargar y preparar los datos
df = pd.read_csv("C:/Users/Ilias/Desktop/TFG/ventas_mensuales.csv")
df['Year-Month'] = pd.to_datetime(df['Year-Month'])
df = df.sort_values("Year-Month")
df['Mes_ordinal'] = np.arange(len(df))

# 2. Variables predictoras y variable objetivo
X = df[['Mes_ordinal']]
y = df['Total_Ventas']

# 3. División en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 5. Predicción sobre el test
y_pred = modelo.predict(X_test)

# 6. Cálculo de métricas
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 7. Visualización
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Valores reales", color='blue')
plt.scatter(X_test, y_pred, label="Predicciones", color='green', marker='x')
plt.plot(X_test.sort_values(by="Mes_ordinal"), 
         modelo.predict(X_test.sort_values(by="Mes_ordinal")), 
         color='red', linewidth=2, label="Línea de tendencia")
plt.title("Predicción de ventas (Regresión Lineal)")
plt.xlabel("Mes ordinal")
plt.ylabel("Total de ventas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
