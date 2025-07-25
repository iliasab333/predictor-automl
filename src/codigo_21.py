import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Crear carpeta de salida
output_dir = "C:/Users/Ilias/Desktop/TFG/graficas_finales_seleccionadas"
os.makedirs(output_dir, exist_ok=True)

# Cargar datos
df = pd.read_csv("C:/Users/Ilias/Desktop/TFG/car_sales_limpio.csv")
df['Purchased Date'] = pd.to_datetime(df['Purchased Date'], errors='coerce')
df['Año'] = df['Purchased Date'].dt.year
df['Mes'] = df['Purchased Date'].dt.month

# 1. Evolución del tipo de energía a lo largo del tiempo
df['Year-Month'] = df['Purchased Date'].dt.to_period('M')
energia_mensual = df.groupby(['Year-Month', 'Energy']).size().unstack().fillna(0)
energia_mensual.plot(figsize=(12, 6))
plt.title("Evolución mensual por tipo de energía")
plt.xlabel("Fecha")
plt.ylabel("Número de vehículos vendidos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/energia_mensual_lineas.png")
plt.close()

# 2. Comparación de ventas acumuladas entre marcas líderes
top_3_marcas = df['Manufacturer Name'].value_counts().head(3).index
df_top_marcas = df[df['Manufacturer Name'].isin(top_3_marcas)]
df_top_marcas = df_top_marcas.copy()
df_top_marcas['Year-Month'] = df_top_marcas['Purchased Date'].dt.to_period('M')

ventas_marcas = df_top_marcas.groupby(['Year-Month', 'Manufacturer Name']).size().unstack().fillna(0).cumsum()
ventas_marcas.plot(figsize=(12, 6))
plt.title("Ventas acumuladas de las 3 marcas principales")
plt.xlabel("Fecha")
plt.ylabel("Ventas acumuladas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/ventas_acumuladas_top3_marcas.png")
plt.close()

# 3. Boxplot: precio de eléctricos vs gasolina
df_filtrado = df[df['Energy'].isin(['Electric', 'Petrol'])]
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_filtrado, x='Energy', y='Price-$')
plt.title("Comparación de precios: Eléctricos vs Gasolina")
plt.ylabel("Precio ($)")
plt.tight_layout()
plt.savefig(f"{output_dir}/boxplot_electricos_vs_petrol.png")
plt.close()

# 4. Correlación entre precio y año del coche
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Manufactured Year', y='Price-$', alpha=0.6)
sns.regplot(data=df, x='Manufactured Year', y='Price-$', scatter=False, color='red')
plt.title("Relación entre año de fabricación y precio")
plt.xlabel("Año de fabricación")
plt.ylabel("Precio ($)")
plt.tight_layout()
plt.savefig(f"{output_dir}/precio_vs_anio_fabricacion.png")
plt.close()

# 5. Predicción de ventas con Random Forest
from sklearn.ensemble import RandomForestRegressor

ventas_mensuales = df.groupby('Year-Month').size().reset_index(name='Total_Ventas')
ventas_mensuales['Year-Month'] = ventas_mensuales['Year-Month'].astype(str)
ventas_mensuales = ventas_mensuales.sort_values('Year-Month')
ventas_mensuales['Mes_ordinal'] = np.arange(len(ventas_mensuales))

X = ventas_mensuales[['Mes_ordinal']]
y = ventas_mensuales['Total_Ventas']
modelo = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_leaf=2, random_state=42)
modelo.fit(X, y)

# Predicción 6 meses adelante
future_months = pd.DataFrame({'Mes_ordinal': np.arange(len(X), len(X)+6)})
future_preds = modelo.predict(future_months)

plt.figure(figsize=(12, 6))
plt.plot(ventas_mensuales['Mes_ordinal'], y, label='Histórico')
plt.plot(future_months['Mes_ordinal'], future_preds, linestyle='--', marker='o', color='red', label='Predicción 6 meses')
plt.title("Predicción de ventas con Random Forest")
plt.xlabel("Mes ordinal")
plt.ylabel("Total de ventas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/prediccion_ventas_6_meses.png")
plt.close()