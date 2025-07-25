import pandas as pd

# Ruta base
ruta_base = "C:/Users/Ilias/Desktop/TFG/"

# Cargar el dataset original
df = pd.read_csv(ruta_base + "used_car_sales.csv")

# Convertir la columna de fechas
df['Purchased Date'] = pd.to_datetime(df['Purchased Date'], errors='coerce')

# Filtrar los datos desde 2015 hasta 2024
df = df[(df['Purchased Date'] >= '2015-01-01') & (df['Purchased Date'] <= '2024-12-31')]

# Crear columna con año y mes
df['Year-Month'] = df['Purchased Date'].dt.to_period('M')

# Columnas innecesarias
columns_to_drop = [
    'ID', 'Distributor Name', 'Sales Agent Name',
    'Sold Date', 'Sold Price-$', 'Purchased Price-$',
    'Margin-%', 'Sales Commission-$', 'Profit-$',
    'Feedback', 'Car Sale Status', 'Sales Rating',
    'Engine Power-HP', 'Mileage-KM'
]

# Filtrar columnas existentes
existing_columns = [col for col in columns_to_drop if col in df.columns]
df_cleaned = df.drop(columns=existing_columns)

# Agrupar por año/mes y contar ventas
ventas_mensuales = df_cleaned.groupby('Year-Month').size().reset_index(name='Total_Ventas')

# Guardar archivos en carpeta TFG
ventas_mensuales.to_csv(ruta_base + "ventas_mensuales.csv", index=False)
df_cleaned.to_csv(ruta_base + "car_sales_limpio.csv", index=False)

print("✅ Archivos guardados correctamente en la carpeta TFG.")
