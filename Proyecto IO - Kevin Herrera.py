import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv('deliverytime.csv')

# Preprocesamiento de datos
df['Type_of_order'] = df['Type_of_order'].str.strip()
df['Type_of_vehicle'] = df['Type_of_vehicle'].str.strip()

# Convertir las columnas 'Type_of_order' y 'Type_of_vehicle' a códigos
df['Type_of_order'] = df['Type_of_order'].astype('category').cat.codes
df['Type_of_vehicle'] = df['Type_of_vehicle'].astype('category').cat.codes

X = df[['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude', 
        'Restaurant_longitude', 'Delivery_location_latitude', 
        'Delivery_location_longitude', 'Type_of_order', 'Type_of_vehicle']]
y = df['Time_taken(min)']

# Normalización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construcción del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# Evaluación del modelo
model.evaluate(X_test, y_test)

#-------------------------------------------------------------------------------------

# Obtener la pérdida en el entrenamiento y la validación
loss = history.history['loss']
val_loss = history.history['val_loss']

# Crear el gráfico
plt.figure()
plt.plot(loss, label='Pérdida en entrenamiento')
plt.plot(val_loss, label='Pérdida en validación')
plt.title('Pérdida en el entrenamiento y la validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test).flatten()

# Crear el gráfico de predicción
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel('Tiempo real')
plt.ylabel('Tiempo predicho')
plt.title('Predicción vs. Realidad')
plt.show()

#-------------------------------------------------------------------------------------

# Crear un nuevo conjunto de datos de ejemplo
new_data = {
    'Delivery_person_Age': [30, 25, 35],
    'Delivery_person_Ratings': [4.5, 3.8, 4.9],
    'Restaurant_latitude': [22.75, 12.91, 11.00],
    'Restaurant_longitude': [75.89, 77.67, 76.97],
    'Delivery_location_latitude': [22.76, 12.92, 11.05],
    'Delivery_location_longitude': [75.91, 77.68, 77.02],
    'Type_of_order': ['Snack', 'Drinks', 'Buffet'],
    'Type_of_vehicle': ['motorcycle', 'motorcycle', 'motorcycle']
}

# Convertir el nuevo conjunto de datos a un DataFrame
new_df = pd.DataFrame(new_data)

# Preprocesar los datos de la misma manera que se hizo con los datos de entrenamiento
new_df['Type_of_order'] = new_df['Type_of_order'].str.strip()
new_df['Type_of_vehicle'] = new_df['Type_of_vehicle'].str.strip()
new_df['Type_of_order'] = new_df['Type_of_order'].astype('category').cat.codes
new_df['Type_of_vehicle'] = new_df['Type_of_vehicle'].astype('category').cat.codes


# Realizar la predicción
predictions = model.predict(X_new)

# Mostrar los resultados
for i, prediction in enumerate(predictions):
    example_data = new_df.iloc[i]
    print(f"Ejemplo {i + 1}:")
    print(f"  - Edad del repartidor: {example_data['Delivery_person_Age']}")
    print(f"  - Calificación del repartidor: {example_data['Delivery_person_Ratings']}")
    print(f"  - Latitud del restaurante: {example_data['Restaurant_latitude']}")
    print(f"  - Longitud del restaurante: {example_data['Restaurant_longitude']}")
    print(f"  - Latitud de entrega: {example_data['Delivery_location_latitude']}")
    print(f"  - Longitud de entrega: {example_data['Delivery_location_longitude']}")
    print(f"  - Tipo de pedido: {example_data['Type_of_order']}")
    print(f"  - Tipo de vehículo: {example_data['Type_of_vehicle']}")
    print(f"El modelo predice un tiempo de entrega de aproximadamente {prediction[0]:.2f} minutos.\n")