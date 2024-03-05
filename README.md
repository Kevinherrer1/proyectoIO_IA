# Predicción de Tiempo de Entrega en Servicio de Delivery

Este repositorio contiene un script en Python que utiliza un modelo de aprendizaje automático para predecir el tiempo de entrega en un servicio de delivery. El modelo se entrena utilizando datos históricos de entregas y se evalúa su rendimiento.

## Requisitos

- Python 3.x
- Bibliotecas de Python:
  - pandas
  - numpy
  - tensorflow
  - scikit-learn
  - matplotlib

## Descripción del Código

El script consta de las siguientes secciones principales:

1. **Carga y Preprocesamiento de Datos**: Se cargan los datos históricos de entregas desde un archivo CSV y se realizan algunas transformaciones como la conversión de variables categóricas a códigos numéricos y la normalización de los datos.

2. **División del Conjunto de Datos**: Se divide el conjunto de datos en conjuntos de entrenamiento y prueba para su posterior uso en el entrenamiento y evaluación del modelo.

3. **Construcción y Entrenamiento del Modelo**: Se define un modelo de red neuronal utilizando TensorFlow y se entrena utilizando el conjunto de entrenamiento.

4. **Evaluación del Modelo**: Se evalúa el rendimiento del modelo utilizando el conjunto de prueba y se muestra la pérdida obtenida durante el entrenamiento y la validación.

5. **Predicción y Visualización**: Se realizan predicciones utilizando el modelo entrenado y se visualiza la comparación entre los tiempos de entrega reales y los tiempos predichos.

6. **Predicción con Nuevos Datos**: Se generan predicciones de tiempo de entrega utilizando un conjunto de datos de ejemplo proporcionado en el código.





## Autor

Kevin Herrera 


