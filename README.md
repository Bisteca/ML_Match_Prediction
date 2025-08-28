⚽ Predicción de Resultados de Partidos de Fútbol con Machine Learning
📌 Descripción del Proyecto

Este proyecto tiene como objetivo predecir los resultados de partidos de fútbol (victoria local o visitante) utilizando modelos de Machine Learning.

La motivación principal es demostrar cómo la analítica de datos y los algoritmos de clasificación pueden aplicarse en el mundo del deporte, generando insights útiles y, potencialmente, apoyando decisiones en contextos como casas de apuestas, scouting y análisis deportivo.

El proyecto incluye:

Preprocesamiento de datos históricos de partidos.

Entrenamiento y evaluación de modelos de clasificación (XGBoost y LightGBM).

Aplicación interactiva en Streamlit para probar el modelo de forma sencilla.

📂 Estructura del Repositorio
ML_project/
│
├── app_streamlit/         # Aplicación web con Streamlit
│   └── app.py
│
├── data/                  # Datos (raw, processed, train, test)
│   ├── raw/
│   │   └── Matches.csv
│   ├── processed/
│   │   └── matches_processed.csv
│   ├── train/
│   │   ├── X_train.csv
│   │   └── y_train.csv
│   └── test/
│       ├── X_test.csv
│       └── y_test.csv
│
├── models/                # Modelos entrenados (.pkl)
│
├── src/                   # Scripts principales
│   ├── data_processing.py # Procesamiento de datos
│   ├── training.py        # Entrenamiento de modelos
│   └── evaluate.py        # Evaluación de modelos
│
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación

📊 Dataset

Fuente: Club Football Match Data 2000-2025

Raw dataset: datos completos de partidos con información de equipos, ligas y resultados originales.

X procesado:

Cantidad: 43.708 partidos

Variables: 108 columnas, incluyendo estadísticas de equipos, resultados y datos de rendimiento.

🎯 Objetivos del Proyecto

Construir un modelo de ML capaz de predecir los resultados de partidos de fútbol.

Explorar diferentes algoritmos y técnicas de procesamiento para mejorar la precisión.

Desarrollar una aplicación interactiva que permita a cualquier usuario simular predicciones fácilmente.

🛠️ Metodología

Preprocesamiento

Limpieza de datos nulos.

Transformación de variables categóricas.

Normalización de variables numéricas.

Eliminación de la clase empate tras pruebas iniciales.

Modelos probados

🔹 Logistic Regression

🔹 Random Forest

🔹 K-Nearest Neighbors (KNN)

🔹 XGBoost

🔹 LightGBM

Métricas utilizadas

Accuracy

📈 Resultados

Predicción con los 3 resultados (victoria local, empate, victoria visitante):

Accuracy ≈ 50%

Predicción solo entre victoria local o visitante (sin empate):

Accuracy ≈ 70%

📌 Esto demuestra que el modelo mejora significativamente cuando se reduce el problema a dos clases, aunque se pierde realismo en los escenarios con empate.

🖥️ Aplicación en Streamlit

Se desarrolló una app interactiva en Streamlit que permite:

Seleccionar equipos (local y visitante).

Definir la fecha del partido.

Predecir el resultado usando el modelo entrenado.

Ejecutar la aplicación:

streamlit run app_streamlit/app.py

⚠️ Limitaciones

El modelo se entrenó solo con datos históricos: no considera factores como lesiones, clima o cambios recientes de entrenador.

Se eliminó la categoría de empate, lo cual aumentó la precisión pero reduce el realismo del modelo.

🚀 Próximos Pasos

Reincorporar el empate usando técnicas de balanceo de clases.

Integrar datos en tiempo real (lesiones, alineaciones, cuotas de apuestas).

Explorar otros modelos y técnicas de tuning más avanzadas.

Mejorar la visualización de variables importantes y explicar las predicciones con técnicas de interpretabilidad (ej. SHAP).

📦 Requisitos

Instalar dependencias:

pip install -r requirements.txt

📌 Conclusión

Este proyecto demuestra cómo Machine Learning puede aplicarse en el fútbol para generar predicciones y análisis útiles. Aunque tiene limitaciones (principalmente en la predicción de empates y factores contextuales), representa una base sólida para aplicaciones más avanzadas en el futuro.
