NASDAQ Open Price Direction Predictor (AAPL)
Predicción de dirección del precio de apertura usando Machine Learning y datos del NASDAQ
Descripción

Este proyecto open source implementa un modelo de aprendizaje estadístico orientado a predecir la dirección del precio de apertura del día siguiente para activos del NASDAQ, con un caso práctico aplicado a Apple Inc. (AAPL).

El sistema utiliza datos históricos OHLCV descargados mediante yfinance y emplea técnicas avanzadas de Machine Learning (Logistic Regression, Gradient Boosting, XGBoost) junto con validación temporal tipo walk-forward.

El objetivo principal es generar señales direccionales:
BUY
SELL
NEUTRAL

Características principales:

Descarga automática de datos financieros con yfinance.
Construcción de variables técnicas de momentum, volatilidad, tendencia y volumen relativo.
Preprocesamiento completo: limpieza, normalización y alineamiento temporal entre series.
Modelos de clasificación binaria para predicción direccional.
Validación temporal mediante esquema walk-forward.
Generación de señales BUY, SELL y NEUTRAL.
Evaluación fuera de muestra con métricas estadísticas.
Proyecto completamente open source bajo licencia MIT.

Activos utilizados

Los siguientes archivos fueron obtenidos mediante yfinance y usados para construir el dataset:

| Archivo  | Descripción                                  |
| -------- | -------------------------------------------- |
| aapl.csv | Precios diarios de Apple Inc. (NASDAQ: AAPL) |
| qqq.csv  | ETF del NASDAQ-100                           |
| spy.csv  | ETF del S&P500                               |
| vix.csv  | Índice de volatilidad VIX                    |
| tnx.csv  | Rendimiento del bono del Tesoro a 10 años    |


Tecnologías usadas
El proyecto utiliza las siguientes librerías y frameworks de Python:

numpy
pandas
tqdm
matplotlib
scikit-learn
matthews_corrcoef
CalibratedClassifierCV
XGBoost (XGBClassifier)
LightGBM (LGBMClassifier)
CatBoost (CatBoostClassifier)

Modelos utilizados

El sistema entrena un conjunto de modelos de clasificación supervisada para predecir la dirección del precio de apertura:
CatBoostClassifier
XGBClassifier
LGBMClassifier
Modelos calibrados mediante CalibratedClassifierCV
Comparación mediante la métrica MCC (Matthews Correlation Coefficient)
Los modelos se calibran para producir probabilidades estables y se evalúan bajo un esquema temporal robusto.
