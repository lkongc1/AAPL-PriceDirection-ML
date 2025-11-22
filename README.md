# NASDAQ Open Price Direction Predictor (AAPL)

Modelo de aprendizaje estadístico para predecir la dirección del precio de apertura de acciones del NASDAQ, con un caso práctico aplicado a Apple (AAPL). El proyecto utiliza datos OHLCV extraídos desde yfinance e implementa técnicas avanzadas de Machine Learning para generar señales BUY, SELL y NEUTRAL. El código es completamente open source bajo licencia MIT.

## Descripción

Este proyecto implementa un sistema de predicción direccional orientado a estimar si el precio de apertura del día siguiente será mayor o menor que el cierre del día previo.

El sistema se basa en:

- Datos históricos OHLCV descargados con yfinance.
- Preprocesamiento estructurado y normalización por ventanas temporales.
- Entrenamiento de modelos ML con validación temporal walk-forward.
- Calibración probabilística de los modelos para obtener señales estables.
- Generación final de señales operativas BUY, SELL y NEUTRAL.

El modelo principal se centra en Apple Inc. (AAPL), uno de los activos más representativos del NASDAQ.

## Características principales

- Descarga automática de datos financieros con yfinance.
- Construcción de variables técnicas de momentum, volatilidad, tendencia y volumen relativo.
- Preprocesamiento completo: limpieza, normalización y alineamiento entre series.
- Entrenamiento de modelos de clasificación binaria.
- Validación temporal mediante esquema walk-forward.
- Generación de señales BUY, SELL y NEUTRAL.
- Evaluación fuera de muestra con métricas estadísticas.
- Proyecto completamente open source bajo licencia MIT.

## Activos utilizados

Los siguientes archivos fueron generados mediante yfinance y utilizados para la construcción del dataset:

| Archivo  | Descripción |
|----------|-------------|
| aapl.csv | Precios diarios de Apple Inc. (NASDAQ: AAPL) |
| qqq.csv  | ETF del NASDAQ-100 |
| spy.csv  | ETF del S&P500 |
| vix.csv  | Índice de volatilidad VIX |
| tnx.csv  | Rendimiento del bono del Tesoro a 10 años |

## Tecnologías usadas

- numpy  
- pandas  
- tqdm  
- matplotlib  
- scikit-learn  
  - matthews_corrcoef  
  - CalibratedClassifierCV  
- XGBoost (XGBClassifier)  
- LightGBM (LGBMClassifier)  
- CatBoost (CatBoostClassifier)

## Modelos utilizados

El sistema entrena un conjunto de modelos supervisados:

- CatBoostClassifier
- XGBClassifier
- LGBMClassifier
- Modelos calibrados mediante CalibratedClassifierCV
- Comparación mediante la métrica MCC (Matthews Correlation Coefficient)

Los modelos son calibrados y evaluados bajo un esquema walk-forward para evitar cualquier fuga temporal y asegurar una validación realista.

## Licencia

Este proyecto es open source y se distribuye bajo la licencia MIT.  
Consultar el archivo LICENSE para más información.

## Versión de prueba para Android (app-debug.apk)

Se incluye el archivo `app-debug.apk` con fines exclusivamente de prueba.  
Este APK fue generado desde Android Studio utilizando el build variant "debug" y sirve únicamente para validar la instalación y funcionamiento básico de la aplicación en dispositivos Android.

La aplicación no cuenta con funcionalidades adicionales y no realiza procesos reales.  
Se proporciona un usuario de prueba para poder acceder a la interfaz:

- **Email:** test@test.com  
- **Contraseña:** 12345678  

Este acceso es únicamente demostrativo y no representa datos reales ni información sensible.
