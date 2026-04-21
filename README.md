# Machine Learning — Predicción de Mortalidad Cardíaca y Propagación del Dengue

> **Autor:** Pablo Sáez Gil

---

## Proyectos incluidos

### 1. Predicción de mortalidad por insuficiencia cardíaca (`heart_failure_death_prediction.ipynb`)

Modelo de clasificación que predice el fallecimiento de pacientes con insuficiencia cardíaca a partir de registros clínicos reales (299 pacientes, 12 variables).

- **Técnicas:** Random Forest con GridSearchCV, transformaciones logarítmicas, análisis de outliers
- **Dataset:** Heart Failure Clinical Records (UCI)
- **Objetivo:** Predecir `DEATH_EVENT` (0/1) — AUC-ROC como métrica principal

### 2. Clustering de patrones climáticos del dengue (`dengai_unsupervised_clustering.ipynb`)

Aprendizaje no supervisado sobre datos epidemiológicos semanales de San Juan (Puerto Rico) e Iquitos (Perú) para identificar semanas de alto riesgo de contagio.

- **Técnicas:** PCA, K-Means, Clustering Jerárquico, DBSCAN, GMM, imputación KNN
- **Dataset:** [DengAI — DrivenData](https://www.drivendata.org/competitions/44/) (1.456 registros)

### 3. Predicción supervisada de casos de dengue (`dengai_supervised_prediction.ipynb`)

Modelos de regresión para estimar el número semanal de casos de dengue a partir de variables climáticas y medioambientales.

- **Técnicas:** Random Forest, XGBoost, LightGBM, validación temporal, ingeniería de features (lags, rolling means)
- **Dataset:** DengAI — DrivenData

## Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `matplotlib` · `seaborn`

## Estructura

```
ml-dengai-prediction/
├── notebooks/
│   ├── heart_failure_death_prediction.ipynb
│   ├── dengai_unsupervised_clustering.ipynb
│   └── dengai_supervised_prediction.ipynb
├── data/
│   ├── DengAI_..._Training_Data_Features.csv
│   ├── DengAI_..._Training_Data_Labels.csv
│   └── DengAI_..._Test_Data_Features.csv
└── README.md
```

## Cómo reproducir

```bash
git clone https://github.com/pablosaezmath/ml-dengai-prediction.git
cd ml-dengai-prediction
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter
jupyter notebook notebooks/
```

## Autor

**Pablo Sáez Gil** — Graduado en Matemáticas · Máster en Big Data y Ciencia de Datos (VIU) · saeznovelda@gmail.com
