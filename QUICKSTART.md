# Quick Start Guide

## Ejecución Rápida

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar entrenamiento
python homework/homework.py
```

Eso es todo! El script:
- ✅ Carga y limpia los datos
- ✅ Crea el pipeline con todos los componentes
- ✅ Entrena el modelo con GridSearchCV
- ✅ Guarda el modelo en `files/models/model.pkl.gz`
- ✅ Guarda las métricas en `files/output/metrics.json`

## Verificar Resultados

```bash
# Ejecutar tests
pytest -v

# Ver métricas generadas
cat files/output/metrics.json

# Verificar que el modelo existe
ls -lh files/models/model.pkl.gz
```

## Componentes del Pipeline (en orden)

1. **OneHotEncoder** - Codifica variables categóricas (SEX, EDUCATION, MARRIAGE)
2. **PCA** - Reduce dimensionalidad
3. **StandardScaler** - Normaliza features
4. **SelectKBest** - Selecciona mejores features
5. **MLPClassifier** - Red neuronal para clasificación

## Grid Search Parameters

```python
{
    "pca__n_components": [20, 25],
    "selector__k": [20, 25],
    "classifier__hidden_layer_sizes": [(100,), (100, 50), (100, 100)],
    "classifier__activation": ["relu"],
    "classifier__alpha": [0.0001, 0.001],
}
```

Total: 24 combinaciones × 10 folds = **240 entrenamientos**

## Salida Esperada

```
Step 1: Loading and cleaning data...
Train data shape: (21000, 24)
Test data shape: (9000, 24)

Step 2: Splitting data...
x_train shape: (21000, 23)

Step 3: Creating pipeline...
Pipeline created successfully

Step 4: Optimizing hyperparameters...
Step 4: Training model (this may take a while)...
Fitting 10 folds for each of 24 candidates, totalling 240 fits
[...progreso del entrenamiento...]

Step 5: Saving model...
Model saved to files/models/model.pkl.gz

Step 6 & 7: Calculating and saving metrics...
Metrics saved to files/output/metrics.json

Done!
```

## Archivos Generados

### `files/models/model.pkl.gz`
Modelo GridSearchCV entrenado y comprimido (~65KB)

### `files/output/metrics.json`
```json
{"type": "metrics", "dataset": "train", "precision": 0.77, "balanced_accuracy": 0.74, "recall": 0.53, "f1_score": 0.63}
{"type": "metrics", "dataset": "test", "precision": 0.52, "balanced_accuracy": 0.65, "recall": 0.39, "f1_score": 0.44}
{"type": "cm_matrix", "dataset": "train", "true_0": {"predicted_0": 15532, "predicted_1": 741}, "true_1": {"predicted_0": 2222, "predicted_1": 2505}}
{"type": "cm_matrix", "dataset": "test", "true_0": {"predicted_0": 6417, "predicted_1": 674}, "true_1": {"predicted_0": 1170, "predicted_1": 739}}
```

## Personalización

Para ajustar hiperparámetros, edita la función `optimize_hyperparameters()` en `homework/homework.py`:

```python
param_grid = {
    "pca__n_components": [15, 20, 25, 30],  # Agregar más valores
    "selector__k": [15, 20, 25, 30],
    "classifier__hidden_layer_sizes": [(50,), (100,), (150,)],
    "classifier__activation": ["relu", "tanh"],
    "classifier__alpha": [0.0001, 0.001, 0.01],
}
```

## Tips para Mejorar Performance

1. **Aumentar iteraciones del MLP**: Cambiar `max_iter` en `create_pipeline()`
2. **Más hiperparámetros**: Agregar valores al `param_grid`
3. **Early stopping**: Agregar `early_stopping=True` al MLPClassifier
4. **Learning rate adaptativo**: Usar `learning_rate='adaptive'`

## Troubleshooting

**"ModuleNotFoundError"** → `pip install -r requirements.txt`

**"File not found"** → Verificar que estás en el directorio correcto

**"Timeout"** → Reduce combinaciones de hiperparámetros o usa `quick_mode=True`

**Scores bajos** → Aumenta `max_iter`, prueba más hiperparámetros, verifica datos

## Tiempo de Ejecución

- **Quick mode (por defecto)**: 5-10 minutos
- **Full mode**: 30-60 minutos

Depende de tu CPU y número de cores (el script usa `n_jobs=-1` para paralelizar).
