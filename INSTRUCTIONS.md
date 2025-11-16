# Instrucciones para el Homework - Predicción de Default con MLP

## Resumen de la Implementación

Se ha implementado un pipeline completo de Machine Learning para predecir el default de crédito usando una red neuronal MLP. El código incluye todos los 7 pasos requeridos.

## Estructura del Código

El archivo `homework/homework.py` contiene:

### 1. Funciones Implementadas

- **`load_and_clean_data()`**: Carga los datos y realiza la limpieza
  - Carga archivos CSV comprimidos desde `files/input/`
  - Renombra columna "default payment next month" a "default"
  - Elimina columna "ID"
  - Elimina registros con valores faltantes
  - Agrupa valores de EDUCATION > 4 en categoría 4

- **`split_data(train_data, test_data)`**: Separa features (X) de target (y)

- **`create_pipeline(x_train)`**: Crea el pipeline de ML con:
  - OneHotEncoder para variables categóricas (SEX, EDUCATION, MARRIAGE)
  - PCA para reducción de dimensionalidad
  - StandardScaler para normalización
  - SelectKBest para selección de features
  - MLPClassifier para la red neuronal

- **`optimize_hyperparameters(pipeline, x_train, y_train, quick_mode)`**: Configura GridSearchCV
  - 10-fold cross-validation (5 en quick_mode)
  - balanced_accuracy como métrica
  - Búsqueda de hiperparámetros configurable

- **`save_model(model, filename)`**: Guarda el modelo comprimido con gzip

- **`calculate_metrics(model, x_train, y_train, x_test, y_test)`**: Calcula métricas
  - precision, balanced_accuracy, recall, f1_score

- **`calculate_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred)`**: Calcula matrices de confusión

- **`save_metrics(metrics, cm_matrices, filename)`**: Guarda métricas en JSON

- **`main(run_training, quick_mode)`**: Función principal que ejecuta todo

## Cómo Usar

### Opción 1: Entrenamiento Rápido (Recomendado)
```bash
cd /ruta/al/repositorio
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
python homework/homework.py
```

Por defecto usa `quick_mode=True` que entrena más rápido con buenos resultados.

### Opción 2: Entrenamiento Completo (Más tiempo)

Editar el final de `homework/homework.py`:
```python
if __name__ == "__main__":
    main(run_training=True, quick_mode=False)
```

Luego ejecutar:
```bash
python homework/homework.py
```

### Opción 3: Solo Verificar el Código (Sin entrenar)

Editar el final de `homework/homework.py`:
```python
if __name__ == "__main__":
    main(run_training=False)
```

## Parámetros de Búsqueda

### Quick Mode (por defecto)
- pca__n_components: [20, 25]
- selector__k: [20, 25]
- classifier__hidden_layer_sizes: [(100,), (100, 50), (100, 100)]
- classifier__activation: ["relu"]
- classifier__alpha: [0.0001, 0.001]
- cv: 5 folds

### Full Mode
- pca__n_components: [15, 20, 25, 30]
- selector__k: [15, 20, 25, 30]
- classifier__hidden_layer_sizes: [(50,), (100,), (100, 50), (100, 100), (150,)]
- classifier__activation: ["relu", "tanh"]
- classifier__alpha: [0.00001, 0.0001, 0.001, 0.01]
- cv: 10 folds

## Archivos Generados

Después del entrenamiento se crean:

1. **`files/models/model.pkl.gz`**: Modelo entrenado comprimido
2. **`files/output/metrics.json`**: Métricas y matrices de confusión

Estos archivos están en `.gitignore` y deben generarse localmente.

## Ejecutar Tests

```bash
pytest -v
```

Los tests verifican:
- Que el modelo existe y está comprimido correctamente
- Que contiene todos los componentes requeridos
- Que el score es mayor a 0.661 en train y 0.666 en test
- Que las métricas son correctas

## Tiempo Estimado de Entrenamiento

- **Quick mode**: ~5-10 minutos (24 combinaciones × 10 folds = 240 entrenamientos)
- **Full mode**: ~30-60 minutos (240+ combinaciones × 10 folds = 2400+ entrenamientos)

El tiempo depende de tu CPU y número de cores disponibles (usa n_jobs=-1 para usar todos).

## Troubleshooting

### Error: "model.pkl.gz not found"
- Necesitas ejecutar el entrenamiento primero

### Scores muy bajos
- Aumenta las iteraciones del MLP editando `max_iter` en create_pipeline()
- Prueba con diferentes hiperparámetros
- Verifica que los datos se cargaron correctamente

### Entrenamiento muy lento
- Usa quick_mode=True
- Reduce el número de combinaciones en param_grid
- Reduce cv de 10 a 5 folds

## Notas Importantes

1. El código está completo y funcional
2. Los warnings sobre "k > n_features" son normales cuando PCA reduce dimensiones
3. El random_state=42 asegura reproducibilidad
4. El modelo usa StandardScaler (no MinMaxScaler) para mejor rendimiento con MLP
5. El entrenamiento es computacionalmente intensivo - se recomienda ejecutar en una máquina con buen CPU

## Siguiente Paso

Ejecuta el entrenamiento en tu máquina local para generar los archivos necesarios y pasar los tests.
