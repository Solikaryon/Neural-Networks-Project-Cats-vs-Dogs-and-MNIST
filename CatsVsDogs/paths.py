from pathlib import Path

# Raíz del proyecto (carpeta ProyectoRedesNeuronales)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Carpetas de datos (existentes en la estructura actual)
RAW_DATA_DIR = PROJECT_ROOT / "PetImages"               # Datos originales (imagenes)
PROCESSED_DATA_DIR = PROJECT_ROOT / "PetImages_preprocesado"  # Datos preprocesados (.npy)

# Subcarpetas dentro de datos procesados
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Carpetas para artefactos y modelo entrenado
METRICS_DIR = PROJECT_ROOT / "Metricas"
MODEL_DIR = PROJECT_ROOT / "ModeloEntrenado"

# Crear si no existen (no falla si ya existen)
METRICS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "TRAIN_DIR",
    "VAL_DIR",
    "TEST_DIR",
    "METRICS_DIR",
    "MODEL_DIR",
]
