from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Carpeta para datos procesados MNIST
MNIST_PROCESSED_DIR = PROJECT_ROOT / "MNIST_preprocesado"
TRAIN_DIR = MNIST_PROCESSED_DIR / "train"
VAL_DIR = MNIST_PROCESSED_DIR / "val"
TEST_DIR = MNIST_PROCESSED_DIR / "test"
# Artefactos y modelo específicos MNIST
METRICS_MNIST_DIR = PROJECT_ROOT / "MetricasMNIST"
MODEL_MNIST_DIR = PROJECT_ROOT / "ModeloMNIST"
for d in [MNIST_PROCESSED_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, METRICS_MNIST_DIR, MODEL_MNIST_DIR]:
    d.mkdir(exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "MNIST_PROCESSED_DIR",
    "TRAIN_DIR",
    "VAL_DIR",
    "TEST_DIR",
    "METRICS_MNIST_DIR",
    "MODEL_MNIST_DIR",
]
