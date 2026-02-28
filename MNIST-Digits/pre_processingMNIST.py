import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import json
import urllib.request
import os
from paths_mnist import TRAIN_DIR, VAL_DIR, TEST_DIR, MNIST_PROCESSED_DIR

# Intentar importar TensorFlow/Keras; si no está disponible usamos descarga directa del archivo mnist.npz
try:
    from tensorflow.keras.datasets import mnist  # type: ignore
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

IMG_SIZE = (28, 28)  # MNIST original
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1  # Usamos el test original de MNIST, pero podemos reservar parte extra si se desea

class MNISTPreprocessor:
    def __init__(self, val_split=VAL_SPLIT):
        self.val_split = val_split

    def load_raw(self):
        if TF_AVAILABLE:
            (x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
            return (x_train, y_train), (x_test, y_test)
        # Fallback: descargar mnist.npz manualmente si no existe aún
        local_path = MNIST_PROCESSED_DIR / "mnist.npz"
        if not local_path.exists():
            url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            print("Descargando mnist.npz (fallback sin TensorFlow)...")
            try:
                urllib.request.urlretrieve(url, str(local_path))
            except Exception as e:
                raise RuntimeError(f"No se pudo descargar MNIST automáticamente: {e}. Instale tensorflow o coloque mnist.npz manualmente en {MNIST_PROCESSED_DIR}.")
        with np.load(local_path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

    def normalize(self, arr):
        return (arr.astype("float32") / 255.0)

    def expand_channel(self, arr):
        # De (H,W) -> (H,W,1)
        return np.expand_dims(arr, axis=-1)

    def save_npy_group(self, images, labels, target_dir: Path):
        for idx, (img, label) in enumerate(tqdm(list(zip(images, labels)), desc=f"Guardando en {target_dir.name}")):
            nombre = f"{label}_{idx}.npy"
            path_out = target_dir / nombre
            np.save(path_out, img, allow_pickle=False)

    def preprocess(self):
        print("\nCargando MNIST...")
        if not TF_AVAILABLE:
            print("TensorFlow no disponible: usando loader de archivo .npz directo.")
        (x_train, y_train), (x_test, y_test) = self.load_raw()
        print(f"Train bruto: {x_train.shape}, Test bruto: {x_test.shape}")

        # Normalizar y expandir canal
        x_train = self.expand_channel(self.normalize(x_train))
        x_test = self.expand_channel(self.normalize(x_test))

        # Crear división train/val a partir de x_train
        total_train = x_train.shape[0]
        n_val = int(total_train * self.val_split)
        indices = list(range(total_train))
        random.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        x_train_final = x_train[train_idx]
        y_train_final = y_train[train_idx]

        print(f"Train final: {x_train_final.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

        # Guardar
        self.save_npy_group(x_train_final, y_train_final, TRAIN_DIR)
        self.save_npy_group(x_val, y_val, VAL_DIR)
        self.save_npy_group(x_test, y_test, TEST_DIR)

        summary = {
            'train_samples': int(x_train_final.shape[0]),
            'val_samples': int(x_val.shape[0]),
            'test_samples': int(x_test.shape[0]),
            'image_shape': IMG_SIZE + (1,),
        }
        with open(MNIST_PROCESSED_DIR / 'mnist_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("Resumen guardado en mnist_summary.json")
        print("Preprocesamiento MNIST completado.")
        if not TF_AVAILABLE:
            print("Nota: Se usó método alternativo sin TensorFlow.")

if __name__ == "__main__":
    prep = MNISTPreprocessor()
    prep.preprocess()
