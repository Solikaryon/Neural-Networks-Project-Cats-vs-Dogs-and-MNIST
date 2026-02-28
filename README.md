# Neural Networks Project: Cats vs Dogs and MNIST

This repository includes two image classification experiments built with CNNs:
- **Cats vs Dogs** (binary classification)
- **MNIST Digits** (10-class classification: 0 to 9)

The project provides a graphical user interface (`interface.py`) to run preprocessing, training (Baseline/Alternative and MinPooling variants), model loading, and image prediction.

## Features

- End-to-end workflows for both datasets from preprocessing to inference.
- Multiple training configurations (baseline, alternative hyperparameters, and MinPooling).
- Automatic artifact generation (plots, confusion matrices, JSON summaries, trained models).
- GUI-based experimentation with optional CLI training scripts.

## Requirements

- Python **3.9+**
- Recommended packages:
  - `torch`
  - `numpy`
  - `Pillow`
  - `matplotlib`
  - `tqdm`
  - `scikit-learn`
- Optional package:
  - `tensorflow` (used for MNIST download in one path; if unavailable, a fallback downloads `mnist.npz` directly)

Quick installation:

```powershell
pip install torch numpy Pillow matplotlib tqdm scikit-learn
# optional
pip install tensorflow
```

## Project Structure

- `interface.py`: Main GUI entry point for both experiments.
- `CatsVsDogs/`: Cats vs Dogs modules (dataset, preprocessing, CNN, training, local interface helpers).
- `MNIST-Digits/`: MNIST modules (dataset, preprocessing, CNN, training).
- `PetImages/`: Raw Cats/Dogs source images.
- `PetImages_preprocesado/`: Preprocessed Cats/Dogs train/val/test arrays.
- `MNIST_preprocesado/`: Preprocessed MNIST arrays and summary file.
- `ModeloEntrenado/`: Trained Cats vs Dogs model files (`.pth`).
- `ModeloMNIST/`: Trained MNIST model files (`.pth`).
- `Metricas/`: Cats vs Dogs training metrics and generated plots.
- `MetricasMNIST/`: MNIST training metrics and generated plots.
- `ImagenesPruebas/`: Images for manual prediction testing.

## Run the GUI

From the project root folder:

```powershell
python interface.py
```

Inside the GUI, choose the dataset workflow you want:
- **Cats vs Dogs:** preprocess, train (Baseline + Alternative / MinPooling), load model, predict one or many images.
- **MNIST:** download + preprocess, train (Baseline + Alternative / MinPooling), load model, view sample predictions, predict external images.

## Recommended Workflow

### Cats vs Dogs

1. **Preprocess Data**
   - Converts source images to `.npy` arrays.
   - Creates train/validation/test splits under `PetImages_preprocesado/`.

2. **Train (Baseline + Alternative)**
   - Runs two experiments with different hyperparameters.
   - Saves plots and metrics in `Metricas/` (PNG and SVG).
   - Stores best model as `ModeloEntrenado/best_pet_cnn_model.pth`.

3. **Train MinPooling**
   - Runs a model variant using MinPooling.
   - Saves model as `ModeloEntrenado/MinPooling_CatVsDog.pth`.
   - Generates additional metrics (including MinPooling accuracy curves).

4. **Load Model**
   - Select any `.pth` model available in the model folder.

5. **Predict Images**
   - Supports single and batch image prediction.
   - For multi-image runs, temporary aggregate accuracy graphics are also generated.

### MNIST

1. **Download/Preprocess Data**
   - Downloads dataset (via TensorFlow path or fallback `.npz` download).
   - Creates split `.npy` files under `MNIST_preprocesado/`.

2. **Train (Baseline + Alternative)**
   - Runs multiple configurations and saves:
     - learning curves,
     - confusion matrices,
     - accuracy charts,
     - summary JSON files.
   - Saves a best model file (e.g., `mnist_best_model.pth`).

3. **Train MinPooling**
   - Trains and stores MinPooling model as `ModeloMNIST/MinPooling_MNIST.pth`.

4. **Load Model**
   - Select any MNIST `.pth` model from `ModeloMNIST/`.

5. **Inference Tools**
   - View sample predictions from test split.
   - Predict custom images (single or multi-image mode).

## Optional CLI Training

You can run training scripts directly from terminal:

```powershell
# Cats vs Dogs
python -m CatsVsDogs.train

# MNIST
python .\MNIST-Digits\trainMNIST.py
```

## Artifacts and Outputs

After running experiments, the project stores:
- Trained model checkpoints (`.pth`).
- Accuracy and loss curves.
- Confusion matrix visualizations.
- Experiment summaries in JSON (hyperparameters, metrics, output paths).
- PNG/SVG figures for reports and presentations.

## Performance Notes

- If CUDA is available, training uses GPU automatically.
- In CPU mode, MinPooling workflows may apply faster loading settings (such as tuned workers/prefetching and larger batch size) to reduce training time.

## Common Issues

- **MNIST preprocessing module not found**
  - Verify `MNIST-Digits/pre_processingMNIST.py` exists.
  - If TensorFlow is not installed, fallback download logic should still fetch `mnist.npz`.

- **SVG previews in Tkinter**
  - Tkinter does not render SVG natively.
  - The GUI usually displays PNG while SVG files are still exported to disk.

## Notes

- Pretrained model files are already included in `ModeloEntrenado/` and `ModeloMNIST/`.
- If you retrain models, existing checkpoints and metric files may be overwritten depending on script settings.

## Author

- Luis Fernando Monjaraz Briseño
