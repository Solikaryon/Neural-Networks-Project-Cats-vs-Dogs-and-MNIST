import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
from tqdm import tqdm

from paths_mnist import TRAIN_DIR, VAL_DIR, TEST_DIR, METRICS_MNIST_DIR, MODEL_MNIST_DIR
from datasetMNIST import MNISTNpyDataset
from convolutional_nnMNIST import CNNMNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001  # baseline
BATCH_SIZE = 64
EPOCHS = 10  # baseline epochs aumentado a 10
ALT_LEARNING_RATE = 0.0005
ALT_EPOCHS = 8  # mantener alt por ahora; se puede ajustar si se desea


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Train"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total if total else 0.0
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, phase="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=phase):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total if total else 0.0
    return epoch_loss, epoch_acc


def evaluate_test(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Test"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().tolist())
    report = classification_report(all_labels, all_preds, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    print("\n=== Test Report ===")
    print(report)
    print("Confusion Matrix:\n", cm)
    return report, cm


def plot_curves(history, suffix: str):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'MNIST Loss ({suffix})'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f'MNIST Accuracy ({suffix})'); plt.legend()
    plt.tight_layout()
    curves_png = METRICS_MNIST_DIR / f'mnist_learning_curves_{suffix}.png'
    curves_svg = METRICS_MNIST_DIR / f'mnist_learning_curves_{suffix}.svg'
    plt.savefig(curves_png, dpi=120)
    plt.savefig(curves_svg)
    plt.close()
    print(f"Curvas guardadas en {curves_png} y {curves_svg}")
    return curves_png, curves_svg


def save_confusion_matrix(cm, suffix: str):
    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'MNIST Confusion Matrix ({suffix})')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xticks(range(10)); plt.yticks(range(10))
    plt.tight_layout()
    cm_png = METRICS_MNIST_DIR / f'mnist_confusion_matrix_{suffix}.png'
    cm_svg = METRICS_MNIST_DIR / f'mnist_confusion_matrix_{suffix}.svg'
    plt.savefig(cm_png, dpi=120)
    plt.savefig(cm_svg)
    plt.close()
    return cm_png, cm_svg


def run_mnist_experiment(lr: float, epochs: int, suffix: str, pool_type: str = 'max', optimize_cpu: bool = False):
    """Ejecuta un experimento MNIST y guarda artefactos con sufijo."""
    print(f"\n=== Experimento {suffix} | lr={lr} epochs={epochs} ===")
    import os
    train_ds = MNISTNpyDataset(TRAIN_DIR)
    val_ds = MNISTNpyDataset(VAL_DIR)
    test_ds = MNISTNpyDataset(TEST_DIR)

    eff_batch = BATCH_SIZE
    num_workers = 0
    prefetch_factor = None
    if optimize_cpu or pool_type.lower() == 'min':
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
        eff_batch = max(128, BATCH_SIZE)  # MNIST puede subir batch en CPU
        num_workers = max(1, (os.cpu_count() or 2)//2)
        prefetch_factor = 2

    tr_args = dict(batch_size=eff_batch, shuffle=True)
    if num_workers:
        tr_args.update(num_workers=num_workers, prefetch_factor=prefetch_factor)
    train_loader = DataLoader(train_ds, **tr_args)

    vl_args = dict(batch_size=eff_batch, shuffle=False)
    if num_workers:
        vl_args.update(num_workers=num_workers)
    val_loader = DataLoader(val_ds, **vl_args)

    te_args = dict(batch_size=eff_batch, shuffle=False)
    if num_workers:
        te_args.update(num_workers=num_workers)
    test_loader = DataLoader(test_ds, **te_args)

    model = CNNMNIST(pool_type=pool_type).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, phase="Val")
        print(f"Train Loss={tr_loss:.4f} Acc={tr_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)

    curves_png, curves_svg = plot_curves(history, suffix)
    report, cm = evaluate_test(model, test_loader)
    # Calcular métricas numéricas (accuracy test y macro f1)
    # Extraer labels y preds nuevamente para F1
    test_ds_iter = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_ds_iter:
            out = model(x.to(DEVICE))
            preds = out.argmax(dim=1).cpu().numpy().tolist()
            all_preds.extend(preds); all_labels.extend(y.numpy().tolist())
    test_acc = sum(p==l for p,l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm_png, cm_svg = save_confusion_matrix(cm, suffix)

    # Guardar gráfico de accuracy
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6,3.6))
    ax.bar(['Accuracy'], [test_acc], color=['#2563eb'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Proporción')
    ax.set_title(f'MNIST Test Accuracy ({suffix})')
    ax.text(0, test_acc+0.02 if test_acc<0.95 else test_acc-0.05, f"{test_acc*100:.1f}%", ha='center', va='bottom' if test_acc<0.95 else 'top')
    fig.tight_layout()
    acc_png = METRICS_MNIST_DIR / f'mnist_accuracy_{suffix}.png'
    acc_svg = METRICS_MNIST_DIR / f'mnist_accuracy_{suffix}.svg'
    fig.savefig(acc_png, dpi=120)
    fig.savefig(acc_svg)
    plt.close(fig)

    # Guardar modelo específico
    model_path = MODEL_MNIST_DIR / f'mnist_cnn_model_{suffix}.pth'
    torch.save(model.state_dict(), model_path)

    metrics_json = METRICS_MNIST_DIR / f'mnist_metrics_{suffix}.json'
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump({
            'hyperparameters': {'learning_rate': lr, 'epochs': epochs, 'batch_size': BATCH_SIZE},
            'history': history,
            'report': report,
            'test_accuracy': test_acc,
            'test_macro_f1': macro_f1,
            'model_path': str(model_path),
            'curves_png': str(curves_png),
            'curves_svg': str(METRICS_MNIST_DIR / f'mnist_learning_curves_{suffix}.svg'),
            'cm_png': str(cm_png),
            'cm_svg': str(METRICS_MNIST_DIR / f'mnist_confusion_matrix_{suffix}.svg'),
            'accuracy_png': str(acc_png),
            'accuracy_svg': str(acc_svg)
        }, f, ensure_ascii=False, indent=2)
    print(f"Resumen guardado en {metrics_json}")

    artifacts = {
        'model_path': str(model_path),
        'curves_png': str(curves_png),
        'curves_svg': str(METRICS_MNIST_DIR / f'mnist_learning_curves_{suffix}.svg'),
        'cm_png': str(cm_png),
        'cm_svg': str(METRICS_MNIST_DIR / f'mnist_confusion_matrix_{suffix}.svg'),
        'accuracy_png': str(acc_png),
        'accuracy_svg': str(acc_svg),
        'metrics_json': str(metrics_json)
    }
    metrics = {'accuracy': test_acc, 'macro_f1': macro_f1}
    return model, history, metrics, artifacts


def main():
    print(f"Dispositivo: {DEVICE}")
    # Ejecutar baseline y alt
    base_model, base_history, base_metrics, base_artifacts = run_mnist_experiment(LEARNING_RATE, EPOCHS, 'baseline')
    alt_model, alt_history, alt_metrics, alt_artifacts = run_mnist_experiment(ALT_LEARNING_RATE, ALT_EPOCHS, 'alt_lr')
    # Selección mejor por macro F1
    best_is_base = base_metrics['macro_f1'] >= alt_metrics['macro_f1']
    best_model = base_model if best_is_base else alt_model
    best_path = MODEL_MNIST_DIR / 'mnist_best_model.pth'
    torch.save(best_model.state_dict(), best_path)
    print(f"Mejor modelo guardado en {best_path} ({'baseline' if best_is_base else 'alt_lr'})")
    # Resumen comparativo
    summary_path = METRICS_MNIST_DIR / 'mnist_experiments_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {'metrics': base_metrics, 'artifacts': base_artifacts},
            'alt_lr': {'metrics': alt_metrics, 'artifacts': alt_artifacts},
            'best': 'baseline' if best_is_base else 'alt_lr'
        }, f, ensure_ascii=False, indent=2)
    print(f"Resumen comparativo guardado en {summary_path}")

if __name__ == '__main__':
    main()
