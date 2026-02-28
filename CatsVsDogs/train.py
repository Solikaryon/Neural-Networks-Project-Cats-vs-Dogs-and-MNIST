import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import PetImageDataset
from .convolutional_nn import CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import json
from .paths import TRAIN_DIR, VAL_DIR, TEST_DIR, METRICS_DIR, MODEL_DIR

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Segundo experimento (hiperparámetros alternativos)
ALT_LEARNING_RATE = 0.0005  # diferente LR
ALT_NUM_EPOCHS = 8          # menos épocas para el experimento alterno

def train_one_epoch(model, loader, optimizer, criterion):
    """
    Runs a single training epoch.
    """
    model.train() 
    running_loss = 0.0
    
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Training"):
        # Move data to the selected device (GPU or CPU)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass: get model predictions
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update the model's weights
        optimizer.step()
        
        # Accumulate the loss
        running_loss += loss.item() * inputs.size(0)

        # Accuracy en entrenamiento (usar logits -> sigmoid -> threshold)
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total if total > 0 else 0.0
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion):
    """
    Runs a single validation epoch.
    """
    model.eval() # Set the model to evaluation mode (disables dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            # Apply sigmoid to get probabilities (0 to 1)
            # Then threshold at 0.5 to get binary predictions (0 or 1)
            preds = torch.sigmoid(outputs) > 0.5
            
            # Count correct predictions
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total if total > 0 else 0.0
    print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate_test(model, loader):
    """Evalúa en el set de prueba y genera métricas detalladas."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().ravel().tolist())

    # Métricas
    report = classification_report(all_labels, all_preds, target_names=["Cat", "Dog"], digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=1)
    accuracy = (cm.trace()) / cm.sum()

    print("\n=== Test Metrics ===")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }


# Directorios de salida
ARTIFACTS_DIR = METRICS_DIR  # alias local para compatibilidad con lógica previa

def plot_learning_curves(history, title_suffix="Baseline"):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10,4))
    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Learning Curve (Loss) - {title_suffix}')
    plt.legend()
    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Learning Curve (Acc) - {title_suffix}')
    plt.legend()
    plt.tight_layout()
    # Guardar PNG y SVG
    safe_title = title_suffix.replace(' ','_').lower()
    png_path = ARTIFACTS_DIR / f"learning_curves_{safe_title}.png"
    svg_path = ARTIFACTS_DIR / f"learning_curves_{safe_title}.svg"
    plt.savefig(png_path, dpi=120)
    plt.savefig(svg_path)
    plt.close()
    print(f"Curvas guardadas en {png_path} y {svg_path}")

def save_accuracy_chart(accuracy: float, title_suffix: str):
    """Genera un gráfico simple (barra) con el porcentaje de acierto y lo guarda en PNG y SVG."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.6,3.6))
    ax.bar(['Accuracy'], [accuracy], color=['#16a34a'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Proporción')
    ax.set_title(f'Accuracy Test - {title_suffix}')
    ax.text(0, accuracy+0.02 if accuracy<0.95 else accuracy-0.05, f"{accuracy*100:.1f}%", ha='center', va='bottom' if accuracy<0.95 else 'top')
    fig.tight_layout()
    safe_title = title_suffix.replace(' ','_').lower()
    acc_png = ARTIFACTS_DIR / f"accuracy_{safe_title}.png"
    acc_svg = ARTIFACTS_DIR / f"accuracy_{safe_title}.svg"
    fig.savefig(acc_png, dpi=120)
    fig.savefig(acc_svg)
    plt.close(fig)
    print(f"Accuracy chart guardado en {acc_png} y {acc_svg}")
    return acc_png, acc_svg

def run_experiment(learning_rate, num_epochs, title_suffix="Baseline", apply_sigmoid=False, pool_type: str = "max", optimize_cpu: bool = False):
    print(f"\n=== Experimento: {title_suffix} | LR={learning_rate} | Epochs={num_epochs} ===")
    import os
    # Datasets
    train_dataset = PetImageDataset(data_dir=TRAIN_DIR)
    val_dataset = PetImageDataset(data_dir=VAL_DIR)
    test_dataset = PetImageDataset(data_dir=TEST_DIR)

    # Optimización CPU solo para MinPooling si se solicita
    eff_batch = BATCH_SIZE
    num_workers = 0
    prefetch_factor = None
    if optimize_cpu or pool_type.lower() == 'min':
        try:
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        except Exception:
            pass
        eff_batch = max(32, BATCH_SIZE)  # mantener al menos 32
        num_workers = max(1, (os.cpu_count() or 2)//2)
        prefetch_factor = 2

    loader_args = dict(batch_size=eff_batch, shuffle=True)
    if num_workers:
        loader_args.update(num_workers=num_workers)
        if prefetch_factor:
            loader_args.update(prefetch_factor=prefetch_factor)
    train_loader = DataLoader(train_dataset, **loader_args)

    val_args = dict(batch_size=eff_batch, shuffle=False)
    if num_workers:
        val_args.update(num_workers=num_workers)
    val_loader = DataLoader(val_dataset, **val_args)

    test_args = dict(batch_size=eff_batch, shuffle=False)
    if num_workers:
        test_args.update(num_workers=num_workers)
    test_loader = DataLoader(test_dataset, **test_args)

    model = CNN(apply_sigmoid=False, pool_type=pool_type).to(DEVICE)  # logits para BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    plot_learning_curves(history, title_suffix=title_suffix)
    test_metrics = evaluate_test(model, test_loader)
    # Guardar gráfico de accuracy
    acc_png, acc_svg = save_accuracy_chart(test_metrics['accuracy'], title_suffix)
    # Guardar matriz de confusión como PNG y SVG
    cm = test_metrics['confusion_matrix']
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title(f'Matriz de Confusión - {title_suffix}')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.xticks([0,1],["Cat","Dog"])
    plt.yticks([0,1],["Cat","Dog"])
    plt.tight_layout()
    safe_title = title_suffix.replace(' ','_').lower()
    cm_png = ARTIFACTS_DIR / f"confusion_matrix_{safe_title}.png"
    cm_svg = ARTIFACTS_DIR / f"confusion_matrix_{safe_title}.svg"
    plt.savefig(cm_png, dpi=120)
    plt.savefig(cm_svg)
    plt.close()
    print(f"Matriz de confusión guardada en {cm_png} y {cm_svg}")
    # Guardar JSON de métricas
    metrics_json = ARTIFACTS_DIR / f"metrics_{safe_title}.json"
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': title_suffix,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'epochs': num_epochs,
                'batch_size': BATCH_SIZE
            },
            'history': history,
            'test_metrics': {
                'accuracy': test_metrics['accuracy'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'f1': test_metrics['f1']
            }
        }, f, ensure_ascii=False, indent=2)
    artifacts = {
        'curves_png': str(ARTIFACTS_DIR / f"learning_curves_{safe_title}.png"),
        'curves_svg': str(ARTIFACTS_DIR / f"learning_curves_{safe_title}.svg"),
        'accuracy_png': str(acc_png),
        'accuracy_svg': str(acc_svg),
        'cm_png': str(cm_png),
        'cm_svg': str(cm_svg),
        'metrics_json': str(metrics_json)
    }
    return model, history, test_metrics, artifacts


def main():
    """
    Main function to run the training process.
    """
    print(f"Using device: {DEVICE}")
    # Experimento base
    base_model, base_history, base_test, base_artifacts = run_experiment(LEARNING_RATE, NUM_EPOCHS, title_suffix="Baseline")

    # Segundo experimento con LR diferente
    alt_model, alt_history, alt_test, alt_artifacts = run_experiment(ALT_LEARNING_RATE, ALT_NUM_EPOCHS, title_suffix="Alt LR")

    print("\n=== Comparativa Métricas Test ===")
    print(f"Baseline: Acc={base_test['accuracy']:.4f} Precision={base_test['precision']:.4f} Recall={base_test['recall']:.4f} F1={base_test['f1']:.4f}")
    print(f"Alt LR  : Acc={alt_test['accuracy']:.4f} Precision={alt_test['precision']:.4f} Recall={alt_test['recall']:.4f} F1={alt_test['f1']:.4f}")

    # Guardar mejor modelo según F1
    best_model = base_model if base_test['f1'] >= alt_test['f1'] else alt_model
    model_path = MODEL_DIR / "best_pet_cnn_model.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"Modelo guardado: {model_path}")
    # Guardar resumen comparativo
    summary_json = METRICS_DIR / "experiments_summary.json"
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': base_artifacts,
            'alt_lr': alt_artifacts,
            'best_model_path': str(model_path),
            'comparison': {
                'baseline_f1': base_test['f1'],
                'alt_lr_f1': alt_test['f1']
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Resumen de experimentos guardado en {summary_json}")

if __name__ == "__main__":
    main()