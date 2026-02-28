import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import time
import os
import traceback
import json
import random
import importlib.util
import sys

import numpy as np
import torch
from PIL import Image

# Rutas generales del proyecto
from CatsVsDogs.paths import RAW_DATA_DIR as CATS_RAW_DIR, PROCESSED_DATA_DIR as CATS_PROCESSED_DIR, METRICS_DIR as CATS_METRICS_DIR, MODEL_DIR as CATS_MODEL_DIR

# Debido al guión en 'MNIST-Digits', debemos cargar dinámicamente sus módulos
PROJECT_ROOT = Path(__file__).resolve().parent
MNIST_DIR = PROJECT_ROOT / 'MNIST-Digits'
MNIST_PATHS_FILE = MNIST_DIR / 'paths_mnist.py'
MNIST_PRE_FILE = MNIST_DIR / 'pre_processingMNIST.py'
MNIST_DATASET_FILE = MNIST_DIR / 'datasetMNIST.py'
MNIST_CNN_FILE = MNIST_DIR / 'convolutional_nnMNIST.py'
MNIST_TRAIN_FILE = MNIST_DIR / 'trainMNIST.py'

# Asegurar que la carpeta MNIST-Digits esté en sys.path para que los imports internos (paths_mnist, etc.) funcionen
mnist_dir_str = str(MNIST_DIR)
if mnist_dir_str not in sys.path:
    sys.path.append(mnist_dir_str)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- Utilidades de import dinámico -----
def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

# Cargar paths MNIST para rutas
try:
    mnist_paths = load_module_from_path('paths_mnist_dyn', MNIST_PATHS_FILE)
    MNIST_PROCESSED_DIR = mnist_paths.MNIST_PROCESSED_DIR
    MNIST_METRICS_DIR = mnist_paths.METRICS_MNIST_DIR
    MNIST_MODEL_DIR = mnist_paths.MODEL_MNIST_DIR
    MNIST_TRAIN_DIR = mnist_paths.TRAIN_DIR
    MNIST_VAL_DIR = mnist_paths.VAL_DIR
    MNIST_TEST_DIR = mnist_paths.TEST_DIR
except Exception:
    mnist_paths = None
    MNIST_PROCESSED_DIR = PROJECT_ROOT / 'MNIST_preprocesado'
    MNIST_METRICS_DIR = PROJECT_ROOT / 'MetricasMNIST'
    MNIST_MODEL_DIR = PROJECT_ROOT / 'ModeloMNIST'
    MNIST_TRAIN_DIR = MNIST_PROCESSED_DIR / 'train'
    MNIST_VAL_DIR = MNIST_PROCESSED_DIR / 'val'
    MNIST_TEST_DIR = MNIST_PROCESSED_DIR / 'test'
    for d in [MNIST_PROCESSED_DIR, MNIST_METRICS_DIR, MNIST_MODEL_DIR, MNIST_TRAIN_DIR, MNIST_VAL_DIR, MNIST_TEST_DIR]:
        d.mkdir(exist_ok=True)

# ----- Interfaz Principal -----
class RootSelector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Seleccionar Dataset')
        self.geometry('350x160')
        self.resizable(False, False)
        ttk.Label(self, text='Seleccione el conjunto de datos:', font=('Segoe UI', 11, 'bold')).pack(pady=10)
        self.choice = tk.StringVar(value='cats')
        frm = ttk.Frame(self)
        frm.pack(pady=5)
        ttk.Radiobutton(frm, text='Cats vs Dogs', variable=self.choice, value='cats').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Radiobutton(frm, text='MNIST Digits', variable=self.choice, value='mnist').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Button(self, text='Continuar', command=self._continue).pack(pady=10)
        self.protocol('WM_DELETE_WINDOW', self.destroy)

    def _continue(self):
        choice = self.choice.get()
        self.destroy()
        if choice == 'cats':
            app = CatsDogsApp()
        else:
            app = MNISTApp()
        app.mainloop()

# ----- Cats vs Dogs App -----
class CatsDogsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Cats vs Dogs - Interfaz')
        self.geometry('840x620')
        self.resizable(False, False)
        self.running_task = False
        self.model = None
        from CatsVsDogs.convolutional_nn import CNN  # lazy import
        self.CNNClass = CNN
        try:
            from CatsVsDogs.pre_processing import PreProcesser, IMG_SIZE as PRE_IMG_SIZE
            self.PreProcesser = PreProcesser
            self.IMG_SIZE = PRE_IMG_SIZE
        except Exception:
            self.PreProcesser = None
            self.IMG_SIZE = (150, 150)
        try:
            from CatsVsDogs.train import run_experiment, LEARNING_RATE, NUM_EPOCHS, ALT_LEARNING_RATE, ALT_NUM_EPOCHS
            self.run_experiment = run_experiment
            self.LEARNING_RATE = LEARNING_RATE
            self.NUM_EPOCHS = NUM_EPOCHS
            self.ALT_LEARNING_RATE = ALT_LEARNING_RATE
            self.ALT_NUM_EPOCHS = ALT_NUM_EPOCHS
        except Exception:
            self.run_experiment = None
            self.LEARNING_RATE = 0.001
            self.NUM_EPOCHS = 5
            self.ALT_LEARNING_RATE = 0.0005
            self.ALT_NUM_EPOCHS = 8
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        paths_frame = ttk.LabelFrame(frm, text='Rutas')
        paths_frame.pack(fill=tk.X, pady=5)
        ttk.Label(paths_frame, text='Dataset bruto:').grid(row=0, column=0, sticky='w')
        self.raw_path_var = tk.StringVar(value=str(CATS_RAW_DIR))
        ttk.Entry(paths_frame, textvariable=self.raw_path_var, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(paths_frame, text='Seleccionar', command=self._select_raw).grid(row=0, column=2, padx=5)

        ttk.Label(paths_frame, text='Procesado destino:').grid(row=1, column=0, sticky='w')
        self.proc_path_var = tk.StringVar(value=str(CATS_PROCESSED_DIR))
        ttk.Entry(paths_frame, textvariable=self.proc_path_var, width=70).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(paths_frame, text='Seleccionar', command=self._select_proc).grid(row=1, column=2, padx=5)

        btn_frame = ttk.LabelFrame(frm, text='Acciones')
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_preprocess = ttk.Button(btn_frame, text='Preprocesar', command=self._run_preprocess)
        self.btn_preprocess.grid(row=0, column=0, padx=5, pady=5)
        self.btn_train = ttk.Button(btn_frame, text='Entrenar (Baseline + Alt)', command=self._run_train)
        self.btn_train.grid(row=0, column=1, padx=5, pady=5)
        self.btn_train_min = ttk.Button(btn_frame, text='Entrenar MinPooling', command=self._run_train_minpool)
        self.btn_train_min.grid(row=0, column=2, padx=5, pady=5)
        self.btn_load = ttk.Button(btn_frame, text='Cargar Modelo', command=self._load_model)
        self.btn_load.grid(row=0, column=3, padx=5, pady=5)
        # Botón actualizado para soportar múltiples imágenes y generar métricas
        self.btn_predict = ttk.Button(btn_frame, text='Predecir Imágenes', command=self._predict_image)
        self.btn_predict.grid(row=0, column=4, padx=5, pady=5)
        self.btn_results = ttk.Button(btn_frame, text='Ver Resultados', command=self._show_results)
        self.btn_results.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(btn_frame, text='Experimento:').grid(row=1, column=0, padx=5)
        self.exp_var = tk.StringVar(value='baseline')
        self.exp_combo = ttk.Combobox(btn_frame, textvariable=self.exp_var, values=['baseline','alt_lr'], state='readonly', width=12)
        self.exp_combo.grid(row=1, column=1, padx=5)

        log_frame = ttk.LabelFrame(frm, text='Log')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, wrap='word', height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log(f'Dispositivo: {DEVICE}')
        if DEVICE == 'cuda':
            try:
                self._log(f'GPU: {torch.cuda.get_device_name(0)}')
            except Exception:
                pass
        model_path = CATS_MODEL_DIR / 'best_pet_cnn_model.pth'
        self._log(f'Modelo existente: {"Sí" if model_path.exists() else "No"}')

    def _select_raw(self):
        d = filedialog.askdirectory()
        if d: self.raw_path_var.set(d)
    def _select_proc(self):
        d = filedialog.askdirectory()
        if d: self.proc_path_var.set(d)
    def _log(self, msg):
        ts = time.strftime('[%H:%M:%S]')
        self.log_text.insert(tk.END, f'{ts} {msg}\n'); self.log_text.see(tk.END)
    def _set_buttons(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in [self.btn_preprocess, self.btn_train, self.btn_train_min, self.btn_load, self.btn_predict, self.btn_results]:
            b.configure(state=state)

    def _run_preprocess(self):
        if self.PreProcesser is None: messagebox.showerror('Error','PreProcesser no disponible'); return
        if self.running_task: return
        self.running_task = True; self._set_buttons(False); self._log('Iniciando preprocesamiento...')
        threading.Thread(target=self._task_preprocess).start()

    def _task_preprocess(self):
        try:
            raw = self.raw_path_var.get(); proc = self.proc_path_var.get(); os.makedirs(proc, exist_ok=True)
            pp = self.PreProcesser(raw, proc, self.IMG_SIZE)
            pp.preprocess()
            self._log('Preprocesamiento completado.')
        except Exception as e:
            self._log(f'Error preprocesando: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _run_train(self):
        if self.run_experiment is None: messagebox.showerror('Error','Función run_experiment no disponible'); return
        if self.running_task: return
        self.running_task = True; self._set_buttons(False); self._log('Ejecutando experimentos: baseline y alt_lr...')
        threading.Thread(target=self._task_train).start()

    def _task_train(self):
        try:
            # Baseline
            base_model, _, base_metrics, base_artifacts = self.run_experiment(self.LEARNING_RATE, self.NUM_EPOCHS, title_suffix='Baseline')
            self._log(f'Baseline -> Acc={base_metrics["accuracy"]:.4f} F1={base_metrics["f1"]:.4f}')
            # Alt LR
            alt_model, _, alt_metrics, alt_artifacts = self.run_experiment(self.ALT_LEARNING_RATE, self.ALT_NUM_EPOCHS, title_suffix='Alt LR')
            self._log(f'Alt LR -> Acc={alt_metrics["accuracy"]:.4f} F1={alt_metrics["f1"]:.4f}')
            # Seleccionar mejor por F1
            best_model = base_model if base_metrics['f1'] >= alt_metrics['f1'] else alt_model
            torch.save(best_model.state_dict(), CATS_MODEL_DIR / 'best_pet_cnn_model.pth')
            self.model = best_model
            self._log('Modelo mejor guardado (best_pet_cnn_model.pth).')
            # Guardar resumen comparativo
            summary_path = CATS_METRICS_DIR / 'experiments_summary_gui.json'
            import json
            json.dump({
                'baseline': base_artifacts,
                'alt_lr': alt_artifacts,
                'baseline_f1': base_metrics['f1'],
                'alt_lr_f1': alt_metrics['f1'],
                'best': 'baseline' if base_metrics['f1'] >= alt_metrics['f1'] else 'alt_lr'
            }, open(summary_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
            self._log(f'Resumen guardado en {summary_path}')
        except Exception as e:
            self._log(f'Error entrenamiento: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _run_train_minpool(self):
        if self.run_experiment is None: messagebox.showerror('Error','Función run_experiment no disponible'); return
        if self.running_task: return
        self.running_task = True; self._set_buttons(False); self._log('Entrenando MinPooling (CatsVsDogs)...')
        threading.Thread(target=self._task_train_minpool).start()

    def _task_train_minpool(self):
        try:
            model, _, metrics, artifacts = self.run_experiment(self.LEARNING_RATE, self.NUM_EPOCHS, title_suffix='MinPooling', pool_type='min', optimize_cpu=True)
            save_path = CATS_MODEL_DIR / 'MinPooling_CatVsDog.pth'
            torch.save(model.state_dict(), save_path)
            self.model = model
            self._log(f'MinPooling entrenado. Acc={metrics["accuracy"]:.4f} F1={metrics["f1"]:.4f}. Modelo: {save_path.name}')
        except Exception as e:
            self._log(f'Error entrenando MinPooling CatsVsDogs: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _load_model(self):
        # Permitir al usuario elegir el archivo de modelo (.pth)
        initialdir = str(CATS_MODEL_DIR)
        file_path = filedialog.askopenfilename(title='Seleccionar modelo (.pth)', initialdir=initialdir, filetypes=[('PyTorch Model','*.pth')])
        if not file_path:
            return
        try:
            model = self.CNNClass(apply_sigmoid=True).to(DEVICE)
            state = torch.load(file_path, map_location=DEVICE)
            model.load_state_dict(state); model.eval(); self.model = model
            self._log(f'Modelo cargado desde {os.path.basename(file_path)}.')
        except Exception as e:
            self._log(f'Error cargando modelo: {e}')

    def _predict_image(self):
        """Permite seleccionar una o varias imágenes. Si se seleccionan varias,
        genera una gráfica de precisión por clase.
        La etiqueta verdadera se infiere del nombre de carpeta ('Cat' / 'Dog')."""
        if self.model is None:
            messagebox.showwarning('Modelo','Primero cargue/entrene el modelo'); return

        file_paths = filedialog.askopenfilenames(title='Seleccionar imágenes o arrays .npy', filetypes=[('Imágenes o NumPy','*.jpg *.jpeg *.png *.bmp *.npy')])
        if not file_paths: return

        # Si solo hay una imagen, mantener comportamiento previo
        if len(file_paths) == 1:
            file_path = file_paths[0]
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == '.npy':
                    arr = np.load(file_path)
                    # Normalizar / adaptar forma
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        pass  # (H,W,3)
                    elif arr.ndim == 2:  # gris -> repetir canales
                        arr = np.stack([arr]*3, axis=-1)
                    elif arr.ndim == 3 and arr.shape[0] == 3:  # (3,H,W) -> convertir a (H,W,3)
                        arr = np.transpose(arr, (1,2,0))
                    # Escalar si valores parecen 0-255
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.max() > 1.5:  # asumir rango 0-255
                        arr = arr / 255.0
                    # Redimensionar
                    pil_img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB').resize(self.IMG_SIZE)
                    arr = np.array(pil_img, dtype=np.float32)/255.0
                else:
                    img = Image.open(file_path).convert('RGB').resize(self.IMG_SIZE)
                    arr = np.array(img, dtype=np.float32)/255.0
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): prob = self.model(tensor).cpu().numpy().ravel()[0]
                label = 'Dog' if prob >= 0.5 else 'Cat'
                self._log(f'Predicción: {label} prob={prob:.4f} archivo={os.path.basename(file_path)}')
                messagebox.showinfo('Resultado', f'Etiqueta: {label}\nProb Dog: {prob:.4f}')
            except Exception as e:
                self._log(f'Error predicción: {e}')
            return

        # Múltiples imágenes: calcular predicciones y métricas
        preds = []
        trues = []
        probs_dog = []
        for fp in file_paths:
            try:
                ext = os.path.splitext(fp)[1].lower()
                if ext == '.npy':
                    arr = np.load(fp)
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        pass
                    elif arr.ndim == 2:
                        arr = np.stack([arr]*3, axis=-1)
                    elif arr.ndim == 3 and arr.shape[0] == 3:
                        arr = np.transpose(arr, (1,2,0))
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.max() > 1.5:
                        arr = arr / 255.0
                    pil_img = Image.fromarray((arr*255).astype(np.uint8)).convert('RGB').resize(self.IMG_SIZE)
                    arr = np.array(pil_img, dtype=np.float32)/255.0
                else:
                    img = Image.open(fp).convert('RGB').resize(self.IMG_SIZE)
                    arr = np.array(img, dtype=np.float32)/255.0
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): prob = self.model(tensor).cpu().numpy().ravel()[0]
                pred_label = 'Dog' if prob >= 0.5 else 'Cat'
                base = os.path.basename(fp)
                name_no_ext = os.path.splitext(base)[0].lower()
                # Patron Cat-N o Dog-N (permitir cat_123, dog-45, etc.)
                if name_no_ext.startswith('cat'):
                    true_label = 'Cat'
                elif name_no_ext.startswith('dog'):
                    true_label = 'Dog'
                else:
                    # Fallback: usar directorios
                    lower_parts = [p.lower() for p in fp.split(os.sep)]
                    if 'dog' in lower_parts: true_label = 'Dog'
                    elif 'cat' in lower_parts: true_label = 'Cat'
                    else: true_label = 'Desconocida'
                preds.append(pred_label); trues.append(true_label); probs_dog.append(prob)
                self._log(f'Archivo={os.path.basename(fp)} -> Pred={pred_label} ProbDog={prob:.4f} True={true_label}')
            except Exception as e:
                self._log(f'Error predicción {os.path.basename(fp)}: {e}')

        # Calcular precisión por clase (solo si hay etiquetas conocidas) y accuracy global
        classes = ['Cat','Dog']
        precision = {}
        for cls in classes:
            tp = sum(1 for p,t in zip(preds,trues) if p==cls and t==cls)
            fp = sum(1 for p,t in zip(preds,trues) if p==cls and t!=cls and t!='Desconocida')
            denom = tp + fp
            precision[cls] = tp/denom if denom>0 else 0.0
        known_mask = [t!='Desconocida' for t in trues]
        total_known = sum(known_mask)
        correct = sum(1 for p,t in zip(preds,trues) if t!='Desconocida' and p==t)
        overall_acc = correct/total_known if total_known>0 else 0.0
        mismatches = total_known - correct

        # Mostrar gráfico de barras incluyendo accuracy global
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.6,3.2))
            bar_labels = classes + ['Accuracy']
            bar_values = [precision[c] for c in classes] + [overall_acc]
            colors = ['steelblue','indianred','darkorange']
            ax.bar(bar_labels, bar_values, color=colors)
            ax.set_ylim(0,1)
            ax.set_ylabel('Valor')
            ax.set_title('Precisión por clase y Accuracy global')
            for i,v in enumerate(bar_values):
                ax.text(i, v+0.02 if v<0.95 else v-0.05, f'{v:.2f}', ha='center', va='bottom' if v<0.95 else 'top', fontsize=9)
            fig.tight_layout()
            win = tk.Toplevel(self); win.title('Métricas de Predicción')
            from PIL import ImageTk
            tmp_path = CATS_METRICS_DIR / 'CvsD_multi_accuracy_temp.png'
            tmp_svg = CATS_METRICS_DIR / 'CvsD_multi_accuracy_temp.svg'
            fig.savefig(tmp_path)
            fig.savefig(tmp_svg)
            plt.close(fig)
            im = Image.open(tmp_path)
            tkimg = ImageTk.PhotoImage(im)
            lbl = ttk.Label(win, image=tkimg); lbl.image = tkimg; lbl.pack(padx=5,pady=5)
            msg_lines = [f'{c} precisión: {precision[c]:.4f}' for c in classes]
            msg_lines.append(f'Accuracy global: {overall_acc:.4f} (sobre {total_known} etiquetadas)')
            msg_lines.append(f'Mismatches: {mismatches}')
            messagebox.showinfo('Resumen múltiple', 'Imágenes evaluadas: '+str(len(file_paths))+'\n'+'\n'.join(msg_lines))
        except Exception as e:
            self._log(f'Error generando gráfica métricas: {e}')

    def _show_results(self):
        exp = self.exp_var.get()
        curves_png = CATS_METRICS_DIR / f'learning_curves_{exp}.png'
        cm_png = CATS_METRICS_DIR / f'confusion_matrix_{exp}.png'
        acc_png = CATS_METRICS_DIR / f'accuracy_{exp}.png'
        metrics_json = CATS_METRICS_DIR / f'metrics_{exp}.json'
        if not metrics_json.exists(): self._log('No existe JSON de métricas para ese experimento'); return
        try:
            data = json.load(open(metrics_json,'r',encoding='utf-8'))
            test = data['test_metrics']
            self._log(f'Resultados {exp}: Acc={test["accuracy"]:.4f} F1={test["f1"]:.4f}')
            imgs = [p for p in [curves_png, cm_png, acc_png] if p.exists()]
            if not imgs: self._log('No hay imágenes de resultados'); return
            win = tk.Toplevel(self); win.title(f'Resultados - {exp}')
            from PIL import ImageTk
            for p in imgs:
                im = Image.open(p); tkimg = ImageTk.PhotoImage(im)
                lbl = ttk.Label(win, image=tkimg); lbl.image = tkimg; lbl.pack(padx=5,pady=5)
        except Exception as e:
            self._log(f'Error mostrando resultados: {e}')

# ----- MNIST App -----
class MNISTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MNIST Digits - Interfaz')
        self.geometry('820x620')
        self.resizable(False, False)
        self.running_task = False
        self.model = None
        self._lazy_load_modules()
        self._build_ui()

    def _lazy_load_modules(self):
        try:
            self.pre_module = load_module_from_path('mnist_pre', MNIST_PRE_FILE)
        except Exception:
            self.pre_module = None
        try:
            self.dataset_module = load_module_from_path('mnist_dataset', MNIST_DATASET_FILE)
        except Exception:
            self.dataset_module = None
        try:
            self.cnn_module = load_module_from_path('mnist_cnn', MNIST_CNN_FILE)
        except Exception:
            self.cnn_module = None
        try:
            self.train_module = load_module_from_path('mnist_train', MNIST_TRAIN_FILE)
        except Exception:
            self.train_module = None

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10); frm.pack(fill=tk.BOTH, expand=True)
        info_frame = ttk.LabelFrame(frm, text='Información')
        info_frame.pack(fill=tk.X, pady=5)
        ttk.Label(info_frame, text='MNIST se descarga automáticamente (no se seleccionan imágenes locales).').pack(anchor='w', padx=5, pady=2)
        ttk.Label(info_frame, text=f'Carpeta procesada: {MNIST_PROCESSED_DIR}').pack(anchor='w', padx=5, pady=2)

        btn_frame = ttk.LabelFrame(frm, text='Acciones')
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_preprocess = ttk.Button(btn_frame, text='Descargar/Preprocesar', command=self._run_preprocess)
        self.btn_preprocess.grid(row=0, column=0, padx=5, pady=5)
        self.btn_train = ttk.Button(btn_frame, text='Entrenar (Baseline + Alt)', command=self._run_train_dual)
        self.btn_train.grid(row=0, column=1, padx=5, pady=5)
        self.btn_train_min = ttk.Button(btn_frame, text='Entrenar MinPooling', command=self._run_train_minpool)
        self.btn_train_min.grid(row=0, column=2, padx=5, pady=5)
        self.btn_load = ttk.Button(btn_frame, text='Cargar Modelo', command=self._load_model)
        self.btn_load.grid(row=0, column=3, padx=5, pady=5)
        self.btn_samples = ttk.Button(btn_frame, text='Ver Predicciones Ejemplo', command=self._show_sample_predictions)
        self.btn_samples.grid(row=0, column=4, padx=5, pady=5)
        self.btn_predict = ttk.Button(btn_frame, text='Predecir Imágenes', command=self._predict_image)
        self.btn_predict.grid(row=0, column=5, padx=5, pady=5)
        self.btn_results = ttk.Button(btn_frame, text='Ver Resultados', command=self._show_results)
        self.btn_results.grid(row=0, column=6, padx=5, pady=5)

        ttk.Label(btn_frame, text='Experimento:').grid(row=1, column=0, padx=5, pady=2)
        self.exp_var = tk.StringVar(value='baseline')
        self.exp_combo = ttk.Combobox(btn_frame, textvariable=self.exp_var, values=['baseline','alt_lr'], state='readonly', width=12)
        self.exp_combo.grid(row=1, column=1, padx=5, pady=2)

        log_frame = ttk.LabelFrame(frm, text='Log')
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, wrap='word', height=25); self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log(f'Dispositivo: {DEVICE}')
        model_path = MNIST_MODEL_DIR / 'mnist_cnn_model.pth'
        self._log(f'Modelo existente: {"Sí" if model_path.exists() else "No"}')

    def _log(self, msg):
        ts = time.strftime('[%H:%M:%S]'); self.log_text.insert(tk.END, f'{ts} {msg}\n'); self.log_text.see(tk.END)
    def _set_buttons(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in [self.btn_preprocess, self.btn_train, self.btn_train_min, self.btn_load, self.btn_samples, self.btn_predict, self.btn_results]: b.configure(state=state)

    def _run_preprocess(self):
        if not self.pre_module: messagebox.showerror('Error','Módulo preprocesamiento MNIST no disponible'); return
        if self.running_task: return
        self.running_task = True; self._set_buttons(False); self._log('Iniciando descarga y preprocesamiento MNIST...')
        threading.Thread(target=self._task_preprocess).start()

    def _task_preprocess(self):
        try:
            prep_class = getattr(self.pre_module, 'MNISTPreprocessor')
            prep = prep_class(); prep.preprocess(); self._log('Preprocesamiento MNIST completado')
        except Exception as e:
            self._log(f'Error preprocesando MNIST: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _run_train_dual(self):
        if not self.train_module: messagebox.showerror('Error','trainMNIST no disponible'); return
        if self.running_task: return
        if not hasattr(self.train_module, 'run_mnist_experiment'):
            messagebox.showerror('Error','run_mnist_experiment no está definido en trainMNIST'); return
        self.running_task = True; self._set_buttons(False); self._log('Ejecutando experimentos MNIST baseline y alt_lr...')
        threading.Thread(target=self._task_train_dual).start()

    def _task_train_dual(self):
        try:
            run_fn = getattr(self.train_module, 'run_mnist_experiment')
            base_model, _, base_metrics, base_artifacts = run_fn(self.train_module.LEARNING_RATE, self.train_module.EPOCHS, 'baseline')
            self._log(f"Baseline -> Acc={base_metrics['accuracy']:.4f} F1={base_metrics['macro_f1']:.4f}")
            alt_model, _, alt_metrics, alt_artifacts = run_fn(self.train_module.ALT_LEARNING_RATE, self.train_module.ALT_EPOCHS, 'alt_lr')
            self._log(f"Alt LR -> Acc={alt_metrics['accuracy']:.4f} F1={alt_metrics['macro_f1']:.4f}")
            best_model = base_model if base_metrics['macro_f1'] >= alt_metrics['macro_f1'] else alt_model
            best_path = MNIST_MODEL_DIR / 'mnist_best_model.pth'
            torch.save(best_model.state_dict(), best_path)
            self.model = best_model
            self._log('Mejor modelo guardado (mnist_best_model.pth)')
            summary_path = MNIST_METRICS_DIR / 'mnist_experiments_summary_gui.json'
            json.dump({
                'baseline': base_artifacts,
                'alt_lr': alt_artifacts,
                'baseline_f1': base_metrics['macro_f1'],
                'alt_lr_f1': alt_metrics['macro_f1'],
                'best': 'baseline' if base_metrics['macro_f1'] >= alt_metrics['macro_f1'] else 'alt_lr'
            }, open(summary_path,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
            self._log(f'Resumen guardado en {summary_path}')
        except Exception as e:
            self._log(f'Error entrenando MNIST: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _run_train_minpool(self):
        if not self.train_module: messagebox.showerror('Error','trainMNIST no disponible'); return
        if self.running_task: return
        if not hasattr(self.train_module, 'run_mnist_experiment'):
            messagebox.showerror('Error','run_mnist_experiment no está definido en trainMNIST'); return
        self.running_task = True; self._set_buttons(False); self._log('Entrenando MNIST con MinPooling...')
        threading.Thread(target=self._task_train_minpool).start()

    def _task_train_minpool(self):
        try:
            run_fn = getattr(self.train_module, 'run_mnist_experiment')
            model, _, metrics, artifacts = run_fn(self.train_module.LEARNING_RATE, self.train_module.EPOCHS, 'minpool', pool_type='min', optimize_cpu=True)
            save_path = MNIST_MODEL_DIR / 'MinPooling_MNIST.pth'
            torch.save(model.state_dict(), save_path)
            self.model = model
            self._log(f'MNIST MinPooling -> Acc={metrics["accuracy"]:.4f} F1={metrics.get("macro_f1", 0):.4f}. Modelo: {save_path.name}')
        except Exception as e:
            self._log(f'Error entrenando MNIST MinPooling: {e}'); traceback.print_exc()
        finally:
            self.running_task = False; self._set_buttons(True)

    def _load_model(self):
        if not self.cnn_module: messagebox.showerror('Error','CNN MNIST no disponible'); return
        # Permitir al usuario elegir el archivo de modelo (.pth)
        initialdir = str(MNIST_MODEL_DIR)
        file_path = filedialog.askopenfilename(title='Seleccionar modelo MNIST (.pth)', initialdir=initialdir, filetypes=[('PyTorch Model','*.pth')])
        if not file_path:
            return
        try:
            CNNMNIST = getattr(self.cnn_module, 'CNNMNIST')
            model = CNNMNIST().to(DEVICE)
            state = torch.load(file_path, map_location=DEVICE)
            model.load_state_dict(state); model.eval(); self.model = model
            self._log(f'Modelo MNIST cargado desde {os.path.basename(file_path)}.')
        except Exception as e:
            self._log(f'Error cargando modelo MNIST: {e}')

    def _show_sample_predictions(self):
        if self.model is None: messagebox.showwarning('Modelo','Primero cargue/entrene el modelo'); return
        if not self.dataset_module: messagebox.showerror('Error','Dataset MNIST no disponible'); return
        try:
            DSClass = getattr(self.dataset_module, 'MNISTNpyDataset')
            ds = DSClass(MNIST_TEST_DIR)
            if len(ds) == 0: self._log('Test MNIST vacío'); return
            indices = random.sample(range(len(ds)), min(9, len(ds)))
            from PIL import ImageTk
            win = tk.Toplevel(self); win.title('Predicciones MNIST')
            grid_frame = ttk.Frame(win); grid_frame.pack(padx=5,pady=5)
            for i, idx in enumerate(indices):
                tensor, label = ds[idx]
                with torch.no_grad():
                    out = self.model(tensor.unsqueeze(0).to(DEVICE))
                    pred = int(out.argmax(dim=1).cpu().item())
                arr = (tensor.squeeze(0).numpy()*255).astype(np.uint8)
                img = Image.fromarray(arr, mode='L').resize((56,56))
                tkimg = ImageTk.PhotoImage(img)
                panel = ttk.Label(grid_frame, image=tkimg, text=f'L:{label} P:{pred}', compound='top')
                panel.image = tkimg
                panel.grid(row=i//3, column=i%3, padx=4, pady=4)
            self._log('Predicciones ejemplo mostradas.')
        except Exception as e:
            self._log(f'Error mostrando ejemplos: {e}')

    def _show_results(self):
        suffix = self.exp_var.get()
        metrics_json = MNIST_METRICS_DIR / f'mnist_metrics_{suffix}.json'
        curves_png = MNIST_METRICS_DIR / f'mnist_learning_curves_{suffix}.png'
        cm_png = MNIST_METRICS_DIR / f'mnist_confusion_matrix_{suffix}.png'
        acc_png = MNIST_METRICS_DIR / f'mnist_accuracy_{suffix}.png'
        if not metrics_json.exists(): self._log(f'No existe mnist_metrics_{suffix}.json'); return
        try:
            data = json.load(open(metrics_json,'r',encoding='utf-8'))
            self._log(f"Acc test={data['test_accuracy']:.4f} F1={data['test_macro_f1']:.4f}")
            imgs = [p for p in [curves_png, cm_png, acc_png] if p.exists()]
            if not imgs: self._log('No hay imágenes de resultados'); return
            win = tk.Toplevel(self); win.title(f'Resultados MNIST - {suffix}')
            from PIL import ImageTk
            for p in imgs:
                im = Image.open(p); tkimg = ImageTk.PhotoImage(im)
                lbl = ttk.Label(win, image=tkimg); lbl.image = tkimg; lbl.pack(padx=5,pady=5)
        except Exception as e:
            self._log(f'Error mostrando resultados MNIST: {e}')

    def _predict_image(self):
        """Permite seleccionar una o varias imágenes tipo dígito. Si son varias,
        calcula accuracy global (acierto) sobre aquellas cuyo label pueda inferirse del nombre del archivo (primer dígito 0-9)."""
        if self.model is None:
            messagebox.showwarning('Modelo','Primero cargue/entrene el modelo'); return
        file_paths = filedialog.askopenfilenames(title='Seleccionar imágenes o arrays .npy (dígitos)', filetypes=[('Imágenes o NumPy','*.jpg *.jpeg *.png *.bmp *.npy')])
        if not file_paths: return
        # Caso único: comportamiento previo
        if len(file_paths) == 1:
            fp = file_paths[0]
            try:
                ext = os.path.splitext(fp)[1].lower()
                if ext == '.npy':
                    arr = np.load(fp)
                    if arr.ndim == 2:
                        pass
                    elif arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.ndim == 3 and arr.shape[2] == 1:
                        arr = arr[:,:,0]
                    else:
                        # Intentar reducir a 2D
                        arr = arr.squeeze()
                        if arr.ndim != 2:
                            raise ValueError('Forma .npy no compatible para MNIST')
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.max() > 1.5:
                        arr = arr/255.0
                    # Redimensionar por si no es 28x28
                    if arr.shape != (28,28):
                        pil_img = Image.fromarray((arr*255).astype(np.uint8), mode='L').resize((28,28))
                        arr = np.array(pil_img, dtype=np.float32)/255.0
                else:
                    img = Image.open(fp).convert('L').resize((28,28))
                    arr = np.array(img, dtype=np.float32)/255.0
                arr = np.expand_dims(arr, axis=0)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
                pred = int(probs.argmax()); prob = float(probs[pred])
                self._log(f'Predicción imagen -> dígito {pred} prob={prob:.4f} archivo={os.path.basename(fp)}')
                messagebox.showinfo('Resultado', f'Dígito: {pred}\nProb: {prob:.4f}')
            except Exception as e:
                self._log(f'Error predicciendo imagen MNIST: {e}')
            return

        # Múltiples imágenes
        preds = []
        trues = []
        for fp in file_paths:
            try:
                ext = os.path.splitext(fp)[1].lower()
                if ext == '.npy':
                    arr = np.load(fp)
                    if arr.ndim == 2:
                        pass
                    elif arr.ndim == 3 and arr.shape[0] == 1:
                        arr = arr[0]
                    elif arr.ndim == 3 and arr.shape[2] == 1:
                        arr = arr[:,:,0]
                    else:
                        arr = arr.squeeze()
                        if arr.ndim != 2:
                            raise ValueError('Forma .npy no compatible para MNIST')
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)
                    if arr.max() > 1.5:
                        arr = arr/255.0
                    if arr.shape != (28,28):
                        pil_img = Image.fromarray((arr*255).astype(np.uint8), mode='L').resize((28,28))
                        arr = np.array(pil_img, dtype=np.float32)/255.0
                else:
                    img = Image.open(fp).convert('L').resize((28,28))
                    arr = np.array(img, dtype=np.float32)/255.0
                arr = np.expand_dims(arr, axis=0)
                tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy().ravel()
                pred = int(probs.argmax())
                base = os.path.basename(fp)
                name_no_ext = os.path.splitext(base)[0]
                # Patrón N-N: tomar token antes de '_' o '-'; si inicia con dígito ese es el label verdadero
                sep_idx = name_no_ext.find('_') if '_' in name_no_ext else name_no_ext.find('-')
                token = name_no_ext if sep_idx == -1 else name_no_ext[:sep_idx]
                true_label = int(token[0]) if token and token[0].isdigit() else None
                preds.append(pred); trues.append(true_label)
                self._log(f'Archivo={base} -> Pred={pred} True={true_label if true_label is not None else "?"}')
            except Exception as e:
                self._log(f'Error predicción {os.path.basename(fp)}: {e}')
                preds.append(None); trues.append(None)

        # Calcular accuracy global sobre aquellos con true_label conocido
        paired = [(p,t) for p,t in zip(preds,trues) if p is not None and t is not None]
        if paired:
            correct = sum(1 for p,t in paired if p==t)
            total = len(paired)
            acc = correct/total
            mismatches = total - correct
        else:
            acc = 0.0; total = 0; mismatches = 0

        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(4.2,3.2))
            ax.bar(['Accuracy'], [acc], color=['purple'])
            ax.set_ylim(0,1)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy múltiple MNIST')
            ax.text(0, acc+0.02 if acc<0.95 else acc-0.05, f'{acc:.2f}', ha='center', va='bottom' if acc<0.95 else 'top')
            fig.tight_layout()
            win = tk.Toplevel(self); win.title('Métrica MNIST múltiple')
            from PIL import ImageTk
            tmp_path = MNIST_METRICS_DIR / 'mnist_multi_accuracy_temp.png'
            tmp_svg = MNIST_METRICS_DIR / 'mnist_multi_accuracy_temp.svg'
            fig.savefig(tmp_path)
            fig.savefig(tmp_svg)
            plt.close(fig)
            im = Image.open(tmp_path); tkimg = ImageTk.PhotoImage(im)
            lbl = ttk.Label(win, image=tkimg); lbl.image = tkimg; lbl.pack(padx=5,pady=5)
            messagebox.showinfo('Resumen múltiple MNIST', f'Imágenes evaluadas: {len(file_paths)}\nCon etiqueta inferida: {total}\nAccuracy: {acc:.4f}\nMismatches: {mismatches}')
            self._log(f'Accuracy múltiple MNIST={acc:.4f} mismatches={mismatches}')
        except Exception as e:
            self._log(f'Error generando gráfica accuracy MNIST: {e}')

if __name__ == '__main__':
    RootSelector().mainloop()
