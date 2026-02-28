import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import time
import sys
import os
import traceback
from PIL import Image
import numpy as np
import torch
import json
from paths import RAW_DATA_DIR, PROCESSED_DATA_DIR, METRICS_DIR, MODEL_DIR

# Importar componentes del proyecto existente
try:
    from pre_processing import PreProcesser, IMG_SIZE as PRE_IMG_SIZE
    RAW_DATASET_PATH = RAW_DATA_DIR
    PROCESSED_PATH = PROCESSED_DATA_DIR
except Exception:
    PreProcesser = None
    RAW_DATASET_PATH = RAW_DATA_DIR
    PROCESSED_PATH = PROCESSED_DATA_DIR
    PRE_IMG_SIZE = (150, 150)

try:
    from convolutional_nn import CNN
except Exception:
    CNN = None

try:
    # Importar función de experimento (devuelve artifacts ahora)
    from train import run_experiment, LEARNING_RATE, NUM_EPOCHS
except Exception:
    run_experiment = None
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = MODEL_DIR / "best_pet_cnn_model.pth"
IMG_SIZE = PRE_IMG_SIZE  # reutilizamos tamaño del preprocesamiento


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificación Cats vs Dogs - Interfaz")
        self.geometry("820x600")
        self.resizable(False, False)

        self._create_widgets()
        self.model = None
        self.running_task = False

    def _create_widgets(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Sección paths
        path_frame = ttk.LabelFrame(frm, text="Rutas")
        path_frame.pack(fill=tk.X, pady=5)
        ttk.Label(path_frame, text="Dataset bruto:").grid(row=0, column=0, sticky="w")
        self.raw_path_var = tk.StringVar(value=str(RAW_DATASET_PATH))
        ttk.Entry(path_frame, textvariable=self.raw_path_var, width=70).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Seleccionar", command=self._select_raw_dir).grid(row=0, column=2, padx=5)

        ttk.Label(path_frame, text="Procesado destino:").grid(row=1, column=0, sticky="w")
        self.processed_path_var = tk.StringVar(value=str(PROCESSED_PATH))
        ttk.Entry(path_frame, textvariable=self.processed_path_var, width=70).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(path_frame, text="Seleccionar", command=self._select_processed_dir).grid(row=1, column=2, padx=5)

        # Botones principales
        btn_frame = ttk.LabelFrame(frm, text="Acciones")
        btn_frame.pack(fill=tk.X, pady=5)
        self.btn_preprocess = ttk.Button(btn_frame, text="Preprocesar", command=self._run_preprocess)
        self.btn_preprocess.grid(row=0, column=0, padx=5, pady=5)

        self.btn_train = ttk.Button(btn_frame, text="Entrenar", command=self._run_train)
        self.btn_train.grid(row=0, column=1, padx=5, pady=5)

        self.btn_load_model = ttk.Button(btn_frame, text="Cargar Modelo", command=self._load_model)
        self.btn_load_model.grid(row=0, column=2, padx=5, pady=5)

        self.btn_predict = ttk.Button(btn_frame, text="Predecir Imagen", command=self._predict_image)
        self.btn_predict.grid(row=0, column=3, padx=5, pady=5)

        self.btn_results = ttk.Button(btn_frame, text="Ver Resultados", command=self._show_results)
        self.btn_results.grid(row=0, column=4, padx=5, pady=5)

        ttk.Label(btn_frame, text="Experimento:").grid(row=1, column=0, padx=5, pady=5)
        self.exp_var = tk.StringVar(value="baseline")
        self.exp_combo = ttk.Combobox(btn_frame, textvariable=self.exp_var, values=["baseline","alt_lr"], width=12, state="readonly")
        self.exp_combo.grid(row=1, column=1, padx=5, pady=5)

        # Área de log
        log_frame = ttk.LabelFrame(frm, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log_text = tk.Text(log_frame, wrap="word", height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self._log("Dispositivo: {}".format(DEVICE))
        self._log("Modelo existente: {}".format("Sí" if MODEL_PATH.exists() else "No"))
        self._log(f"Directorio métricas: {METRICS_DIR}")

    def _select_raw_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.raw_path_var.set(d)

    def _select_processed_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.processed_path_var.set(d)

    def _log(self, msg):
        ts = time.strftime("[%H:%M:%S]")
        self.log_text.insert(tk.END, f"{ts} {msg}\n")
        self.log_text.see(tk.END)

    def _set_buttons_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for b in [self.btn_preprocess, self.btn_train, self.btn_load_model, self.btn_predict, self.btn_results]:
            b.configure(state=state)

    def _run_preprocess(self):
        if PreProcesser is None:
            messagebox.showerror("Error", "No se pudo importar PreProcesser.")
            return
        if self.running_task:
            return
        self.running_task = True
        self._set_buttons_state(False)
        self._log("Iniciando preprocesamiento...")
        t = threading.Thread(target=self._task_preprocess)
        t.start()

    def _task_preprocess(self):
        try:
            raw = self.raw_path_var.get()
            proc = self.processed_path_var.get()
            os.makedirs(proc, exist_ok=True)
            pp = PreProcesser(raw, proc, IMG_SIZE)
            # Llama secuencia completa
            pp.preprocess()
            self._log("Preprocesamiento completado.")
        except Exception as e:
            self._log("Error en preprocesamiento: {}".format(e))
            traceback.print_exc()
        finally:
            self.running_task = False
            self._set_buttons_state(True)

    def _run_train(self):
        if run_experiment is None:
            messagebox.showerror("Error", "No se pudo importar run_experiment del módulo train.")
            return
        if self.running_task:
            return
        self.running_task = True
        self._set_buttons_state(False)
        self._log("Iniciando entrenamiento (baseline)...")
        t = threading.Thread(target=self._task_train)
        t.start()

    def _task_train(self):
        try:
            # Ejecuta solo experimento baseline (evita segundo experimento para rapidez GUI)
            model, history, test_metrics, artifacts = run_experiment(LEARNING_RATE, NUM_EPOCHS, title_suffix="Baseline")
            torch.save(model.state_dict(), MODEL_PATH)
            self.model = model
            self._log("Entrenamiento finalizado. Modelo guardado en {}".format(MODEL_PATH))
            self._log("Métricas test: Acc={accuracy:.4f} Prec={precision:.4f} Recall={recall:.4f} F1={f1:.4f}".format(**test_metrics))
            self._log(f"Artifacts: curvas={artifacts['curves_png']} cm={artifacts['cm_png']} json={artifacts['metrics_json']}")
        except Exception as e:
            self._log("Error en entrenamiento: {}".format(e))
            traceback.print_exc()
        finally:
            self.running_task = False
            self._set_buttons_state(True)

    def _load_model(self):
        if not MODEL_PATH.exists():
            messagebox.showwarning("Aviso", "No existe archivo de modelo: {}".format(MODEL_PATH))
            return
        if CNN is None:
            messagebox.showerror("Error", "No se pudo importar CNN.")
            return
        try:
            model = CNN(apply_sigmoid=True).to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self._log("Modelo cargado correctamente para inferencia.")
        except Exception as e:
            self._log("Error al cargar modelo: {}".format(e))
            traceback.print_exc()

    def _predict_image(self):
        if self.model is None:
            messagebox.showwarning("Modelo", "Primero carga o entrena el modelo.")
            return
        file_path = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if not file_path:
            return
        try:
            img = Image.open(file_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            arr = np.array(img, dtype=np.float32) / 255.0
            # [H,W,C] -> tensor [1,C,H,W]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                prob = self.model(tensor).cpu().numpy().ravel()[0]
            label = "Dog" if prob >= 0.5 else "Cat"
            self._log(f"Predicción: {label} (prob={prob:.4f}) archivo={os.path.basename(file_path)}")
            messagebox.showinfo("Resultado", f"Etiqueta: {label}\nProbabilidad Dog: {prob:.4f}")
        except Exception as e:
            self._log("Error en predicción: {}".format(e))
            traceback.print_exc()

    def _show_results(self):
        exp = self.exp_var.get()
        curves_png = METRICS_DIR / f"learning_curves_{exp}.png"
        cm_png = METRICS_DIR / f"confusion_matrix_{exp}.png"
        metrics_json = METRICS_DIR / f"metrics_{exp}.json"
        if not metrics_json.exists():
            self._log(f"No existe metrics JSON para experimento {exp}: {metrics_json}")
            messagebox.showwarning("Resultados", "Primero entrena o selecciona experimento correcto.")
            return
        try:
            with open(metrics_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            test = data['test_metrics']
            self._log(f"Resultados {exp}: Acc={test['accuracy']:.4f} Prec={test['precision']:.4f} Recall={test['recall']:.4f} F1={test['f1']:.4f}")
            imgs_to_show = []
            for p in [curves_png, cm_png]:
                if p.exists():
                    imgs_to_show.append(p)
            if not imgs_to_show:
                self._log("No se encontraron imágenes de resultados para mostrar.")
                return
            win = tk.Toplevel(self)
            win.title(f"Resultados - {exp}")
            from PIL import ImageTk
            for p in imgs_to_show:
                try:
                    img = Image.open(p)
                    tk_img = ImageTk.PhotoImage(img)
                    lbl = ttk.Label(win, image=tk_img)
                    lbl.image = tk_img
                    lbl.pack(padx=5, pady=5)
                except Exception as e:
                    self._log(f"Error mostrando {p}: {e}")
        except Exception as e:
            self._log(f"Error cargando métricas: {e}")
            traceback.print_exc()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()