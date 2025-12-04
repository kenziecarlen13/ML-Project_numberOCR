import cv2
import numpy as np
import pickle
import os
import csv
import pandas as pd
from tqdm import tqdm

# === 1. DEFINISI KELAS AI (WAJIB ADA) ===
class NeuralNetworkManual:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1, self.b1, self.W2, self.b2 = None, None, None, None

    def sigmoid(self, z): 
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=0), np.max(A2, axis=0) # Return Index & Confidence

# === KONFIGURASI ===
TEST_FOLDER = "dataset_TEST"
IMG_SIZE = 45
OUTPUT_REPORT = "laporan_prediksi.csv"

def run_batch_prediction():
    # 1. Load Label Map
    if not os.path.exists("label_map.pkl"):
        print("ERROR: label_map.pkl tidak ditemukan!")
        return
    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    
    # Buat pembalik (Angka -> Karakter)
    inv_label_map = {v: k for k, v in label_map.items()}
    print(f"Kamus Label Terload: {len(label_map)} kelas")

    # 2. Load Models
    models = {}
    model_list = ["dataset100", "dataset200", "dataset300"]
    
    print("Loading Models...")
    for name in model_list:
        fname = f"model_{name}.pkl"
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                models[name] = pickle.load(f)
    
    if not models:
        print("Tidak ada model ditemukan.")
        return

    # 3. Baca Daftar Soal
    csv_path = os.path.join(TEST_FOLDER, "labels.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Tidak ada labels.csv di {TEST_FOLDER}")
        return

    # Siapkan List untuk Laporan
    report_data = []

    print(f"Mulai memproses batch testing dari '{TEST_FOLDER}'...")
    
    # Baca CSV Soal
    df_test = pd.read_csv(csv_path)
    
    # Loop setiap baris soal (Gunakan tqdm untuk loading bar)
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Processing"):
        fname = row['filename']
        label_asli = row['label'] # Karakter asli (misal '+', 'a')
        
        # Handle Path
        if "images" in fname:
             # Cek apakah path di csv sudah match dengan folder os
            full_path = os.path.join(TEST_FOLDER, os.path.basename(fname))
            full_path_check = os.path.join(TEST_FOLDER, "images", os.path.basename(fname))
            if os.path.exists(full_path_check): full_path = full_path_check
        else:
            full_path = os.path.join(TEST_FOLDER, "images", fname)
            
        # Preprocessing
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        if img.shape != (IMG_SIZE, IMG_SIZE):
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
        img_flat = img.flatten() / 255.0
        input_vector = img_flat.reshape(-1, 1)

        # --- PREDIKSI OLEH SEMUA MODEL ---
        row_result = {
            "Filename": fname,
            "Kunci Jawaban": label_asli
        }
        
        # Prediksi Model 100
        if "dataset100" in models:
            idx, conf = models["dataset100"].predict(input_vector)
            pred_char = inv_label_map[int(idx[0])]
            row_result["Pred_100"] = pred_char
            row_result["Conf_100"] = f"{float(conf[0])*100:.1f}%"
            row_result["Cek_100"] = "✅" if pred_char == str(label_asli) else "❌"

        # Prediksi Model 200
        if "dataset200" in models:
            idx, conf = models["dataset200"].predict(input_vector)
            pred_char = inv_label_map[int(idx[0])]
            row_result["Pred_200"] = pred_char
            row_result["Cek_200"] = "✅" if pred_char == str(label_asli) else "❌"

        # Prediksi Model 300
        if "dataset300" in models:
            idx, conf = models["dataset300"].predict(input_vector)
            pred_char = inv_label_map[int(idx[0])]
            row_result["Pred_300"] = pred_char
            row_result["Conf_300"] = f"{float(conf[0])*100:.1f}%"
            row_result["Cek_300"] = "✅" if pred_char == str(label_asli) else "❌"

        report_data.append(row_result)

    # 4. Simpan ke CSV
    df_report = pd.DataFrame(report_data)
    df_report.to_csv(OUTPUT_REPORT, index=False)
    
    print("\n" + "="*50)
    print(f"SELESAI! Laporan tersimpan di: {OUTPUT_REPORT}")
    print("="*50)
    
    # 5. Tampilkan Preview Singkat (Analisis Error)
    if "Cek_300" in df_report.columns:
        total = len(df_report)
        benar = df_report[df_report["Cek_300"] == "✅"].shape[0]
        salah = total - benar
        print(f"Ringkasan Model 300:")
        print(f"Total Soal: {total}")
        print(f"Benar     : {benar}")
        print(f"Salah     : {salah}")
        print(f"Akurasi   : {(benar/total)*100:.2f}%")
        
        if salah > 0:
            print("\nContoh 5 Kesalahan Model 300:")
            print(df_report[df_report["Cek_300"] == "❌"][["Kunci Jawaban", "Pred_300", "Conf_300"]].head(5))

if __name__ == "__main__":
    run_batch_prediction()