import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt # <--- KITA PAKAI INI SEKARANG

# === 1. DEFINISI KELAS (WAJIB ADA) ===
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
        return np.argmax(A2, axis=0), np.max(A2, axis=0)

# === 2. FUNGSI PREPROCESSING ===
def preprocess_image(image_path, target_size=45):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Gambar tidak ditemukan.")
        return None, None

    # Inversi otomatis jika background putih
    if np.mean(img) > 127:
        print("   [Info] Background terang terdeteksi -> Inversi Warna.")
        img = cv2.bitwise_not(img)

    # Thresholding
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Resize
    img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Flatten
    img_flat = img_resized.flatten() / 255.0
    img_vector = img_flat.reshape(-1, 1)

    return img_vector, img_resized

# === 3. EKSEKUSI UTAMA ===
def main():
    if not os.path.exists("label_map.pkl"):
        print("FATAL ERROR: 'label_map.pkl' tidak ditemukan!")
        return

    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    print(f"Label Map Terload ({len(inv_label_map)} kelas).")

    # Load Models
    models = {}
    dataset_names = ["dataset100", "dataset200", "dataset300"]
    
    print("\nMeload Model...")
    for name in dataset_names:
        filename = f"model_{name}.pkl"
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[name] = pickle.load(f)
            print(f"   ✅ {name}: Siap.")
        else:
            print(f"   ❌ {name}: Tidak ditemukan.")
    
    if not models:
        print("Tidak ada model ditemukan.")
        return

    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n--- Pilih Gambar ---")
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )

        if not file_path:
            break

        print(f"File: {os.path.basename(file_path)}")
        input_vector, img_display = preprocess_image(file_path)
        if input_vector is None: continue

        # --- PREDIKSI ---
        print("-" * 50)
        print(f"{'MODEL':<15} | {'PREDIKSI':<10} | {'CONFIDENCE':<10}")
        print("-" * 50)

        results_title = "Hasil Prediksi:\n"

        for name, model in models.items():
            idx, confidence = model.predict(input_vector)
            idx = int(idx[0])
            conf_val = float(confidence[0]) * 100
            pred_char = inv_label_map.get(idx, "Unknown")
            
            print(f"{name:<15} | {pred_char:<10} | {conf_val:.2f}%")
            results_title += f"{name}: {pred_char} ({conf_val:.1f}%)\n"
        
        print("-" * 50)
        
        # --- TAMPILKAN GAMBAR DENGAN MATPLOTLIB (GANTI CV2) ---
        plt.figure(figsize=(4, 4))
        plt.imshow(img_display, cmap='gray')
        plt.title(results_title, fontsize=10)
        plt.axis('off')
        print("Tutup jendela gambar untuk lanjut...")
        plt.show() # Program akan berhenti di sini sampai kamu close gambarnya

        lagi = input("Coba lagi? (y/n): ")
        if lagi.lower() != 'y':
            break

if __name__ == "__main__":
    main()