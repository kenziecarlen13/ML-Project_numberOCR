import numpy as np
import pickle
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === KONFIGURASI ===
# Path ke data yang sudah disiapkan di Langkah 2 TA
DATASET_PATH = "dataset_ta_final"
MODEL_A_PATH = 'model_a_knn_manual.pkl'
MODEL_OUTPUT_PATH = 'model_b_svm_klasik.pkl'

# === FUNGSI UTAMA PELATIHAN ===
def main():
    print("=" * 60)
    print("TAHAP 3: MELATIH MODEL B - KLASIK SVM")
    print("=" * 60)

    # 1. Muat Data HOG (Fitur) dari file Model A
    print("[1] Memuat data HOG...")
    try:
        with open(MODEL_A_PATH, 'rb') as f:
            reference_data = pickle.load(f)
        X_hog = reference_data['features']
        y_encoded = reference_data['labels']
        
    except FileNotFoundError:
        print(f"Error: File {MODEL_A_PATH} tidak ditemukan.")
        print("Pastikan Langkah 2 TA (pembuatan file PKL) sudah selesai.")
        return

    print(f"Dimensi Data Latih: {X_hog.shape}")
    
    # 2. Pembagian Data (Training dan Testing)
    # Kita bagi data menjadi 90% Latih, 10% Uji
    print("[2] Membagi data menjadi Training (90%) dan Testing (10%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_hog, y_encoded, test_size=0.10, random_state=42
    )
    
    # 3. Inisialisasi dan Latih Model SVM
    # Pengaturan C=1.0 (regularisasi) dan kernel='rbf' (kernel default yang sangat baik)
    print("[3] Memulai pelatihan Support Vector Machine (SVM). Ini akan memakan waktu.")
    
    start_time = time.time()
    # Gunakan 'linear' untuk kecepatan jika 'rbf' terlalu lambat
    model = SVC(kernel='rbf', C=1.0, verbose=True) 
    model.fit(X_train, y_train)
    end_time = time.time()

    print(f"\nPelatihan SVM Selesai dalam {end_time - start_time:.2f} detik.")

    # 4. Evaluasi Model
    print("[4] Mengevaluasi model pada data testing...")
    y_pred = model.predict(X_test)
    
    # Hasil evaluasi akan menunjukkan betapa bagusnya Model B
    print("\nLaporan Klasifikasi Model B (SVM Klasik):")
    # Karena kita tidak punya nama kelas (string), kita tampilkan berdasarkan ID
    print(classification_report(y_test, y_pred))

    # 5. Simpan Model
    print(f"[5] Menyimpan Model B ke: {MODEL_OUTPUT_PATH}")
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print("=" * 60)
    print("🎉 Model B (SVM Klasik) berhasil dilatih dan disimpan.")
    print("Sekarang pindah ke Model C (Deep Learning).")
    print("=" * 60)

if __name__ == "__main__":
    main()