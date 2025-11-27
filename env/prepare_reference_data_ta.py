import os
import cv2
import pickle
import numpy as np
from skimage.feature import hog
import time
from sklearn.preprocessing import LabelEncoder
import csv

# === KONFIGURASI MODEL A (k-NN Manual) ===
DATASET_PATH = "dataset_ta_final"
CSV_PATH = os.path.join(DATASET_PATH, 'char_labels_ta.csv') 
IMAGES_DIR = os.path.join(DATASET_PATH, 'images')

# Parameter HOG (Optimal dari eksperimen sebelumnya)
IMG_SIZE = 45
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (6, 6) 
HOG_CELLS_PER_BLOCK = (2, 2)

# Nama file output
MODEL_OUTPUT_PATH = 'model_a_knn_manual.pkl'
LABEL_MAP_PATH = 'label_map_ta.pkl'

# --- BARU: Output untuk Model B & C ---
NPY_IMAGES_PATH = os.path.join(DATASET_PATH, 'dataset_images.npy')
NPY_LABELS_PATH = os.path.join(DATASET_PATH, 'dataset_labels.npy')

# === FUNGSI INGEST DAN HOG ===

def load_and_process_data(csv_path, images_dir):
    """Memuat gambar, label, dan mengekstrak fitur HOG."""
    
    print(f"Membaca data dari: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File CSV tidak ditemukan di {csv_path}")
        return None, None, None

    # Memuat CSV (menggunakan csv module untuk keandalan)
    image_paths = []
    labels = []
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        for row in reader:
            if len(row) == 2:
                image_paths.append(row[0])
                labels.append(row[1])

    images_list = [] # Untuk menyimpan gambar mentah (dipakai Model C)
    hog_features = []
    total_files = len(labels)
    print(f"Total {total_files} file gambar akan diproses...")
    
    start_time_hog = time.time()

    for index, filename in enumerate(image_paths):
        
        img_path = os.path.join(DATASET_PATH, filename)
        
        if not os.path.exists(img_path): continue
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
            
        # Pastikan ukuran konsisten
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Simpan gambar mentah untuk Model C
        images_list.append(img_resized) 
        
        # Ekstraksi HOG untuk Model A dan B
        features = hog(img_resized, orientations=HOG_ORIENTATIONS,
                       pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK,
                       block_norm='L2-Hys',
                       visualize=False)
        hog_features.append(features)
        
        if (index + 1) % 50000 == 0: # Diperbarui: Laporan setiap 50.000 sampel
            print(f"  ... {index + 1} fitur diekstraksi.")

    end_time_hog = time.time()
    print(f"Ekstraksi HOG selesai dalam {end_time_hog - start_time_hog:.2f} detik.")
    
    return np.array(hog_features, dtype="float"), np.array(images_list, dtype="uint8"), labels

# === PROSES UTAMA PELATIHAN ===
def main():
    print("=" * 60)
    print("LANGKAH 2 TA: MEMBUAT MODEL A (k-NN) DAN FILE NPY")
    print("=" * 60)

    # 1. Muat data, ekstrak HOG, dan dapatkan gambar mentah
    X_hog_features, X_images_raw, labels_str = load_and_process_data(CSV_PATH, IMAGES_DIR)
    
    if X_hog_features is None or not labels_str:
        print("Gagal memuat data. Pelatihan dihentikan.")
        return

    # 2. Preprocessing Label
    le = LabelEncoder()
    y_labels_encoded = le.fit_transform(labels_str)
    
    # Buat kamus kebalikannya
    int_to_label = {i: label for i, label in enumerate(le.classes_)}
    
    print(f"Bentuk data HOG (Model A & B): {X_hog_features.shape}")
    print(f"Bentuk gambar mentah (Model C): {X_images_raw.shape}")

    # --- Bagian Output ---
    
    # 3. Simpan Model A (Database Referensi k-NN)
    print("\n[Output 1] Menyimpan Model A (Database k-NN) ke file...")
    
    reference_data = {
        'features': X_hog_features,
        'labels': y_labels_encoded.astype(np.int32)
    }
    
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(reference_data, f)
        
    print(f"  > Model A disimpan di: {MODEL_OUTPUT_PATH}")

    # 4. Simpan Data NPY (Untuk Model B dan C)
    print("\n[Output 2] Menyimpan data gambar mentah ke .npy untuk Model C...")
    # Normalisasi gambar mentah (0-255 -> 0.0-1.0)
    X_normalized = X_images_raw.astype("float32") / 255.0 
    np.save(NPY_IMAGES_PATH, X_normalized)
    np.save(NPY_LABELS_PATH, y_labels_encoded)
    
    print(f"  > Gambar disimpan di: {NPY_IMAGES_PATH}")
    print(f"  > Label disimpan di: {NPY_LABELS_PATH}")

    # 5. Simpan Pemetaan Label
    print("\n[Output 3] Menyimpan Pemetaan Label (Kamus)")
    with open(LABEL_MAP_PATH, 'wb') as f:
        pickle.dump(int_to_label, f)
        
    print(f"  > Penerjemah label disimpan di: {LABEL_MAP_PATH}")

    print("=" * 60)
    print("🎉 Dataset, Model A, dan file NPY untuk Model B/C telah siap.")
    print("=" * 60)

if __name__ == "__main__":
    main()