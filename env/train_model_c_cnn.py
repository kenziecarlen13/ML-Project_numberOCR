import numpy as np
import time
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === KONFIGURASI ===
# Path ke data yang sudah disiapkan di Langkah 2 TA
DATASET_PATH = "dataset_ta_final"
NPY_IMAGES_PATH = os.path.join(DATASET_PATH, 'dataset_images.npy')
NPY_LABELS_PATH = os.path.join(DATASET_PATH, 'dataset_labels.npy')
MODEL_OUTPUT_PATH = 'model_c_cnn_dl.h5'

# Parameter Model
IMG_SIZE = 45
IMG_CHANNELS = 1 # Grayscale
BATCH_SIZE = 128 # Meningkatkan batch size untuk efisiensi GPU, karena kita optimalkan RAM
EPOCHS = 10 

# === FUNGSI MEMBANGUN MODEL CNN ===
def build_cnn_model(input_shape, num_classes):
    """Membangun arsitektur CNN sederhana dan efisien."""
    model = Sequential([
        # Lapisan Konvolusi 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        # Lapisan Konvolusi 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        # Lapisan Konvolusi 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Lapisan Klasifikasi
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# === FUNGSI UTAMA PELATIHAN ===
def main():
    print("=" * 60)
    print("TAHAP 3: MELATIH MODEL C - DEEP LEARNING CNN (Memory Optimized)")
    print("Memverifikasi ketersediaan GPU...")
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU DITEMUKAN. Pelatihan akan menggunakan VRAM.")
    else:
        print("⚠️ GPU TIDAK DITEMUKAN. Pelatihan akan berjalan lambat di CPU.")
    print("=" * 60)

    # 1. Muat Data Mentah (.npy)
    # Ini adalah satu-satunya saat CPU RAM digunakan secara masif
    print("[1] Memuat data gambar mentah dan label...")
    try:
        X_raw = np.load(NPY_IMAGES_PATH)
        y_encoded = np.load(NPY_LABELS_PATH)
    except FileNotFoundError:
        print(f"Error: File NPY tidak ditemukan di {DATASET_PATH}.")
        print("Pastikan Langkah 2 TA (pembuatan file NPY) sudah selesai.")
        return

    # 2. Reshape dan Kategorisasi Data
    X_reshaped = X_raw.reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
    num_classes = len(np.unique(y_encoded))
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    print(f"Dimensi Data Input CNN: {X_reshaped.shape}")
    
    # 3. Pembagian Data (Training dan Testing)
    print("[2] Membagi data menjadi Training (90%) dan Testing (10%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_categorical, test_size=0.10, random_state=42
    )
    
    # --- KUNCI SOLUSI: Menggunakan tf.data.Dataset ---
    print("[3] Mengubah data ke format tf.data.Dataset untuk efisiensi RAM...")
    
    # Membuat dataset dari array NumPy (slicing)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Mengatur cache, shuffle, batch, dan prefetch untuk kinerja optimal
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.cache().shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"Data Latih: {X_train.shape[0]} sampel")
    print(f"Data Validasi: {X_test.shape[0]} sampel")
    print(f"Batch Size: {BATCH_SIZE}")

    # 4. Bangun Model
    model = build_cnn_model((IMG_SIZE, IMG_SIZE, IMG_CHANNELS), num_classes)
    model.summary()

    # 5. Latih Model (menggunakan dataset yang dioptimalkan)
    print("[5] Memulai pelatihan CNN. Ini akan stabil dan cepat.")
    
    start_time = time.time()
    history = model.fit(
        train_dataset, # MENGGUNAKAN DATASET YANG DIOPTIMALKAN
        epochs=EPOCHS,
        validation_data=test_dataset, # MENGGUNAKAN DATASET YANG DIOPTIMALKAN
        verbose=1
    )
    end_time = time.time()

    print(f"\nPelatihan CNN Selesai dalam {end_time - start_time:.2f} detik.")

    # 6. Simpan Model
    model.save(MODEL_OUTPUT_PATH)
    print(f"[6] Model C disimpan di: {MODEL_OUTPUT_PATH}")

    # 7. Evaluasi Final
    loss, acc = model.evaluate(test_dataset, verbose=0)
    print(f"\n[7] Akurasi Akhir Model C pada Data Tes: {acc*100:.2f}%")

    print("=" * 60)
    print("🎉 Tiga Model siap. Anda bisa mulai menyusun Aplikasi Web.")
    print("=" * 60)

if __name__ == "__main__":
    main()