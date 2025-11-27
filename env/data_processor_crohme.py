import os
import cv2
import csv
import time

# === KONFIGURASI PROYEK ===
# GANTI PATH INI: Ini adalah folder yang berisi sub-folder simbol Anda.
CROHME_ROOT_DIR = 'data/extracted_images' 

# Output akhir proyek TA Anda
OUTPUT_DIR = "dataset_ta_final"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "char_labels_ta.csv")

# Ukuran standar yang sudah disepakati
IMG_SIZE = 45 

# === FUNGSI UTAMA PROSES DATA ===

def main():
    print("=" * 60)
    print("LANGKAH 1 TA: DATA INGEST DARI CROHME (SCAN FOLDER)")
    print("Membaca data berdasarkan nama folder (Label)")
    print("=" * 60)
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    total_samples = 0
    total_skipped = 0

    print(f"[1] Memindai simbol dari folder: {CROHME_ROOT_DIR}")

    with open(CSV_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        
        start_time = time.time()
        
        # Iterasi melalui setiap sub-folder di CROHME_ROOT_DIR
        for label_name in os.listdir(CROHME_ROOT_DIR):
            
            # Label adalah nama folder (misal: 'sin', '3', 'times')
            label_dir = os.path.join(CROHME_ROOT_DIR, label_name)
            
            if not os.path.isdir(label_dir):
                continue # Lewati jika itu bukan folder
                
            print(f"  Memproses simbol: '{label_name}'")
            
            # Iterasi melalui setiap gambar di folder simbol
            for index, filename in enumerate(os.listdir(label_dir)):
                
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue # Hanya proses file gambar
                    
                full_img_path = os.path.join(label_dir, filename)
                
                # Membaca gambar
                img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    total_skipped += 1
                    continue
                
                # Pastikan ukuran konsisten (jika ada yang berbeda)
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # --- Inversi (PENTING untuk HOG/ML) ---
                # Asumsi data CROHME adalah Teks Hitam di Latar Putih (invert)
                # Standar ML adalah Teks Putih di Latar Hitam.
                img_inverted = cv2.bitwise_not(img)
                
                # Menyimpan ke folder final proyek
                new_filename = f"{label_name}_{total_samples:06d}.png"
                save_path = os.path.join(IMAGES_DIR, new_filename)
                cv2.imwrite(save_path, img_inverted)
                
                # Menulis ke CSV final proyek
                writer.writerow([os.path.join("images", new_filename), label_name])
                total_samples += 1
                
        end_time = time.time()
        
    print(f"\n[2] Proses Selesai.")
    print(f"Total Simbol Disimpan: {total_samples}")
    print(f"Total File Dilewati/Rusak: {total_skipped}")
    print(f"Waktu pemrosesan: {end_time - start_time:.2f} detik.")
    
    print("=" * 60)
    print(f"Dataset final TA siap di: {OUTPUT_DIR}")
    print("Selanjutnya: Melatih Model A (k-NN Manual).")
    print("=" * 60)

if __name__ == "__main__":
    main()