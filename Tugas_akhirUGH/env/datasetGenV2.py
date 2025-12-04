import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib.pyplot as plt

# === 1. KONFIGURASI ===
FONT_DIR = "fonts"
OUTPUT_DIR = "dataset100"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

# Ukuran Output
IMG_SIZE = 45

# PARAMETER DATASET
SAMPLES_PER_FONT = 100
LIMIT_FONTS = 8 # Batasi hanya menggunakan 8 font pertama

# === 2. DEFINISI KARAKTER ===
ANGKA = "0123456789"
OPERATOR = "+-x/()="
HURUF_FUNGSI = "sincostanlog" # Ini akan dipecah menjadi karakter unik: a, c, g, i, l, n, o, s, t

# Mengambil karakter unik dari gabungan string
CHARS = sorted(list(set(ANGKA + OPERATOR + HURUF_FUNGSI)))
print(f"Total Kelas ({len(CHARS)}): {CHARS}")

FILE_NAME_MAP = {'/': 'div', '+': 'plus', '-': 'minus', '(': 'buka', ')': 'tutup', '=': 'eq'}

def get_safe_name(char):
    return FILE_NAME_MAP.get(char, char)

# === 3. FUNGSI AUGMENTASI (GAYA HURUF) ===
def augment_style(img_pil):
    """
    Fokus mengubah GAYA (Tebal/Tipis) tanpa memutar/menggeser.
    Geser/Zoom akan dilakukan saat slicing.
    """
    img = np.array(img_pil)
    
    # 1. HAPUS ROTASI (Agar tegak lurus)
    
    # 2. VARIASI KETEBALAN (Erosion/Dilation)
    # Ini mensimulasikan font weight (Bold/Light)
    rand_thick = random.random()
    kernel = np.ones((2, 2), np.uint8)

    if rand_thick < 0.3:
        # Menebalkan huruf
        img = cv2.dilate(img, kernel, iterations=1)
    elif rand_thick > 0.8:
        # Menipiskan huruf
        img = cv2.erode(img, kernel, iterations=1)

    return img

# === 4. FUNGSI RENDER & SLICING ===
# === 4. FUNGSI RENDER & SLICING (DIPERBAIKI) ===
def generate_char_image(char, font_path):
    font_size = 80
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        return None

    # Gunakan kanvas besar agar aman
    canvas_size = 200 
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Render di tengah kanvas
    x = (canvas_size - text_w) // 2 - bbox[0]
    y = (canvas_size - text_h) // 2 - bbox[1]
    
    draw.text((x, y), char, font=font, fill=255)
    
    # Terapkan Augmentasi Gaya (Tebal/Tipis)
    img_aug = augment_style(img)

    # Deteksi area teks (Tight Bounding Box Asli)
    coords = cv2.findNonZero(img_aug)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # === PERBAIKAN: SAFE RANDOM PADDING ===
    # Alih-alih menggeser kotak 'x' yang bisa memotong huruf,
    # Kita tambahkan padding acak di LUAR huruf.
    # Ini mensimulasikan geseran (shift) dan zoom tanpa memotong pixel huruf.
    
    # Random padding (0 sampai 8 pixel)
    # Kalau pad_left besar & pad_right kecil --> Efek geser kanan
    pad_left = random.randint(0, 8)
    pad_right = random.randint(0, 8)
    pad_top = random.randint(0, 8)
    pad_bottom = random.randint(0, 8)

    # Hitung koordinat crop baru dengan padding
    # Gunakan max/min agar tidak keluar dari kanvas hitam
    crop_x1 = max(0, x - pad_left)
    crop_y1 = max(0, y - pad_top)
    crop_x2 = min(canvas_size, x + w + pad_right)
    crop_y2 = min(canvas_size, y + h + pad_bottom)

    # Lakukan Crop
    crop = img_aug[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Ambil ukuran crop baru
    crop_h, crop_w = crop.shape

    # Hindari error jika crop gagal (kosong)
    if crop_w == 0 or crop_h == 0: return None

    # Resize ke ukuran final (45x45)
    # Kita beri sedikit margin statis (misal 2px) agar tidak nempel dinding banget
    target_size = IMG_SIZE
    margin = 2 
    
    scale = min((target_size - margin*2) / crop_w, (target_size - margin*2) / crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    
    if new_w <= 0: new_w = 1
    if new_h <= 0: new_h = 1

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Tempel ke background hitam ukuran 45x45 (Centering)
    final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    off_x = (IMG_SIZE - new_w) // 2
    off_y = (IMG_SIZE - new_h) // 2
    
    # Pastikan koordinat penempelan valid
    end_y = min(IMG_SIZE, off_y + new_h)
    end_x = min(IMG_SIZE, off_x + new_w)
    src_h = end_y - off_y
    src_w = end_x - off_x
    
    final_img[off_y:end_y, off_x:end_x] = resized[:src_h, :src_w]

    return final_img

# === 5. EKSEKUSI GENERATOR ===
if not os.path.exists(FONT_DIR):
    print(f"ERROR: Folder '{FONT_DIR}' tidak ditemukan!")
else:
    # Ambil semua font
    all_fonts = [
        os.path.join(FONT_DIR, f)
        for f in os.listdir(FONT_DIR)
        if f.endswith(('.ttf', '.otf'))
    ]

    if not all_fonts:
        print("ERROR: Tidak ada font di dalam folder!")
    else:
        # Batasi jumlah font sesuai request (8 font)
        font_files = all_fonts[:LIMIT_FONTS]
        
        os.makedirs(IMAGES_DIR, exist_ok=True)
        dataset_data = []
        num_fonts = len(font_files)

        TOTAL_SAMPLES = SAMPLES_PER_FONT * len(CHARS) * num_fonts
        print(f"Menggunakan {num_fonts} font.")
        print(f"Target: {SAMPLES_PER_FONT} gambar per char per font.")
        print(f"Total estimasi file: {TOTAL_SAMPLES}...")

        file_counter = 0

        # Loop karakter
        for char in tqdm(CHARS, desc="Processing Characters"):
            # Loop font
            for font_idx, font_path in enumerate(font_files):
                # Loop sample (300 variasi)
                for sample_num in range(SAMPLES_PER_FONT):

                    img = generate_char_image(char, font_path)
                    if img is None:
                        continue

                    font_name = os.path.basename(font_path).split('.')[0]
                    filename = f"{file_counter:06d}_{font_idx:02d}_{get_safe_name(char)}.png"
                    save_path = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(save_path, img)

                    dataset_data.append([os.path.join("images", filename), char])
                    file_counter += 1

        df = pd.DataFrame(dataset_data, columns=['filename', 'label'])
        df.to_csv(CSV_PATH, index=False)

        print(f"\nSelesai! Dataset tersimpan di '{OUTPUT_DIR}' dengan {len(df)} sampel.")

        print("Contoh 10 Gambar Hasil Generate:")
        plt.figure(figsize=(15, 2))

        for k in range(10):
            if len(df) > 0:
                idx = random.randint(0, len(df)-1)
                path = os.path.join(OUTPUT_DIR, df.iloc[idx]['filename'])
                label = df.iloc[idx]['label']

                img = cv2.imread(path, 0)
                if img is not None:
                    plt.subplot(1, 10, k+1)
                    plt.imshow(img, cmap='gray')
                    plt.title(label)
                    plt.axis('off')

        plt.show()