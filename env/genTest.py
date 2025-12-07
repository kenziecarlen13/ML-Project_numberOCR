import os
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# === 1. KONFIGURASI KHUSUS DATA TES ===
FONT_DIR = "fonts"
OUTPUT_DIR = "dataset_TEST"  # Folder terpisah untuk ujian
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

IMG_SIZE = 45

# Jumlah sampel SEDIKIT SAJA (cukup untuk ujian)
# 20 gambar per karakter x 26 karakter = 520 soal ujian
SAMPLES_PER_FONT = 20 
LIMIT_FONTS = 8 

# === 2. DEFINISI KARAKTER ===
ANGKA = "0123456789"
OPERATOR = "+-x/()="
HURUF_FUNGSI = "sincostanlog"
CHARS = sorted(list(set(ANGKA + OPERATOR + HURUF_FUNGSI)))
FILE_NAME_MAP = {'/': 'div', '+': 'plus', '-': 'minus', '(': 'buka', ')': 'tutup', '=': 'eq'}

def get_safe_name(char): return FILE_NAME_MAP.get(char, char)

# === 3. FUNGSI AUGMENTASI & RENDER (SAMA PERSIS) ===
def augment_style(img_pil):
    img = np.array(img_pil)
    rand_thick = random.random()
    kernel = np.ones((2, 2), np.uint8)
    if rand_thick < 0.3: img = cv2.dilate(img, kernel, iterations=1)
    return img

def generate_char_image(char, font_path):
    # ... (Logic generator yang sama dengan Safe Padding) ...
    # Agar adil, cara generate soal harus sama dengan cara belajar,
    # bedanya hanya di nilai random padding-nya nanti.
    
    font_size = 80
    try: font = ImageFont.truetype(font_path, font_size)
    except: return None

    canvas_size = 200 
    img = Image.new('L', (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size - text_w) // 2 - bbox[0]
    y = (canvas_size - text_h) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=255)
    
    img_aug = augment_style(img)
    coords = cv2.findNonZero(img_aug)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)

    # RANDOM PADDING (Ini yang membuat data tes BEDA dengan data latih)
    pad_left = random.randint(0, 8)
    pad_right = random.randint(0, 8)
    pad_top = random.randint(0, 8)
    pad_bottom = random.randint(0, 8)

    crop_x1 = max(0, x - pad_left)
    crop_y1 = max(0, y - pad_top)
    crop_x2 = min(canvas_size, x + w + pad_right)
    crop_y2 = min(canvas_size, y + h + pad_bottom)

    crop = img_aug[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_h, crop_w = crop.shape
    if crop_w == 0 or crop_h == 0: return None

    target_size = IMG_SIZE
    margin = 2 
    scale = min((target_size - margin*2) / crop_w, (target_size - margin*2) / crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    if new_w <= 0: new_w = 1
    if new_h <= 0: new_h = 1

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    final_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    off_x = (IMG_SIZE - new_w) // 2
    off_y = (IMG_SIZE - new_h) // 2
    
    end_y = min(IMG_SIZE, off_y + new_h)
    end_x = min(IMG_SIZE, off_x + new_w)
    src_h = end_y - off_y
    src_w = end_x - off_x
    final_img[off_y:end_y, off_x:end_x] = resized[:src_h, :src_w]

    return final_img

# === 4. EKSEKUSI ===
if not os.path.exists(FONT_DIR):
    print(f"ERROR: Folder '{FONT_DIR}' tidak ditemukan!")
else:
    all_fonts = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]
    if not all_fonts:
        print("ERROR: Tidak ada font!")
    else:
        font_files = all_fonts[:LIMIT_FONTS]
        os.makedirs(IMAGES_DIR, exist_ok=True)
        dataset_data = []
        
        print(f"Membuat DATA TES (soal ujian) di '{OUTPUT_DIR}'...")
        file_counter = 0

        for char in tqdm(CHARS, desc="Membuat Soal"):
            for font_idx, font_path in enumerate(font_files):
                for sample_num in range(SAMPLES_PER_FONT):
                    img = generate_char_image(char, font_path)
                    if img is None: continue

                    filename = f"test_{file_counter:05d}_{get_safe_name(char)}.png"
                    save_path = os.path.join(IMAGES_DIR, filename)
                    cv2.imwrite(save_path, img)

                    # Simpan path relatif "images/namafile.png" agar konsisten
                    dataset_data.append([os.path.join("images", filename), char])
                    file_counter += 1

        df = pd.DataFrame(dataset_data, columns=['filename', 'label'])
        df.to_csv(CSV_PATH, index=False)
        print(f"Selesai! {len(df)} soal ujian siap dikerjakan AI.")