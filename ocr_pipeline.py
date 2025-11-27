import os
import cv2
import pickle
import numpy as np
import time
import re
from collections import Counter
from skimage.feature import hog
import joblib 
from tensorflow.keras.models import load_model 
import streamlit as st

# === KONFIGURASI MODEL ===
MODEL_A_PATH = 'model_a_knn_manual.pkl'
MODEL_B_PATH = 'model_b_svm_klasik.pkl'
MODEL_C_PATH = 'model_c_cnn_dl_10e.h5'
MODEL_D_PATH = 'model_c_cnn_dl_30e.h5'
LABEL_MAP_PATH = 'label_map_ta.pkl'

# Parameter HOG/Model
K_NEAREST = 3 
IMG_SIZE = 45
IMG_CHANNELS = 1 
HOG_ORIENTERS = 9
HOG_PIXELS_PER_CELL = (6, 6) 
HOG_CELLS_PER_BLOCK = (2, 2)

# Aturan Regex
ALL_RULES = [
    (r"1og", "log", "Fungsi l->1, o->0"), (r"l0g", "log", "Fungsi o->0"),
    (r"1og", "log", "Fungsi l->1"), (r"00g", "cos", "Fungsi c->0, o->0, s->g"),
    (r"c0s", "cos", "Fungsi o->0"), (r"co5", "cos", "Fungsi s->5"),
    (r"cog", "cos", "Fungsi s->g"), (r"0os", "cos", "Fungsi c->0"),
    (r"00s", "cos", "Fungsi c->0, o->0"), (r"5in", "sin", "Fungsi s->5"),
    (r"sln", "sin", "Fungsi i->l"), (r"gin", "sin", "Fungsi s->g"),
    (r"si9", "sin", "Fungsi n->9"), (r"gi9", "sin", "Fungsi s->g, n->9"),
    (r"s1n", "sin", "Fungsi i->1"), (r"g1n", "sin", "Fungsi s->g, i->1"),
    (r"g19", "sin", "Fungsi s->g, i->1, n->9"), (r"t4n", "tan", "Fungsi a->4"),
    (r"1n", "ln", "Fungsi l->1"),
    (r"÷g", "+3", "Kombinasi ÷g -> +3"), (r"tg", "+3", "Kombinasi tg -> +3"),
    (r"t3", "+3", "Kombinasi t3 -> +3"), (r"÷3", "+3", "Kombinasi ÷3 -> +3"),
    (r"(\d)t(\d)", r"\1+\2", "Operator t->+"), (r"(\d)x(\d)", r"\1*\2", "Operator x->*"),
]
ALL_RULES_SORTED = sorted(ALL_RULES, key=lambda x: len(x[0]), reverse=True)

# === LOAD MODELS ===
def load_all_models(st_cache_resource):
    @st_cache_resource
    def load_cached():
        models = {}
        try:
            with open(LABEL_MAP_PATH, 'rb') as f: label_map = pickle.load(f)
            with open(MODEL_A_PATH, 'rb') as f:
                data = pickle.load(f)
                ref_features = data['features']
                ref_labels = data['labels']
            models['A'] = 'Manual-kNN'
            models['B'] = joblib.load(MODEL_B_PATH)
            models['C'] = load_model(MODEL_C_PATH, compile=False)
            models['D'] = load_model(MODEL_D_PATH, compile=False)
            return models, label_map, ref_features, ref_labels
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None, None
    return load_cached()

# === FEATURE PROCESSING ===
def extract_single_hog(img):
    return hog(img, orientations=HOG_ORIENTERS, pixels_per_cell=HOG_PIXELS_PER_CELL,
               cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys')

def predict_knn_manual(new_feat, ref_features, ref_labels, k):
    distances = np.linalg.norm(ref_features - new_feat, axis=1)
    idx = np.argsort(distances)[:k]
    vote = Counter([ref_labels[i] for i in idx])
    return vote.most_common(1)[0][0]

def preprocess_roi(roi):
    if len(roi.shape) > 2: roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_thresh = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    crop = roi_thresh[y:y+h, x:x+w]
    longest = max(w, h)
    canvas = np.zeros((longest+20, longest+20), dtype=np.uint8)
    ox, oy = (canvas.shape[1] - w)//2, (canvas.shape[0] - h)//2
    canvas[oy:oy+h, ox:ox+w] = crop
    return 255 - cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def perbaiki_konteks(raw, rules=ALL_RULES_SORTED):
    for p, repl, _ in rules:
        if re.search(p, raw): raw = re.sub(p, repl, raw)
    return raw

# === HELPER: MERGE BOXES (Untuk '=', 'i', ':') ===
def merge_close_boxes(boxes, vertical_thresh=15, horizontal_thresh=5):
    """Menggabungkan kotak yang bertumpuk secara vertikal (seperti = atau i)."""
    if not boxes: return []
    
    # Sort berdasarkan X
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used: continue
        
        current = boxes[i]
        x1, y1, w1, h1 = current
        
        # Cek kandidat untuk di-merge (biasanya tepat setelahnya dalam urutan X)
        # Kita cari kotak yang punya X mirip tapi Y berbeda
        merged_flag = False
        
        for j in range(i + 1, len(boxes)):
            if j in used: continue
            
            next_b = boxes[j]
            x2, y2, w2, h2 = next_b
            
            # Cek kedekatan Horizontal (X) -> Mereka harus segaris vertikal
            center1 = x1 + w1/2
            center2 = x2 + w2/2
            dist_x = abs(center1 - center2)
            
            # Cek kedekatan Vertikal (Y) -> Mereka harus dekat atas-bawah
            # Jarak antara bagian bawah kotak atas dan bagian atas kotak bawah
            y_gap = min(abs(y1 + h1 - y2), abs(y2 + h2 - y1))
            
            if dist_x < max(w1, w2) * 0.5 and y_gap < vertical_thresh:
                # MERGE!
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1+w1, x2+w2) - new_x
                new_h = max(y1+h1, y2+h2) - new_y
                
                merged.append((new_x, new_y, new_w, new_h))
                used.add(j)
                merged_flag = True
                break # Hanya merge 2 kotak (cukup untuk = dan i)
        
        if not merged_flag:
            merged.append(current)
            
    return sorted(merged, key=lambda b: b[0])


# === MAIN PIPELINE (DINAMIS) ===
def run_prediction_pipeline(image_full, models, label_map, ref_features, ref_labels, 
                            # Parameter Dinamis dari Slider
                            min_area=50, 
                            dilation_iter=1,
                            merge_vertical=True):
    
    start_total = time.time()
    gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
    (_, bin_img) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Morphology
    # Kernel fix 2x2 untuk opening (buang noise kecil)
    open_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    
    # Kernel 4x4 untuk dilation (bisa diulang dengan 'iterations' dari slider)
    kernel_dil = np.ones((4, 4), np.uint8) 
    img_final = cv2.dilate(open_img, kernel_dil, iterations=dilation_iter)

    # 2. Contours
    contours, _ = cv2.findContours(img_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter Area (Dinamis)
    valid_boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            valid_boxes.append(cv2.boundingRect(c))

    # 3. Merge Logic (Opsional/Dinamis)
    if merge_vertical:
        final_boxes = merge_close_boxes(valid_boxes)
    else:
        final_boxes = sorted(valid_boxes, key=lambda b: b[0])

    if not final_boxes:
        return {"error": "Tidak ada simbol."}, None, None

    # 4. Process ROIs & Predict
    image_visual = image_full.copy()
    rois, cnn_list = [], []

    for (x,y,w,h) in final_boxes:
        cv2.rectangle(image_visual, (x,y), (x+w,y+h), (0,255,0), 2)
        pad=5
        roi = gray[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad] # Ambil dari Gray Asli
        proc = preprocess_roi(roi)
        rois.append(proc)
        cnn_list.append(proc.astype(np.float32)/255.0)

    X_hog = np.array([extract_single_hog(r) for r in rois])
    X_cnn = np.array(cnn_list).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

    results = {}
    def run_model(name, instance, X):
        t0 = time.time()
        if name == 'A':
            raw = [predict_knn_manual(f, ref_features, ref_labels, K_NEAREST) for f in X]
        elif name in ['C','D']:
            raw = np.argmax(instance.predict(X, verbose=0), axis=1)
        else: 
            raw = instance.predict(X)
        
        txt = "".join([label_map[i] for i in raw])
        return perbaiki_konteks(txt), time.time()-t0

    results['A'] = {'result': run_model('A', None, X_hog)[0], 'time': run_model('A', None, X_hog)[1]}
    results['B'] = {'result': run_model('B', models['B'], X_hog)[0], 'time': run_model('B', models['B'], X_hog)[1]}
    results['C'] = {'result': run_model('C', models['C'], X_cnn)[0], 'time': run_model('C', models['C'], X_cnn)[1]}
    results['D'] = {'result': run_model('D', models['D'], X_cnn)[0], 'time': run_model('D', models['D'], X_cnn)[1]}

    return results, time.time()-start_total, image_visual