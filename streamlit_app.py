import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd 
from ocr_pipeline import load_all_models, run_prediction_pipeline, IMG_SIZE, IMG_CHANNELS, K_NEAREST

def cv2_to_pil(image_cv2):
    return Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

# === FUNGSI PREPROCESS YANG BISA DI-TUNING ===
def preprocess_roi_dynamic(roi, thickness=0, invert_input=False):
    """
    Memproses ROI dengan opsi Inversi Warna Dinamis.
    """
    if len(roi.shape) > 2: roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 1. Thresholding Awal
    (thresh, roi_thresh) = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 2. Penebalan (Thickness)
    if thickness > 0:
        kernel = np.ones((3,3), np.uint8)
        roi_thresh = cv2.dilate(roi_thresh, kernel, iterations=thickness)
    
    # 3. INVERSI MANUAL (Jika dicentang di Sidebar)
    # Ini kuncinya! Jika model dilatih dengan latar putih, kita harus balikkan ini.
    if invert_input:
        roi_thresh = 255 - roi_thresh

    # 4. Cropping & Resizing (Sama seperti sebelumnya)
    contours, _ = cv2.findContours(roi_thresh if not invert_input else (255-roi_thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    
    c = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    
    # Crop
    if invert_input:
        # Jika invert aktif, kita crop dari gambar yang sudah di-invert
        char_crop = roi_thresh[y:y+h, x:x+w]
    else:
        char_crop = roi_thresh[y:y+h, x:x+w]

    # Padding & Square Canvas
    sisi_terpanjang = max(w, h)
    padding_pixel = 8 
    canvas_size = sisi_terpanjang + (padding_pixel * 2)
    
    # Warna Canvas tergantung mode
    bg_color = 255 if invert_input else 0
    kanvas = np.full((canvas_size, canvas_size), bg_color, dtype=np.uint8)
    
    offset_x = (canvas_size - w) // 2
    offset_y = (canvas_size - h) // 2
    kanvas[offset_y:offset_y+h, offset_x:offset_x+w] = char_crop
    
    processed_img = cv2.resize(kanvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    
    return processed_img 

# Patching fungsi pipeline
import ocr_pipeline

def run_prediction_pipeline_debug(image_full, models, label_map, ref_features, ref_labels, 
                                   min_area=50, dilation_iter=1, merge_vertical=True, 
                                   char_thickness=0, invert_mode=False):
    
    start_total = time.time()
    gray = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)
    (_, bin_img) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Slicing Logic
    open_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    kernel_dil = np.ones((4, 4), np.uint8) 
    img_dilated = cv2.dilate(open_img, kernel_dil, iterations=dilation_iter)
    kernel_close = np.ones((3, 3), np.uint8)
    img_final_morph = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv2.findContours(img_final_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            valid_boxes.append(cv2.boundingRect(c))

    if merge_vertical:
        final_boxes = ocr_pipeline.merge_close_boxes(valid_boxes)
    else:
        final_boxes = sorted(valid_boxes, key=lambda b: b[0])

    if not final_boxes:
        return {"error": "Tidak ada simbol."}, None, None, []

    image_visual = image_full.copy()
    rois, cnn_list = [], []
    debug_images = [] 

    for (x,y,w,h) in final_boxes:
        cv2.rectangle(image_visual, (x,y), (x+w,y+h), (0,255,0), 2)
        pad=5
        y_start = max(0,y-pad); y_end = min(gray.shape[0], y+h+pad)
        x_start = max(0,x-pad); x_end = min(gray.shape[1], x+w+pad)
        roi = gray[y_start:y_end, x_start:x_end]
        
        # PANGGIL FUNGSI PREPROCESS DINAMIS
        proc = preprocess_roi_dynamic(roi, thickness=char_thickness, invert_input=invert_mode)
        
        rois.append(proc)
        # Normalisasi 0-1
        cnn_list.append(proc.astype(np.float32)/255.0)
        debug_images.append(proc) 

    X_hog = np.array([ocr_pipeline.extract_single_hog(r) for r in rois])
    X_cnn = np.array(cnn_list).reshape(-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

    results = {}
    def run_model(name, instance, X):
        t0 = time.time()
        if name == 'A':
            raw = [ocr_pipeline.predict_knn_manual(f, ref_features, ref_labels, K_NEAREST) for f in X]
        elif name in ['C','D']:
            raw = np.argmax(instance.predict(X, verbose=0), axis=1)
        else: 
            raw = instance.predict(X)
        
        txt = "".join([label_map[i] for i in raw])
        # Kembalikan Text Mentah DAN Text Final
        return txt, ocr_pipeline.perbaiki_konteks(txt), time.time()-t0

    # Run Models (Dapatkan Raw & Final)
    rA_raw, rA_fin, tA = run_model('A', None, X_hog)
    rB_raw, rB_fin, tB = run_model('B', models['B'], X_hog)
    rC_raw, rC_fin, tC = run_model('C', models['C'], X_cnn)
    rD_raw, rD_fin, tD = run_model('D', models['D'], X_cnn)

    results['A'] = {'raw': rA_raw, 'result': rA_fin, 'time': tA}
    results['B'] = {'raw': rB_raw, 'result': rB_fin, 'time': tB}
    results['C'] = {'raw': rC_raw, 'result': rC_fin, 'time': tC}
    results['D'] = {'raw': rD_raw, 'result': rD_fin, 'time': tD}

    return results, time.time()-start_total, image_visual, debug_images

import time 

def main_app():
    st.set_page_config(layout="wide", page_title="TA OCR Comparison")
    
    st.sidebar.header("⚙️ Tuning Parameter")
    
    st.sidebar.subheader("1. Slicing & Preprocessing")
    dilation_val = st.sidebar.slider("Gabung Karakter (Slicing)", 0, 5, 2)
    min_area_val = st.sidebar.slider("Filter Noise", 10, 500, 50)
    merge_vert = st.sidebar.checkbox("Gabung Vertikal (=, i)", value=True)
    char_thickness = st.sidebar.slider("Tebalkan Huruf", 0, 5, 2)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ FIX CNN ERROR")
    # CHECKBOX BARU INI KUNCINYA
    invert_mode = st.sidebar.checkbox("Balik Warna Input (Hitam <-> Putih)", value=False, 
                                      help="Centang ini jika CNN memprediksi ngawur. CNN sensitif terhadap latar belakang hitam vs putih.")

    st.title("Tugas Akhir: Perbandingan 4 Model OCR")

    models, label_map, ref_features, ref_labels = load_all_models(st.cache_resource)
    if models is None: st.stop()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("1. Input")
        uploaded_file = st.file_uploader("Upload Gambar", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            col1.image(image, caption="Asli", use_container_width=True)
            
            uploaded_file.seek(0)
            img_bytes = uploaded_file.read()
            np_arr = np.frombuffer(img_bytes, np.uint8)
            image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if st.button("Proses Analisis", type="primary"):
                with st.spinner("Memproses..."):
                    results, total_time, image_visual, debug_imgs = run_prediction_pipeline_debug(
                        image_cv2, models, label_map, ref_features, ref_labels,
                        min_area=min_area_val,
                        dilation_iter=dilation_val,
                        merge_vertical=merge_vert,
                        char_thickness=char_thickness,
                        invert_mode=invert_mode # Kirim parameter invert
                    )
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.session_state['results'] = results
                    st.session_state['total_time'] = total_time
                    st.session_state['visual_image'] = image_visual
                    st.session_state['debug_images'] = debug_imgs

    with col2:
        st.header("2. Hasil")
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            debug_imgs = st.session_state.get('debug_images', [])
            
            # TAMPILKAN GAMBAR DEBUG (PENTING UNTUK CEK WARNA)
            st.subheader("🔍 Apa yang dilihat Model?")
            st.caption("Pastikan gambar di bawah ini terlihat jelas (Teks Terang di Latar Gelap, atau sebaliknya). Jika CNN salah, coba centang 'Balik Warna Input' di kiri.")
            
            if debug_imgs:
                cols = st.columns(min(len(debug_imgs), 8))
                for i, img in enumerate(debug_imgs):
                    with cols[i % 8]:
                        st.image(img, width=50, clamp=True, channels='GRAY')

            st.markdown("---")
            
            # Tampilkan Hasil dengan Kolom "Mentah"
            df = pd.DataFrame({
                "Model": ["A (k-NN)", "B (SVM)", "C (10e)", "D (30e)"],
                "Mentah (Mata)": [results['A']['raw'], results['B']['raw'], results['C']['raw'], results['D']['raw']],
                "Final (+Regex)": [results['A']['result'], results['B']['result'], results['C']['result'], results['D']['result']],
                "Waktu": [f"{results['A']['time']:.3f}s", f"{results['B']['time']:.3f}s", f"{results['C']['time']:.3f}s", f"{results['D']['time']:.3f}s"]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main_app()