import cv2
import numpy as np
import pickle
import os
import sys
import tkinter as tk
from tkinter import filedialog
import re
# import sympy

from sympy import symbols, Eq, solve, sympify, N

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


def fix_syntax_math(text):
    # ngubah x => menjadi * menggunakan regex
    text = re.sub(r'(\d)x(\d)', r'\1*\2', text)
    
    new_text = ""
    i = 0
    while i < len(text):
        char = text[i]
        if char.isalpha():
            if i > 0:
                prev_char = text[i-1]
                if prev_char.isdigit() or prev_char == ')':
                    new_text += "*"
            new_text += char
        else:
            new_text += char
        i += 1
    return new_text

def calculate_math(equation_str):

    try:
        raw_eq = equation_str.replace(" ", "").lower()
        processed_eq = fix_syntax_math(raw_eq) 

        if "=" not in processed_eq:
            expr = sympify(processed_eq)
            result = expr.evalf()
            if float(result).is_integer():
                return str(int(result))
            else:
                return f"{float(result):.4f}"

        else:
            lhs_str, rhs_str = processed_eq.split("=")
            lhs = sympify(lhs_str)
            rhs = sympify(rhs_str)
            equation = Eq(lhs, rhs)
            
            free_vars = list(equation.free_symbols)
            if not free_vars:
                return "Benar" if lhs == rhs else "Salah"
            
            target_var = free_vars[0]
            solution = solve(equation, target_var)
            
            if not solution: return "Tidak ada solusi"
            
            final_ans = solution[0]
            res_val = N(final_ans)
            
            if float(res_val).is_integer():
                ans_str = str(int(res_val))
            else:
                ans_str = f"{float(res_val):.2f}"
                
            return f"{target_var} = {ans_str}"

    except Exception as e:
        return "Error Syntax"


#IndieFlower-regular.ttf (font yang digunakan di model)
# MODEL_PATH = r"model\\model1\\IF-100.pkl"
# MODEL_PATH = r"model\\model1\\IF-200.pkl" 
MODEL_PATH = r"model\\model1\\IF-300.pkl" 
LABEL_MAP_PATH = r"model\\model1\\label_map.pkl"


#PatrickHand-regular.ttf (font yang digunakan di model)

# MODEL_PATH = r"model\\model2\\PH-100.pkl"
# MODEL_PATH = r"model\\model2\\PH-200.pkl" 
# MODEL_PATH = r"model\\model2\\PH-300.pkl" 
# LABEL_MAP_PATH = r"model\\model2\\label_map.pkl"


IMG_SIZE = 45

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print(f"[!] ERROR: File model tidak ditemukan di: {MODEL_PATH}")
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f: model = pickle.load(f)
        with open(LABEL_MAP_PATH, "rb") as f: label_map = pickle.load(f)
        inv_label_map = {v: k for k, v in label_map.items()}
        return model, inv_label_map
    except Exception as e:
        print(f"[!] Gagal load model: {e}")
        return None, None

def post_process_text(text):
    """Auto-Correct Typo"""
    corrections = {
        "sln": "sin", "s1n": "sin", "5in": "sin", "sjn": "sin",
        "c0s": "cos", "co5": "cos", "cas": "cos", "ccs": "cos",
        "t0n": "tan", "lan": "tan", "1an": "tan", "ton": "tan",
        "l0g": "log", "1og": "log", "iog": "log", "log": "log",
        "x": "x" 
    }
    for typo, correct in corrections.items():
        text = text.replace(typo, correct)
    return text

def process_segment(roi, model, inv_label_map):
    h, w = roi.shape
    scale = min((IMG_SIZE - 10) / w, (IMG_SIZE - 10) / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    off_x, off_y = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    img_flat = canvas.flatten() / 255.0
    input_vector = img_flat.reshape(-1, 1)
    idx, _ = model.predict(input_vector)
    return inv_label_map.get(int(idx[0]), "?")

def smart_segmentation(img_binary):
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w * h > 20) or (w > 8 and h > 1): 
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    
    merged_boxes = []
    skip_next = False
    for i in range(len(boxes)):
        if skip_next:
            skip_next = False
            continue
        x, y, w, h = boxes[i]
        if i < len(boxes) - 1:
            x2, y2, w2, h2 = boxes[i+1]
            center1, center2 = x + w/2, x2 + w2/2
            if abs(center1 - center2) < max(w, w2) * 0.5: 
                gap = y2 - (y + h)
                if gap < 30: 
                    min_x, min_y = min(x, x2), min(y, y2)
                    max_x, max_y = max(x+w, x2+w2), max(y+h, y2+h2)
                    merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
                    skip_next = True
                    continue
        merged_boxes.append((x, y, w, h))
    return merged_boxes

def solve_image_ocr(image_path, model, inv_label_map):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127: gray = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    boxes = smart_segmentation(thresh)
    full_text = ""
    for (x, y, w, h) in boxes:
        roi = thresh[y:y+h, x:x+w]
        char = process_segment(roi, model, inv_label_map)
        full_text += char
    return post_process_text(full_text)

def main():
    print("\n" + "="*40)
    print("   SMART MATH SOLVER (LITE)")
    print("="*40 + "\n")

    model, inv_label_map = load_resources()
    if model is None: 
        input("Tekan Enter untuk menutup...")
        return

    print("[+] System Ready.")
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    while True:
        # print("\n>>> Menunggu input gambar...")
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Pilih Gambar",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )

        if not file_path:
            print("[!] Batal. Keluar.")
            break

        try:
            # 1. OCR
            soal_teks = solve_image_ocr(file_path, model, inv_label_map)
            
            if not soal_teks:
                print("    [!] Gagal baca gambar.")
                continue

            jawaban = calculate_math(soal_teks)

            print("-" * 30)
            print(f"SOAL   : {soal_teks}")
            print(f"JAWAB  : {jawaban}")
            print("-" * 30)

        except Exception as e:
            print(f"[!] Error: {e}")

        lagi = input("Lagi? (y/n): ").lower()
        if lagi != 'y': break
    
    root.destroy()

if __name__ == "__main__":
    main()