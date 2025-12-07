import cv2
import numpy as np
import pickle
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

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


MODEL_PATH = "model\\model_dataset200.pkl" 
LABEL_MAP_PATH = "model\\label_map.pkl"
IMG_SIZE = 45

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("ERROR: Model atau Label Map tidak ditemukan!")
        return None, None
    
    print(f"Loading {MODEL_PATH}...")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    with open(LABEL_MAP_PATH, "rb") as f:
        label_map = pickle.load(f)
    
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, inv_label_map

def process_segment(roi, model, inv_label_map):
    """
    Menerima potongan gambar (ROI), memformat ke 45x45, dan memprediksi.
    """
    h, w = roi.shape

    target_size = IMG_SIZE
    scale = min((target_size - 10) / w, (target_size - 10) / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w < 1: new_w = 1
    if new_h < 1: new_h = 1
    
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    off_x = (target_size - new_w) // 2
    off_y = (target_size - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
    
    img_flat = canvas.flatten() / 255.0
    input_vector = img_flat.reshape(-1, 1)
    
    idx, conf = model.predict(input_vector)
    char = inv_label_map.get(int(idx[0]), "?")
    return char, canvas

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
            
            center1 = x + w/2
            center2 = x2 + w2/2
            dist_x = abs(center1 - center2)
            

            if dist_x < max(w, w2) * 0.5: 
                gap = y2 - (y + h) 
                
                if gap < 30:
                    min_x = min(x, x2)
                    min_y = min(y, y2)
                    max_x = max(x+w, x2+w2)
                    max_y = max(y+h, y2+h2)
                    
                    merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))
                    skip_next = True
                    continue
        
        merged_boxes.append((x, y, w, h))
        
    return merged_boxes

def solve_image(image_path, model, inv_label_map):
    img_original = cv2.imread(image_path)
    if img_original is None: return
    
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    if np.mean(gray) > 127:
        gray = cv2.bitwise_not(gray)
        
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    boxes = smart_segmentation(thresh)
    
    full_text = ""
    annotated_img = img_original.copy()
    
    print("-" * 30)
    print(f"Terdeteksi {len(boxes)} karakter/simbol.")
    
    for (x, y, w, h) in boxes:
        roi = thresh[y:y+h, x:x+w]
        
        char, processed_img = process_segment(roi, model, inv_label_map)
        full_text += char

        color = (0, 255, 0)
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated_img, char, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        print(f"Posisi: {x},{y} | Prediksi: {char}")

    return full_text, annotated_img, thresh

def main():
    model, inv_label_map = load_resources()
    if model is None: return

    root = tk.Tk()
    root.withdraw()

    while True:
        print("\n--- PILIH GAMBAR RUMUS ---")
        file_path = filedialog.askopenfilename(
            title="Pilih Gambar Rumus Matematika",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if not file_path: break
        
        print(f"Memproses: {os.path.basename(file_path)}")
        

        hasil_teks, gambar_hasil, gambar_biner = solve_image(file_path, model, inv_label_map)
        
        print(f"\n>> HASIL PEMBACAAN: {hasil_teks}")

        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Input (Binary)")
        plt.imshow(gambar_biner, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Hasil: {hasil_teks}")
        plt.imshow(cv2.cvtColor(gambar_hasil, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()
        
        lagi = input("Coba gambar lain? (y/n): ")
        if lagi.lower() != 'y': break

if __name__ == "__main__":
    main()
