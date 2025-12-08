import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
import pickle  

IMG_SIZE = 45
INPUT_NEURONS = IMG_SIZE * IMG_SIZE  
HIDDEN_NEURONS = 128
OUTPUT_NEURONS = 26 

DATASETS = ["dataset100", "dataset200", "dataset300"] #langsung train 3 model untuk 3 dataset

class NeuralNetworkManual:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        np.random.seed(42)
        self.W1 = np.random.randn(self.hidden_size, self.input_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((self.hidden_size, 1))
        self.W2 = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((self.output_size, 1))

    def sigmoid(self, z): 
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a): return a * (1 - a)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.W2.T, dZ2) * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train_with_history(self, X, y_indices, epochs=1000):
        m = X.shape[1]
        Y_one_hot = np.zeros((self.output_size, m))
        for i, label_idx in enumerate(y_indices):
            Y_one_hot[label_idx, i] = 1
        
        loss_history = []
        print(f"   Mulai training {epochs} epochs...")
        
        for i in range(epochs):
            predictions = self.forward(X)
            self.backward(X, Y_one_hot)
            
            if i % 100 == 0:
                loss = -np.mean(np.sum(Y_one_hot * np.log(predictions + 1e-9), axis=0))
                loss_history.append(loss)
        
        return loss_history
    
    def save_weights(self, filename):
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"   Model disimpan ke {filename}")

def load_data(folder_path, csv_filename="labels.csv"):
    images = []
    labels_raw = []
    
    csv_path = os.path.join(folder_path, csv_filename)
    images_folder = os.path.join(folder_path, "images")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} tidak ditemukan.")
        return None, None

    print(f"Loading data dari {folder_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if rows[0][0] == "filename": rows = rows[1:]
        
        for row in rows:
            img_filename = row[0]
            if "images" in img_filename:
                full_path = os.path.join(folder_path, os.path.basename(img_filename))
                full_path = os.path.join(images_folder, os.path.basename(img_filename))
            else:
                full_path = os.path.join(images_folder, img_filename)

            label = row[1]
            
            img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img.flatten() / 255.0)
                labels_raw.append(label)

    if len(images) == 0:
        return None, None

    X = np.array(images).T
    return X, labels_raw


plt.figure(figsize=(12, 7))
results = {}

_, sample_labels = load_data("dataset300") 
if sample_labels:
    unique_labels = sorted(list(set(sample_labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    with open("label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    print(f"Label Map disimpan ke 'label_map.pkl'")
else:
    print("Gagal load dataset300 untuk label map.")
    exit()

for ds_name in DATASETS:
    print(f"\n--- Memproses {ds_name} ---")
    
    X, y_raw = load_data(ds_name)
    if X is None: continue
    
    y_indices = [label_map[lbl] for lbl in y_raw]
    print(f"Jumlah Data: {X.shape[1]}")

    nn = NeuralNetworkManual(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, learning_rate=0.05)
    
    history = nn.train_with_history(X, y_indices, epochs=1500)
    results[ds_name] = history
    
    filename = f"model_{ds_name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(nn, f)
    print(f"Saved Model: '{filename}'")
    
    plt.plot(history, label=f"Model ({ds_name}) - Final Loss: {history[-1]:.4f}", linewidth=2)

plt.title(f"Perbandingan Training Loss (Disimpan ke File .pkl)")
plt.xlabel("Epoch (x100)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()