# Smart Math Solver & OCR (From Scratch)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Status](https://img.shields.io/badge/Status-Completed-green.svg)
![AI](https://img.shields.io/badge/AI-MLP%20From%20Scratch-orange.svg)

Aplikasi **End-to-End OCR (Optical Character Recognition)** yang mampu membaca gambar rumus matematika dan menyelesaikan perhitungannya secara otomatis. Proyek ini dibangun menggunakan **Neural Network (MLP) manual** tanpa menggunakan framework Deep Learning instan (seperti TensorFlow atau PyTorch).

## 🌟 Fitur Utama

* **MLP From Scratch:** Arsitektur *Neural Network* dibangun murni menggunakan NumPy (Forward/Backward Propagation manual).
* **Synthetic Data Generator:** Generator dataset otomatis yang mensimulasikan *slicing error* (padding & scaling acak) untuk melatih ketangguhan model.
* **Smart Segmentation:** Algoritma pemotongan gambar cerdas yang mampu menggabungkan simbol terpisah (seperti `=` dan `i`) serta mengurutkan posisi karakter.
* **Math Engine Integration:** Terintegrasi dengan library `SymPy` untuk menyelesaikan persamaan linear dan ekspresi matematika.
* **Multi-Model Experiment:** Menyediakan perbandingan performa antar 6 model berbeda.

## 🧠 Model Zoo (Eksperimen Dataset)

Proyek ini mengevaluasi performa AI menggunakan **6 Model Berbeda** untuk membandingkan pengaruh jenis font dan jumlah data latih terhadap akurasi.

| ID Model | Base Font | Dataset Size (per char) | Keterangan |
| :--- | :--- | :--- | :--- |
| **PH-100** | PatrickHand-Regular | 100 sampel | Model ringan, training cepat. |
| **PH-200** | PatrickHand-Regular | 200 sampel | Keseimbangan kecepatan & akurasi. |
| **PH-300** | PatrickHand-Regular | 300 sampel | Akurasi tertinggi untuk font PatrickHand. |
| **IF-100** | IndieFlower-Regular | 100 sampel | Baseline untuk font gaya tulisan tangan. |
| **IF-200** | IndieFlower-Regular | 200 sampel | Model menengah untuk IndieFlower. |
| **IF-300** | IndieFlower-Regular | 300 sampel | Model paling *robust* untuk variasi IndieFlower. |

> **Catatan:** Semua model dilatih untuk mengenali 26 kelas karakter (Angka 0-9, Operator Matematika, dan Simbol Fungsi seperti sin/cos/tan).

## 🛠️ Tech Stack

* **Bahasa:** Python 3.13+
* **Core Logic:** NumPy (Matriks & Kalkulasi Neural Network)
* **Computer Vision:** OpenCV (Preprocessing & Segmentasi)
* **Data Processing:** Pandas, Pillow (PIL)
* **Math Solver:** SymPy
* **Interface:** CLI Hybrid (Terminal Output + Tkinter File Dialog)

## 📂 Struktur Folder

```text
├── main.py                # Program Utama (OCR + Solver)
├── math_engine.py         # Modul Logika Matematika (SymPy Wrapper)
├── datasetGenV2.py        # Script Generator Data Sintetis
├── training.py            # Script Pelatihan Model (MLP Manual)
├── fonts/                 # Koleksi Font (.ttf)
│   ├── PatrickHand-Regular.ttf
│   └── IndieFlower-Regular.ttf
└── model/                 # Penyimpanan Model (.pkl)
    ├── model_dataset200.pkl
    └── label_map.pkl
```
<br>

## Instalasi

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/kenziecarlen13/ML-Project_numberOCR.git](https://github.com/kenziecarlen13/ML-Project_numberOCR.git)
    cd ML-Project_numberOCR
    ```
    ```
    cd ML-Project_numberOCR
    ```

2.  **Install Dependencies**
    ```bash
    pip install numpy opencv-python matplotlib pandas sympy tqdm pillow
    ```