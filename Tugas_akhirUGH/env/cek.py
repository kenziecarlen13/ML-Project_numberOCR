import pandas as pd

# Membaca file CSV
df = pd.read_csv('laporan_prediksi.csv')

# 1. Menghitung Akurasi Total per Model
acc_100 = (df['Cek_100'] == '✅').mean() * 100
acc_200 = (df['Cek_200'] == '✅').mean() * 100
acc_300 = (df['Cek_300'] == '✅').mean() * 100

# 2. Analisis Peningkatan (Dimana 100 Salah tapi 300 Benar)
# Ini membuktikan 'belajar' itu ada gunanya
improvement = df[(df['Cek_100'] == '❌') & (df['Cek_300'] == '✅')]
num_improved = len(improvement)

# 3. Analisis Kesalahan Fatal (Dimana 300 masih Salah)
errors_300 = df[df['Cek_300'] == '❌']
top_errors = errors_300['Kunci Jawaban'].value_counts().head(5)

# 4. Kebingungan Terbesar (Confusion)
# Apa yang paling sering disalahartikan oleh Model 300?
most_confused = errors_300.groupby(['Kunci Jawaban', 'Pred_300']).size().sort_values(ascending=False).head(3)

print(f"Akurasi Model 100: {acc_100:.2f}%")
print(f"Akurasi Model 200: {acc_200:.2f}%")
print(f"Akurasi Model 300: {acc_300:.2f}%")
print("-" * 30)
print(f"Jumlah soal yang gagal dijawab Model 100 tapi SUKSES di Model 300: {num_improved} soal")
print("-" * 30)
print("5 Karakter Tersulit bagi Model 300:")
print(top_errors)
print("-" * 30)
print("Top 3 Kesalahan Spesifik (Kunci -> Tebakan Salah):")
print(most_confused)