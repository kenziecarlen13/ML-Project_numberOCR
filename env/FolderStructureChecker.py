import os

def print_tree(path, prefix=""):
    if not os.path.isdir(path):
        print(f"Path tidak valid: {path}")
        return

    # Ambil hanya folder
    entries = [e for e in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, e))]
    entries_count = len(entries)

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)

        # Tentukan simbol garis
        connector = "└── " if i == entries_count - 1 else "├── "
        child_prefix = prefix + ("    " if i == entries_count - 1 else "│   ")

        # Print node folder
        print(prefix + connector + entry + "/")

        # Recursive untuk anak
        print_tree(full_path, child_prefix)

# ---- Cara pakai ----
root = r"E:\kultivasi_4_tahun\Semester_5\ML\Tugas_akhir(final)\data"  # ganti dengan folder utama kamu
print(os.path.basename(root) + "/")
print_tree(root)