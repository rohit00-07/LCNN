import os
import torchaudio

root = "data"

bad_files = []

for folder in ["bona_fide", "spoof"]:
    full_path = os.path.join(root, folder)
    print(f"\nChecking folder: {full_path}")
    for fname in os.listdir(full_path):
        if not fname.lower().endswith(".wav"):
            continue
        path = os.path.join(full_path, fname)
        try:
            torchaudio.load(path)
        except Exception as e:
            print(f"❌ BAD FILE: {path}")
            bad_files.append(path)

print("\nTotal bad files found:", len(bad_files))