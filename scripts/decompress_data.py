import gzip
import shutil
from pathlib import Path

src = Path("data/variant_summary.txt.gz")
dst = Path("data/variant_summary.txt")
if not dst.exists():
    print(f"[+] Decompressing {src} -> {dst} ...")
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("[+] Done.")
else:
    print("[+] Decompressed file already exists.")
