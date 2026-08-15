"""Download ClinVar variant_summary.txt.gz."""
import urllib.request
from pathlib import Path

URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT = DATA_DIR / "variant_summary.txt.gz"

def main():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Downloading ClinVar to {OUTPUT} ...")
    urllib.request.urlretrieve(URL, OUTPUT)
    print("Done.")

if __name__ == "__main__":
    main()
