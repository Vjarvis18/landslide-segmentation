import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

"""Landslide4Sense lives on Zenodo; download manually or with zenodo_get, then unzip to data/raw/."""
EXPECTED = ["data/raw/TrainData/img", "data/raw/TrainData/mask"]

if __name__ == "__main__":
    print("1) Download the Landslide4Sense dataset from its Zenodo record (search 'Landslide4Sense Zenodo').")
    print("   e.g.  pip install zenodo_get && zenodo_get <RECORD_ID> -o data/raw")
    print("2) Unzip so that the following folders exist:")
    ok = True
    for p in EXPECTED:
        n = len(os.listdir(p)) if os.path.isdir(p) else 0
        print(f"   {p:32s} {'OK' if n else 'MISSING'} ({n} files)")
        ok &= n > 0
    print("Ready." if ok else "Not ready yet.")
