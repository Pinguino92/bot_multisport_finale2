import os, requests

BASE_DIR = "data"
DESTS = {
    "tennis": os.path.join(BASE_DIR,"tennis"),
    "soccer": os.path.join(BASE_DIR,"soccer"),
    "basketball": os.path.join(BASE_DIR,"basketball"),
}
for d in DESTS.values():
    os.makedirs(d, exist_ok=True)

def dl_file(file_id, dest):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    out = os.path.join(dest, file_id + ".csv")
    try:
        r = requests.get(url, allow_redirects=True, timeout=120)
        r.raise_for_status()
        with open(out,"wb") as f: f.write(r.content)
        print("✅", out)
    except Exception as e:
        print("❌ Errore", file_id, e)

# === Inserisci qui i tuoi ID Google Drive (uno per ogni file CSV) ===
tennis_ids = ["1HoG667ZVQPWzSvqJyZLDhlibopM3KFkW"]
soccer_ids = ["1IwH4OWw8K7d6lA6L_yOHDv0sPWzAjB7R"]
basket_ids = ["1jW1s1ZsMPG9nqRSv7eMncSeYxGl7zaJ1"]

print("⬇️ Scarico CSV Tennis...")
for fid in tennis_ids: dl_file(fid, DESTS["tennis"])

print("⬇️ Scarico CSV Soccer...")
for fid in soccer_ids: dl_file(fid, DESTS["soccer"])

print("⬇️ Scarico CSV Basket...")
for fid in basket_ids: dl_file(fid, DESTS["basketball"])

print("🎉 Download CSV completato")
