"""Download / locate the datasets (run this on YOUR machine with internet).
  python download_data.py --search                       # list mmWave outdoor DeepMIMO scenarios
  python download_data.py --scenario <name>              # fetch one DeepMIMO scenario
  python download_data.py --lumos-url <direct .zip/.csv url>   # optional direct link
"""
import argparse, os, zipfile, requests
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--scenario")
    ap.add_argument("--lumos-url")
    a = ap.parse_args()
    os.makedirs(os.path.join(DATA, "lumos5g"), exist_ok=True)
    if a.search:
        import deepmimo as dm
        res = dm.search({"bands": ["mmW"], "environment": "outdoor"})
        print(res)
    if a.scenario:
        import deepmimo as dm
        dm.download(a.scenario)      # dm.load() also offers to download if missing
        print("scenarios dir:", dm.get_scenarios_dir())
    if a.lumos_url:
        out = os.path.join(DATA, "lumos5g", os.path.basename(a.lumos_url.split("?")[0]) or "lumos.zip")
        r = requests.get(a.lumos_url, timeout=120); r.raise_for_status()
        open(out, "wb").write(r.content)
        if out.endswith(".zip"):
            zipfile.ZipFile(out).extractall(os.path.join(DATA, "lumos5g"))
        print("saved to", os.path.join(DATA, "lumos5g"))
    if not (a.search or a.scenario or a.lumos_url):
        print("Lumos5G manual download: open https://lumos5g.umn.edu/ -> 'Download (lumos5G-v1.0) Dataset'\n"
              "(or IEEE DataPort, free login), unzip the CSV files into data/lumos5g/")

if __name__ == "__main__":
    main()
