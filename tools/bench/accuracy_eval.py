#!/usr/bin/env python3
"""
Evaluate API for top-1/top-5 on a small validation set.
Assumes validation images in tests/valset/ with corresponding .txt files containing the integer label.
Usage:
  python accuracy_eval.py --url http://localhost:8000/predict --val-dir tests/valset
"""
import argparse, base64, requests, json
from pathlib import Path
from tqdm import tqdm

def predict(api_url, img_path):
    b = base64.b64encode(img_path.read_bytes()).decode()
    r = requests.post(api_url, json={"image": b}, timeout=30)
    return r.json()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--val-dir", required=True)
    args = p.parse_args()

    val_dir = Path(args.val_dir)
    imgs = list(val_dir.glob("*.jpg"))
    top1 = 0
    top5 = 0
    total = 0
    results = []
    for img in tqdm(imgs):
        label_file = img.with_suffix(".txt")
        if not label_file.exists():
            print("Skipping (no label):", img)
            continue
        true = int(label_file.read_text().strip())
        resp = predict(args.url, img)
        pred = resp.get("predicted_class_index")
        top5_list = [p["class_index"] for p in resp.get("top5_predictions",[])]
        total += 1
        if pred == true:
            top1 += 1
        if true in top5_list:
            top5 += 1
        results.append({"image": str(img.name), "true": true, "pred": pred, "top5": top5_list})
    out = {"total": total, "top1": top1/total if total else None, "top5": top5/total if total else None, "results": results}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
