"""Scientific QC contact sheets and bounded pretrained-model acquisition."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_bcss import fetch


def main():
    out = ROOT / "reports/encoder_study"
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/bcss_encoder_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = {}
    for row in rows:
        pairs.setdefault(row["patient_id"], {})[row["kind"]] = row
    fig, axes = plt.subplots(len(pairs), 3, figsize=(12, 3 * len(pairs)))
    for i, (patient, pair) in enumerate(sorted(pairs.items())):
        with Image.open(ROOT / pair["image"]["path"]) as im:
            im.thumbnail((450, 450))
            rgb = np.array(im.convert("RGB"))
        with Image.open(ROOT / pair["mask"]["path"]) as im:
            mask = np.array(im.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST))
        color = np.zeros_like(rgb)
        for code, shade in [(1, (255, 0, 0)), (2, (0, 255, 0)), (3, (0, 80, 255))]:
            color[mask == code] = shade
        overlay = rgb.copy()
        selected = np.isin(mask, [1, 2, 3])
        overlay[selected] = (0.65 * rgb[selected] + 0.35 * color[selected]).astype("uint8")
        for ax, arr, label in zip(
            axes[i], [rgb, color, overlay], ["RGB", "Mask", "Overlay"], strict=True
        ):
            ax.imshow(arr)
            ax.set_title(f"{patient} {label}", fontsize=9)
            ax.axis("off")
    fig.suptitle(
        "BCSS QC: red=tumor, green=stroma, blue=lymphocytes; black=other/ignored", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out / "alignment.png", dpi=100)
    plt.close(fig)
    config = json.loads((ROOT / "configs/encoder_protocol.json").read_text())
    name = config["encoder"] + ".pth"
    path = ROOT / "data/pretrained" / name
    url = f"https://huggingface.co/TIACentre/TIAToolbox_pretrained_weights/resolve/{config['revision']}/{name}"
    fetch(url, path, max_bytes=4 * 1024**2, budget=[4 * 1024**2])
    if (
        path.stat().st_size != config["weight_bytes"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != config["weight_sha256"]
    ):
        raise ValueError("Model does not match provider size/hash")
    print(f"QC sheet generated; official weights verified ({path.stat().st_size} bytes).")


if __name__ == "__main__":
    main()
