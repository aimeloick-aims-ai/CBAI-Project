"""Fixed evaluation on all official MoNuSeg test images; no model fitting."""

import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.draw import polygon
from skimage.segmentation import find_boundaries
from tiatoolbox.models.architecture import get_pretrained_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.segmentation.hovernet import infer_tiled
from src.segmentation.nuclei import instance_metrics, watershed_baseline

OUT = ROOT / "reports/segmentation_evaluation"


def xml_instances(path, shape):
    """Rasterize XML X/Y vertices directly on the image grid; no rescaling."""
    mask = np.zeros(shape, dtype=np.int32)
    overlap = 0
    for index, region in enumerate(ET.parse(path).getroot().iter("Region"), start=1):
        vertices = [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in region.iter("Vertex")]
        if len(vertices) < 3:
            raise ValueError("Nuclear polygon needs at least three vertices")
        x, y = np.asarray(vertices).T
        rr, cc = polygon(y, x, shape=shape)
        overlap += int(np.count_nonzero(mask[rr, cc]))
        mask[rr, cc] = index
    if not mask.any():
        raise ValueError("No nuclear annotations found")
    return mask, overlap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = ROOT / "data/raw/MoNuSeg_test/receipt.json"
    files = json.loads(receipt.read_text())["files"]
    images = {
        Path(r["path"]).stem: r
        for r in files
        if Path(r["path"]).suffix.lower() in {".tif", ".tiff", ".png"}
    }
    annotations = {
        Path(r["path"]).stem: r for r in files if Path(r["path"]).suffix.lower() == ".xml"
    }
    if set(images) != set(annotations) or len(images) != 14:
        raise ValueError("Expected 14 complete official image/XML pairs")
    weights = ROOT / "data/pretrained/hovernet_fast-pannuke.pth"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != json.loads(weights.with_suffix(".receipt.json").read_text())["sha256"]:
        raise ValueError("Model checksum mismatch")
    protocol = {
        "dataset": "MoNuSeg official 2018 test, all 14 full images",
        "models": ["hovernet_fast-pannuke", "watershed_baseline"],
        "tuning": "none; frozen existing parameters",
        "resampling": "none; native image pixels",
        "xml_rasterization": "skimage.draw.polygon using XML X/Y directly; later polygon wins on overlap",
        "iou_matching": "one-to-one IoU strictly >0.5",
        "weight_sha256": digest,
        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independence": "Overlap of source TCGA slides with PanNuke pretraining has not been ruled out; do not call independent external validation",
        "scope": "multi-organ benchmark evaluation; does not certify BRACS segmentation or physical scale",
        "acceptance": "no retrospective pass threshold; report every image and both methods",
    }
    with (OUT / "protocol.json").open("x") as f:
        json.dump(protocol, f, indent=2)
    torch.set_num_threads(2)
    model, _ = get_pretrained_model("hovernet_fast-pannuke", pretrained_weights=weights)
    model.eval().requires_grad_(False)
    metrics = []
    for name in sorted(images):
        for record in (images[name], annotations[name]):
            if hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("Evaluation source checksum mismatch")
        with Image.open(ROOT / images[name]["path"]) as image:
            rgb = np.array(image.convert("RGB"))
        reference, overlap = xml_instances(ROOT / annotations[name]["path"], rgb.shape[:2])
        predictions = {"hovernet": infer_tiled(model, rgb), "watershed": watershed_baseline(rgb)}
        np.save(OUT / f"{name}_reference.npy", reference, allow_pickle=False)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, title, mask in zip(
            axes,
            ["Original", "Reference", "HoVer-Net", "Watershed"],
            [None, reference, predictions["hovernet"], predictions["watershed"]],
            strict=True,
        ):
            overlay = rgb.copy()
            if mask is not None:
                overlay[find_boundaries(mask)] = [0, 255, 0]
            ax.imshow(overlay)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_comparison.png", dpi=120)
        plt.close(fig)
        for method, prediction in predictions.items():
            np.save(OUT / f"{name}_{method}.npy", prediction, allow_pickle=False)
            metrics.append(
                {
                    "image": name,
                    "method": method,
                    "reference_overlap_pixels": overlap,
                    "reference_instances": len(np.unique(reference)) - int(0 in reference),
                    "predicted_instances": len(np.unique(prediction)) - int(0 in prediction),
                    **instance_metrics(prediction, reference),
                }
            )
        with (OUT / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        print(f"Evaluated {name} ({len(metrics) // 2}/14)", flush=True)
    summary = {"images": 14, "protocol": protocol, "methods": {}}
    for method in ("hovernet", "watershed"):
        selected = [r for r in metrics if r["method"] == method]
        summary["methods"][method] = {
            key: float(np.mean([r[key] for r in selected if r[key] is not None]))
            for key in (
                "foreground_dice",
                "detection_f1",
                "segmentation_quality",
                "panoptic_quality",
            )
        }
        summary["methods"][method]["reference_instances"] = sum(
            r["reference_instances"] for r in selected
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
