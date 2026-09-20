"""Reproducible image + optional instance maps -> candidate cell graph and QC."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.segmentation import find_boundaries

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.graphs.cells import cell_graph
from src.segmentation.nuclei import instance_metrics, validate_instances, watershed_baseline


def load_mask(path):
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--segmenter", choices=["watershed", "hovernet"], default="watershed")
    parser.add_argument("--weights", type=Path, help="Local HoVer-Net fast PanNuke checkpoint")
    parser.add_argument(
        "--instances", type=Path, help="Predicted instance IDs in NPY or integer TIFF/PNG"
    )
    parser.add_argument(
        "--reference", type=Path, help="Independent annotated instance IDs, same pixel grid"
    )
    parser.add_argument(
        "--mpp", type=float, help="Known source micrometers per pixel; do not guess"
    )
    parser.add_argument(
        "--radius", type=float, default=50, help="Micrometers if mpp supplied, otherwise pixels"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=768,
        help="Central native-resolution crop; 0 means full image",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/cell_graph_smoke")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.segmenter == "hovernet" and args.instances:
        parser.error("Choose either an external instance map or HoVer-Net inference")
    if args.segmenter == "hovernet" and (args.weights is None or not args.weights.is_file()):
        parser.error("HoVer-Net requires --weights pointing to the downloaded checkpoint")
    if args.crop_size < 0:
        parser.error("crop-size must be nonnegative")
    source_row = None
    if args.image is None:
        with (ROOT / "data/bracs_expanded_study_manifest.csv").open() as f:
            source_row = next(r for r in csv.DictReader(f) if r["proposed_split"] == "train")
        args.image = ROOT / source_row["path"]
    digest = hashlib.sha256(args.image.read_bytes()).hexdigest()
    if source_row and digest != source_row["sha256"]:
        raise ValueError("Source checksum mismatch")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Preserve existing outputs; choose a new output directory")
    with Image.open(args.image) as image:
        width, height = image.size
        size = args.crop_size or max(width, height)
        left, top = max(0, (width - size) // 2), max(0, (height - size) // 2)
        box = (left, top, min(width, left + size), min(height, top + size))
        rgb = np.array(image.convert("RGB").crop(box))
    provenance = {str(args.image): digest}

    def crop_mask(path):
        mask = validate_instances(load_mask(path), (height, width))
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return mask[box[1] : box[3], box[0] : box[2]]

    if args.instances:
        instances = crop_mask(args.instances)
    elif args.segmenter == "hovernet":
        from src.segmentation.hovernet import segment_pretrained

        provenance[str(args.weights)] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        instances = segment_pretrained(rgb, args.weights)
    else:
        instances = watershed_baseline(rgb)
    reference = crop_mask(args.reference) if args.reference else None
    graph = cell_graph(rgb, instances, mpp=args.mpp, radius=args.radius)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "instances.npy", instances, allow_pickle=False)
    torch.save(graph.to_dict(), args.output / "graph.pt")
    degree = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes)
    report = {
        "scope": "engineering integration only; not validated nuclear segmentation or diagnostic performance",
        "segmentation": "external instance map" if args.instances else args.segmenter,
        "source": provenance,
        "manifest_row": source_row,
        "crop_xyxy": box,
        "original_size_wh": [width, height],
        "resized": False,
        "mpp": args.mpp,
        "coordinate_units": graph.coordinate_units,
        "radius": args.radius,
        "nodes": graph.num_nodes,
        "directed_edges": graph.num_edges,
        "isolated_nodes": int((degree == 0).sum()),
        "feature_names": graph.feature_names,
        "reference_metrics": instance_metrics(instances, reference)
        if reference is not None
        else None,
        "segmentation_validated": False,
        "boundary_policy": "retains crop-truncated nuclei; metrics apply to cropped masks only",
        "training": "not performed; existing patch-classifier weights incompatible with these features",
    }
    (args.output / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original native-resolution crop")
    overlay = rgb.copy()
    overlay[find_boundaries(instances)] = [0, 255, 0]
    axes[1].imshow(overlay)
    axes[1].set_title("Candidate nuclei - not validated")
    axes[2].imshow(rgb)
    positions = graph.pos.numpy() / (args.mpp or 1.0)
    for a, b in graph.edge_index.T.tolist():
        if a < b:
            axes[2].plot(
                positions[[a, b], 0], positions[[a, b], 1], color="cyan", alpha=0.25, lw=0.4
            )
    axes[2].scatter(positions[:, 0], positions[:, 1], s=3, c="yellow")
    axes[2].set_title(f"Candidate cell graph: {graph.num_nodes} nodes")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output / "overlay.png", dpi=130)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
