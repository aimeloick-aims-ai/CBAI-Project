"""Build a graph from a real image and an existing instance mask; no segmentation inference."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.graph import cell_graph


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--instances", type=Path, required=True)
    p.add_argument("--mpp", type=float, required=True, help="Verified micrometers per pixel; do not guess")
    p.add_argument("--radius", type=float, required=True, help="Radius in micrometers")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with Image.open(args.image) as image:
        rgb = np.array(image.convert("RGB"))
    graph = cell_graph(rgb, np.load(args.instances, allow_pickle=False), mpp=args.mpp, radius=args.radius)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.output)


if __name__ == "__main__":
    main()
