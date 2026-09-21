"""Exploratory BRACS measured-area positive control; reuse existing graph tensors."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.synthetic.run import execute
from src.evaluation import load_config
from src.graph import load_bracs


def main():
    config = load_config(ROOT)
    splits, metadata = load_bracs(ROOT, config["bracs"])
    metadata["scope"] = "development/exploratory; historical test already explored"
    metadata["unavailable_concepts"] = config["bracs"]["missing_evidence"]
    execute("bracs", splits, metadata, config)


if __name__ == "__main__":
    main()
