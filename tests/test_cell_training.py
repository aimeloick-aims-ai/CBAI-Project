import csv
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

from scripts.train_cell_gnns import check_rows, main
from src.graphs.cells import cell_graph


def rows():
    return [
        {"patient_id": f"{split}_{label}", "split": split, "label": str(label)}
        for split in ("train", "validation", "test")
        for label in (0, 1)
    ]


def test_patient_leakage_is_rejected():
    manifest = rows()
    check_rows(manifest)
    manifest[-1]["patient_id"] = manifest[0]["patient_id"]
    with pytest.raises(ValueError, match="distinct patient"):
        check_rows(manifest)


def test_missing_class_is_rejected():
    with pytest.raises(ValueError, match="both binary classes"):
        check_rows(rows()[:-1])


def test_training_pipeline_writes_all_architectures_and_locks_outputs(tmp_path, monkeypatch):
    manifest_rows = rows()
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[1:5, 1:5] = 1
    mask[8:12, 8:12] = 2
    for index, row in enumerate(manifest_rows):
        graph = cell_graph(np.full((16, 16, 3), 100 + 30 * int(row["label"]), dtype=np.uint8), mask)
        path = tmp_path / f"{index}.pt"
        torch.save(graph.to_dict(), path)
        row.update(graph_path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    out = tmp_path / "outputs"
    monkeypatch.setattr(
        sys, "argv", ["train_cell_gnns", "--manifest", str(manifest), "--output", str(out)]
    )
    main()
    result = json.loads((out / "results.json").read_text())
    assert set(result["test"]) == {"GCN", "GraphSAGE", "GATv2"}
    assert len(list(out.glob("*.pt"))) == 9
    assert (out / "test_opened.json").exists()
    with pytest.raises(FileExistsError):
        main()
