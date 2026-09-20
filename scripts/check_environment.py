"""CPU software checks only; generated arrays are never research observations."""

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "torch": "torch",
    "torch-geometric": "torch_geometric",
    "openslide-python": "openslide",
    "openslide-bin": "openslide_bin",
    "tiatoolbox": "tiatoolbox",
    "scikit-learn": "sklearn",
    "pandas": "pandas",
    "hydra-core": "hydra",
    "pytest": "pytest",
    "pyarrow": "pyarrow",
}


def check_gnn(architecture, config):
    import torch
    from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool

    torch.manual_seed(config["seed"])
    layer_type = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GATv2": GATv2Conv}[architecture]
    layer = layer_type(config["input_features"], config["hidden_features"])
    head = torch.nn.Linear(config["hidden_features"], config["output_classes"])
    x = torch.randn(config["nodes"], config["input_features"], requires_grad=True)
    nodes = torch.arange(config["nodes"])
    edge_index = torch.stack((nodes, nodes.roll(1)))
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    batch = torch.zeros(config["nodes"], dtype=torch.long)
    logits = head(global_mean_pool(layer(x, edge_index).relu(), batch))
    if logits.shape != (1, config["output_classes"]):
        raise AssertionError("Incorrect graph classification output shape")
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(1, dtype=torch.long))
    loss.backward()
    if not torch.isfinite(logits).all() or x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Non-finite output or missing input gradient")
    if x.grad.abs().sum() == 0:
        raise AssertionError("Zero input gradient; interventions cannot be checked")
    return {"status": "pass", "device": "cpu", "fixture": "synthetic_ring_no_patient"}


def check_slide(path=None):
    import numpy as np
    import openslide
    import tifffile

    with tempfile.TemporaryDirectory(prefix="gcia_wsi_") as directory:
        if path is None:
            path = Path(directory) / "software_fixture.tiff"
            pixels = np.zeros((64, 64, 4), dtype=np.uint8)
            pixels[:, :, 0] = 127
            pixels[:, :, 3] = 255
            tifffile.imwrite(
                path,
                pixels,
                tile=(16, 16),
                photometric="rgb",
                extrasamples="unassalpha",
                compression="deflate",
                metadata=None,
            )
            kind = "synthetic_tiled_tiff_no_patient"
        else:
            kind = "provided_wsi"
        with openslide.OpenSlide(str(path)) as slide:
            region = slide.read_region((0, 0), 0, (16, 16)).convert("RGB")
            if region.size != (16, 16):
                raise AssertionError("Unexpected region dimensions")
            if kind.startswith("synthetic") and region.getpixel((0, 0)) != (127, 0, 0):
                raise AssertionError("Incorrect pixel decoded by OpenSlide")
            return {"status": "pass", "fixture": kind, "dimensions": list(slide.dimensions)}


def run_check(function, *args):
    try:
        return function(*args)
    except Exception as error:  # noqa: BLE001 - Record plugin failures in the diagnostic report.
        return {"status": "fail", "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/smoke_test.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/environment_check.json")
    parser.add_argument("--wsi", type=Path)
    args = parser.parse_args()
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "scope": "software_smoke_only_not_scientific_validation",
        "packages": {},
        "checks": {},
    }
    for distribution, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            report["packages"][distribution] = {
                "status": "pass",
                "version": importlib.metadata.version(distribution),
            }
        except Exception as error:  # noqa: BLE001 - Continue checking other package imports.
            report["packages"][distribution] = {"status": "fail", "error": str(error)}
    try:
        import torch
        import yaml

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu_count"] = torch.cuda.device_count()
        for architecture in config["architectures"]:
            report["checks"][architecture] = run_check(check_gnn, architecture, config)
        report["checks"]["openslide_fixture"] = run_check(check_slide)
        if args.wsi:
            report["checks"]["real_wsi"] = run_check(check_slide, args.wsi)
        else:
            report["checks"]["real_wsi"] = {
                "status": "not_run",
                "reason": "No authorized research WSI supplied; required before phase 4",
            }
    except Exception as error:  # noqa: BLE001 - Always write the failure report before exiting.
        report["checks"]["runtime"] = {"status": "fail", "error": str(error)}
    failed = any(
        item["status"] == "fail"
        for group in (report["packages"], report["checks"])
        for item in group.values()
    )
    report["software_smoke_passed"] = not failed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
