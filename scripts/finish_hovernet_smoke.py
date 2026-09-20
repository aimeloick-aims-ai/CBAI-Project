"""Wait for an existing acquisition, then run one native-resolution integration example."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    weights = ROOT / "data/pretrained/hovernet_fast-pannuke.pth"
    receipt = weights.with_suffix(".receipt.json")
    status = ROOT / "reports/hovernet_job.json"

    def update(state, **extra):
        status.write_text(json.dumps({"status": state, **extra}, indent=2))

    update("waiting_for_existing_download")
    deadline = time.monotonic() + 7200
    while not (weights.exists() and receipt.exists()):
        if time.monotonic() >= deadline:
            update("failed", reason="Download did not complete within two hours")
            return
        time.sleep(10)
    record = json.loads(receipt.read_text())
    if hashlib.sha256(weights.read_bytes()).hexdigest() != record["sha256"]:
        update("failed", reason="Checksum mismatch")
        return
    update("running_inference")
    with (ROOT / "reports/hovernet_smoke.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_cell_graph.py"),
                "--segmenter",
                "hovernet",
                "--weights",
                str(weights),
                "--crop-size",
                "256",
                "--output",
                str(ROOT / "reports/cell_graph_hovernet"),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    update(
        "complete" if result.returncode == 0 else "failed",
        exit_code=result.returncode,
        log="reports/hovernet_smoke.log",
        output="reports/cell_graph_hovernet",
        scientific_segmentation_validation=False,
    )


if __name__ == "__main__":
    main()
