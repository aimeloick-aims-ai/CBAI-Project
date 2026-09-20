"""Download the official checkpoint once, pinning its revision and verifying LFS SHA256."""

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    repo = "TIACentre/TIAToolbox_pretrained_weights"
    name = "hovernet_fast-pannuke.pth"
    path = ROOT / "data/pretrained" / name
    receipt = path.with_suffix(".receipt.json")
    if path.exists():
        if not receipt.exists():
            raise ValueError("Existing checkpoint lacks a receipt; verify manually")
        saved = json.loads(receipt.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        print("Checkpoint already verified")
        return
    response = requests.get(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=60)
    response.raise_for_status()
    info = response.json()
    metadata = next(row for row in info["siblings"] if row["rfilename"] == name)
    expected = metadata["lfs"]["sha256"]
    url = f"https://huggingface.co/{repo}/resolve/{info['sha']}/{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    digest, size = hashlib.sha256(), 0
    with requests.get(url, stream=True, timeout=(30, 120)) as data:
        data.raise_for_status()
        with temporary.open("wb") as output:
            for block in data.iter_content(1024 * 1024):
                output.write(block)
                digest.update(block)
                size += len(block)
    if digest.hexdigest() != expected or size != metadata["lfs"]["size"]:
        raise ValueError("Download failed provider checksum/size validation")
    temporary.replace(path)
    receipt.write_text(
        json.dumps(
            {
                "source": url,
                "revision": info["sha"],
                "sha256": expected,
                "bytes": size,
                "checksum_kind": "provider_LFS_SHA256",
                "license_metadata": info.get("cardData", {}).get("license"),
            },
            indent=2,
        )
    )
    print(f"Verified {name}: {size} bytes")


if __name__ == "__main__":
    main()
