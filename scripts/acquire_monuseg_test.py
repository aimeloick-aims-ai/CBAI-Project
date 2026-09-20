"""Acquire official public MoNuSeg test archive; keep source and local checksums."""

import hashlib
import json
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "data/raw/MoNuSeg_test"
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "official_test.zip"
    file_id = "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw"
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    if not archive.exists():
        session = requests.Session()
        response = session.get(url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            page = BeautifulSoup(response.text, "html.parser")
            form = page.find("form", id="download-form")
            if form is None:
                raise RuntimeError("Public download unavailable; no archive returned")
            params = {
                i["name"]: i.get("value", "") for i in form.find_all("input") if i.get("name")
            }
            response = session.get(form["action"], params=params, stream=True, timeout=(30, 120))
            response.raise_for_status()
        temporary = out / "official_test.part"
        with temporary.open("wb") as f:
            for block in response.iter_content(1024 * 1024):
                f.write(block)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Response is not a ZIP archive")
        temporary.replace(archive)
    records = []
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if item.is_dir() or Path(item.filename).suffix.lower() not in {
                ".tif",
                ".tiff",
                ".xml",
                ".png",
            }:
                continue
            destination = (out / item.filename).resolve()
            if not destination.is_relative_to(out.resolve()):
                raise ValueError("Archive member outside data directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(item)
            destination.write_bytes(data)
            records.append(
                {
                    "path": str(destination.relative_to(ROOT)),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "source_page": "https://monuseg.grand-challenge.org/Data/",
                "source_file": file_id,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "checksum_kind": "locally_computed_not_provider_digest",
                "files": records,
            },
            indent=2,
        )
    )
    print(f"MoNuSeg official test archive verified: {len(records)} extracted files")


if __name__ == "__main__":
    main()
